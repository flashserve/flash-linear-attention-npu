# A2：experimental 功能与正式算子隔离验证方案

当前真实实验对象仅有 RWKV8；清理后结果见 [2026-11-06 回归](../../validation-rwkv-only-20261106.md)。
下文环境描述和第 7 节保留先前验证记录，不能将旧制品结果计入当前制品的通过数。

状态：2026-11-03 A2 定向集成验证完成，结果及未覆盖项见
[A2 验证记录](../../validation-a2-20261103.md)。下表是验证设计全集，不能将未执行项计入通过数。
本轮只针对 A2（ascend910b），不复用上一轮 A5 wheel/run，不扩展到 A5 性能或正式发布验收。

## 1. 要证明的行为

1. experimental 开关确实选择实验闭包，单算子与全量包都能在 A2 执行正确的实验算子。
2. `.run`、wheel、`.pth`、显式共享库加载及不同导入顺序，不会让实验调用静默落到正式库。
3. 正式与实验包允许文件共存；公共 Python 入口同进程混用必须被拒绝。切换使用新进程。
4. 实验安装、卸载、错误装载和异常参数，不改写原有正式包；正式算子的运行结果不变。

`import` 成功、发现符号、NPU 实际执行、数值正确四项分别记录。加载检查不替代精度检查。
失败用例必须匹配预期异常类型和原因，不能把进程崩溃、超时、缺 torch/CANN 当成隔离成功。

## 2. 环境和制品准备

记录源码 commit + dirty/source hash、A2 型号、物理/逻辑卡映射、CANN、torch/torch_npu、ATK、GCC，
以及每个 wheel/run 的 SHA256、双清单与 launcher 指纹。`npu-smi info` 留档；脚本另外核验所选
逻辑设备名称包含 910B，否则不执行装载矩阵。

本轮目标为 `z00879345@179.178.1.2`，工作目录 `/home/z00879345/`，conda 环境 `zzy-cann900`。
用户提供的初始化脚本为 `/home/z00879345/Ascend/9.0.0/ascend-toolkit/set_env.sh`；
经 `share/info/asc-devkit/version.info` 确认该路径实际为 CANN 9.0.0；本轮按指定初始化命令使用此版本。
设备为 910B2 / aarch64，Python 3.11.18、torch 2.12.0+cpu、torch_npu 2.12.0、ATK 26.9.8。
本轮使用空闲卡 5。测试根目录为 `/home/z00879345/experimental-a2-20261103-9R2eAj`。

需要四种本轮构建的制品：

| 制品 | 用途 |
| --- | --- |
| 正式完整 wheel 或至少含 `chunk_fwd_o` 的匹配测试 wheel | 正式功能基线与正式 Python/launcher/OPP |
| 正式 `chunk_fwd_o` 的同 ABI overlay | 验证正式已有 overlay 能力没有被实验模式破坏 |
| RWKV8 单算子实验 `.run` + wheel | 单算子集合、未选算子拒绝、实际精度 |
| 全量实验 `.run` + wheel（当前只有 RWKV8） | 全量选择入口、共存与精确注册集合；多算子规则由单元测试夹具覆盖 |

示例构建入口（按先初始化 CANN/conda 的约定执行）：

```bash
export FLA_NPU_DISABLE_PTH=1
CI_SOC=ascend910b CI_EXPERIMENTAL_OPS=fused_recurrent_rwkv8 CI_JOBS=4 bash ci/run_experimental_validation.sh
# 先归档单算子制品，再构建全量；两个构建不能同时使用同一 build 目录。
CI_SOC=ascend910b CI_EXPERIMENTAL_OPS=all CI_JOBS=4 bash ci/run_experimental_validation.sh
```

创建一次性测试环境，使用 `pip --no-deps` 安装相同版本依赖下的匹配 wheel，不覆盖用户当前环境。
`experimental_only`、`stable_only`、`combined`、`single`、`overlay` 五个 profile 的 Python 必须
实际对应指定安装状态；不能把 `PYTHONPATH` 指向源码树来冒充已安装 wheel。
若采用共享依赖的 venv，需检查母环境 `.pth` 和正式包没有意外泄漏到“仅实验”profile。

另外准备专用 `experimental_atk` 环境。ATK worker 可能先导入 torch_npu，因此在该环境自己
的 site-packages 中增加仅用于验证的 `00_experimental_atk_bootstrap.pth`，内容为：

```python
import fla_npu_experimental
```

启动 Python/ATK 之前 source 该 wheel 内实验 vendor 的 `set_env.bash`，设 `FLA_NPU_DISABLE_PTH=1`。
这个 bootstrap 不随产品 wheel 发布，也不能放进 28 项装载顺序测试的五个 profile，否则会改变
被测导入顺序。不得为方便 ATK 启动而删除 loader 的时序检查。

## 3. 构建与包功能

| ID | 操作 | 通过标准 |
| --- | --- | --- |
| B01 | 分别运行 `build.sh --list-ops`、`--experimental --list-ops` | 正式列表不含 RWKV8 实验名；实验列表当前仅含 RWKV8 |
| B02 | A2 单算子构建、安装、审计、wheel 加载 | 清单与实际包仅含 RWKV8，FP16/BF16/FP32 变体齐全，注册集合精确一致 |
| B03 | A2 全量实验构建 | 当前与 RWKV8 单算子选择相同，没有正式算法源/adapter/schema 混入；不宣称真实多算子覆盖 |
| B04 | 缺失实验依赖、循环、common 缺文件、越界链接 | 明确失败，没有回退到正式树；复用 `ci/tests/test_experimental_cmake_isolation.py` |
| B05 | 正式 wheel 入口传入实验参数；实验构建请求正式算子 | 在清理/替换输出前拒绝；既有输出哨兵文件 hash 不变 |
| B06 | 未开启 experimental 的同环境定向构建与原 main 对照 | 文件集合、ABI、算子能力和运行输出不回归；二进制可复现时再做字节比较 |

## 4. 安装、卸载及文件隔离

所有写操作只在一次性环境、制品副本和新目录中执行。原始正式安装只读，前后做全文件 hash。

| ID | 操作 | 通过标准 |
| --- | --- | --- |
| I01 | 实验 `.run --install-path=<绝对空目录>` | 只有实验 vendor；来源/ABI 清单及实际文件通过审计 |
| I02 | 实验 `.run` 无参数、`--install`、`--full`、`--cann`、`--overlay` | 按不支持参数拒绝；正式安装和目标目录不变 |
| I03 | 指向正式 wheel/OPP/overlay、非空目录、链接别名、相对路径 | 明确拒绝；哨兵及正式内容 hash 不变 |
| I04 | 同一 `.run` 二次安装至已有目录，再安装至另一空目录 | 原目录拒绝并保持不变；新目录成功且内容一致 |
| I05 | 两种 pip 顺序：正式→实验、实验→正式 | distribution/文件无冲突；L02/L03 和正式/实验数值基线均通过 |
| I06 | 在副本环境卸载实验 wheel、重装实验 wheel | 正式包 hash、导入和控制算子输出不变；卸载后实验不能导入，重装后恢复 |
| I07 | 用正式管理器把实验 vendor 作为 overlay 输入 | `python -m fla_npu_opp_manage --overlay --source <实验vendor>` 必须拒绝身份/ABI不匹配；不能留下半安装 overlay |
| I08 | 正式合法 overlay 的安装/移除；同时安装实验 wheel | 正式 overlay 优先级正确、base 保留；实验独立进程不消费正式 overlay |

I02–I04 已用本轮实际 A2 `.run` 验证。I05 反向 pip 顺序补测了两种装载，未重复数值测试；
I06 三阶段数值检查覆盖安装前、共存和卸载后，重装后仅检查安装成功。
I08 使用同 ABI、同内容的正式包副本验证管理和加载顺序，未验证替换 kernel 后的数值差异。

## 5. 装载隔离：28 个可执行用例

[`cases.json`](cases.json) 是可执行清单；[`probe.py`](probe.py) 在每个新进程内记录实际模块路径、
dispatcher 名称、环境及 `/proc/self/maps`，[`run.py`](run.py) 做判定和正式目录前后 hash。

| ID | 场景 | 预期 |
| --- | --- | --- |
| L01 | 只有实验 wheel | 成功；仅对应包和注册、库路径 |
| L02 | 同环境两包：仅加载 experimental | 成功；仅对应包和注册、库路径 |
| L03 | 同环境两包：正式 wheel 自动 .pth | 成功；仅对应包和注册、库路径 |
| L04 | 只有正式 wheel | 成功；仅对应包和注册、库路径 |
| L05 | 正式 wheel 显式加载 | 成功；仅对应包和注册、库路径 |
| L06 | 正式 wheel + 合法 overlay | 成功；仅对应包和注册、库路径 |
| L07 | 未关闭稳定 .pth | 明确拒绝（核对异常原因） |
| L08 | Python 启动后才关闭 .pth | 明确拒绝（核对异常原因） |
| L09 | 搜索路径：正式在前 | 明确拒绝（核对异常原因） |
| L10 | 搜索路径：实验在前 | 明确拒绝（核对异常原因） |
| L11 | 遗留正式库环境变量不劫持实验库 | 成功；仅对应包和注册、库路径 |
| L12 | 先正式后实验 | 明确拒绝（核对异常原因） |
| L13 | 先实验后正式 | 明确拒绝（核对异常原因） |
| L14 | 先 torch_npu 后实验 | 明确拒绝（核对异常原因） |
| L15 | 先原生装载正式 launcher | 明确拒绝（核对异常原因） |
| L16 | 先原生装载正式 opapi | 明确拒绝（核对异常原因） |
| L17 | 实验之后原生装载正式 launcher，再次检查 | 明确拒绝（核对异常原因） |
| L18 | 加载后切换 OPP 路径 | 明确拒绝（核对异常原因） |
| L19 | 加载后修改来源清单（副本） | 明确拒绝（核对异常原因） |
| L20 | 替换 host 库（副本） | 明确拒绝（核对异常原因） |
| L21 | 缺失 ABI 清单（副本） | 明确拒绝（核对异常原因） |
| L22 | 替换为正式 launcher（副本） | 明确拒绝（核对异常原因） |
| L23 | 源码指纹不匹配（副本） | 明确拒绝（核对异常原因） |
| L24 | wheel 拼外部实验 OPP | 明确拒绝（核对异常原因） |
| L25 | RWKV8 wheel 调用未注册的测试名称 | 明确拒绝（核对异常原因） |
| L26 | launcher 与 OPP 算子集合不一致（副本） | 明确拒绝（核对异常原因） |
| L27 | host 库链接越出 vendor（副本） | 明确拒绝（核对异常原因） |
| L28 | 修改 Python 包（副本） | 明确拒绝（核对异常原因） |

L15–L17 不是要把 Python 包做成动态链接安全沙箱：直接 `torch.ops.load_library` 可以装入任意
共享库。L17 验证绕开 Python 导入后，实验入口再次检查时必须拒绝继续使用；不宣称底层 dlopen
本身被拦截。自有符号不回退正式/CANN 的独立 C++ 对照测试继续使用
`ci/tests/test_experimental_symbol_source.py`（mock ELF 证据单列，不能代替真实 NPU 路径）。

运行方法：

```bash
# 填写实际一次性环境路径；必须使用新的输出目录。
cp experimental/_tests/a2_isolation/config.example.json /absolute/a2-test-config.json
python experimental/_tests/a2_isolation/run.py \
    --config /absolute/a2-test-config.json --output /absolute/results/a2-isolation-001
```

`summary.json` 只有在全部 28 项完成、全部符合预期、受保护文件不变时才会 `complete=true`。
缺 profile、缺制品或 A2 设备身份未确认均标为 BLOCKED；负例只有在目标操作阶段按预期拒绝才算 PASS。
篡改类用例由 runner 先复制实验包，probe 拒绝直接篡改正式或原始实验安装。

## 6. A2 实际功能/精度用例

ATK 是数值检查入口。已恢复 [RWKV8 ATK 工程](../../../tests/atk/fused_recurrent_rwkv8/README.md)：
来源 #446 固定提交 `103490da…`，原 209 条精度、26 条内存、38 条性能资产保留；CPU 标杆、
输入生成函数和原 JSON 未改；DUT 调用改到实验包，并在 ATK 过滤 None 前检查三个返回槽位、
shape/dtype/device，防止可选输出缺失被过滤。本任务不要求运行性能矩阵。
正式控制算子使用 main 的 [chunk_fwd_o ATK 工程](../../../tests/atk/chunk_fwd_o/README.md)。

| ID | 输入/动作 | 核对内容 |
| --- | --- | --- |
| F01 | RWKV8 FP32/FP16/BF16，B=1,H=2,T=64,K=V=64 | `o` 精度、dtype、shape，默认 s/sa 为 None |
| F02 | initial_state × output_chunk_state × output_sa 的 8 组合 | CPU 参考相同分支；三个返回槽位、状态朝向和空输出语义 |
| F03 | T=1/15/16/17/33，chunk_len=8/16；s 开启 | 零快照、整块、尾块；T<chunk_len 不越界，不遗失临时存储生命周期 |
| F04 | K/V=(8,128)/(128,8)/(96,64)/(128,128) | K≠V、pad、容量边界；数据不串到其他 head |
| F05 | reverse=true/false，带初态、s/sa 全开 | 与既有 PTA CPU golden 的反向递推一致；**原 209 条 ATK 不覆盖，须补 executor/JSON 并校准后执行** |
| F06 | 非法 rank/dtype/K/V、初态shape、chunk_len=0/-1/2^32/INT64_MAX | 在 host/公共入口拒绝，包含实际/期望约束；错误后用合法 case 验证新进程正常 |
| F07 | 正式 chunk_fwd_o FP16/BF16，小 fixed + varlen | 安装实验前、共存、卸载实验后与同一 CPU golden 比较，结果不受实验包影响 |
| F09 | 默认队列、inline、非默认 NPU stream；重复零快照/多输出 | 同一输入输出一致，无读取提前、释放提前或跨 stream 串扰；经 ATK executor 包装，不绕过测试入口 |
| F10 | 正式与实验两个独立进程顺序交替，随后并行 | 分别与 CPU golden 比较；记录 PID、卡号、库路径；单卡资源不足时并行项 BLOCKED，不以顺序测试替代 |

新增逻辑分支使用至少 3 个固定种子。沿用已确认的 CPU 参考与 `mixed_tolerance_bm`；不通过时
保留失败并按精度定位流程处理，不放宽阈值。不能把“209 条旧资产已恢复”写成 F05/F09 已覆盖。
对于 `.run` 原生 ACLNN 入口，复用 C++ Example 做加载/接口连通；数值对拍仍从 ATK executor
调用和归档，额外记录实际 `libcust_opapi.so` 路径，与 wheel 路径分别列结果。

```bash
# 在专用 experimental_atk 环境中；FLA_NPU_ENV 指向实验 wheel 内的环境脚本。
FLA_NPU_ENV=/absolute/experimental_atk/site-packages/fla_npu_experimental/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash \
ATK_OUTPUT_ROOT=/absolute/results/rwkv8-a2 \
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -soc=ascend910b -npu_device_id=0 -scope=accuracy
# 使用同一实验环境/独立结果目录补 determinism 和 mssanitizer；正式控制算子在正式 profile 执行。
```

## 7. 历史证据与完成判定

以下为 2026-11-03/04 的历史结果，制品包含现已移除的样例。清理后制品的定向回归单列于
[2026-11-06 记录](../../validation-rwkv-only-20261106.md)，不累计历史通过数。
F08 随样例移除，其余用例编号保持稳定。experimental 不支持 overlay；
表中的正式 overlay 仅是正式功能不受影响的对照项。

- A2 runner 自测 15/15 通过（包含在下述 117 项中），没有跳过。
  覆盖错误分类、阶段判定、崩溃/超时不作成功、真实 .pth 开关、保护快照、根目录删除、篡改保护和 CANN Python 路径保留。
- 原 ATK JSON：209/26/38 条可解析；输入生成、CPU reference、run_cpu 的 AST 与来源提交一致。
- A2：构建相关 117 项、Python 包 24 项单元测试通过；单算子/全量实验及正式对照 `.run`、wheel 构建通过。
- 装载矩阵 L01–L28：28/28；真实安装/卸载检查：19/19；反向安装顺序和正式 overlay 移除补测：5/5。
  受保护的正式包内容哈希保持不变。
- RWKV8 单算子 wheel：209/209 精度通过；历史全量 wheel：RWKV8 默认队列 3/3、inline 3/3。
  三条真实执行轨迹记录实验 host 路径、dispatcher 和实际 queue/inline 模式。
- 正式 chunk_fwd_o：安装实验前、共存、卸载实验后三阶段各 2/2（FP16/BF16 fixed）通过，正式包哈希不变。
- F05 reverse、F07 varlen、F09 非默认 stream、F10 并行均未执行；完整异常输入 NPU 矩阵、
  determinism、MSS、性能和独立 `.run` 原生数值对拍也未执行。F01–F04 以原 209 条资产实际参数为准，
  不把设计表中每个参数点都算成已执行。本轮是定向集成验证，不是完整正式算子验收。

每项报告保留预期、实际、命令、退出码、环境/制品摘要和 PASS/FAIL/BLOCKED。
功能执行缺项、错误被掩盖、正式目录改变或只完成 import 均不能报整体验证通过。
