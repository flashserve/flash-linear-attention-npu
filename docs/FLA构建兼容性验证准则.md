# FLA 构建兼容性验证准则

- **状态**：试行；PR #878 为首个落地场景
- **用途**：长期守护 `build.sh`、CMake、wheel/run 包打包、离线依赖和 OPP overlay 等构建链修改
- **适用对象**：所有会改变构建输入、构建图、产物布局、安装行为或运行时产物选择的改动
- **原则**：本准则定义验证模型和判定规则，不把一次 PR 的机器、分支或 CANN 路径写死
- **关联设计**：[《FLA NPU 算子包「单算子叠加层」增量设计》](OPP叠加层增量设计.md)

本文同时包含当前必须执行的判定规则和后续自动化目标。标为“长期目标”的 manifest、矩阵调度或报告生成器尚未落地时，验证者必须归档等价的命令、日志和产物证据，但不得把规划中的文件描述成当前构建产物。

## 1. 目标

本准则需要回答六个独立问题：

1. 构建入口是否正确接受、拒绝和解释参数；
2. 构建过程是否干净、可重复、无缓存串扰和并发竞态；
3. 产物是否只发生预期变化；
4. 上游所有 active branch 是否仍可使用该构建能力；
5. A2/A3/A5 是否能在各自机器上原生构建并运行；
6. 同版本和跨版本 CANN 组合是否满足声明的兼容范围。

最终结果不能只有一个总 PASS。每个维度必须单独给出结论，使“能编译”“产物正确”“本机运行正确”“跨 CANN 兼容”互不替代。

## 2. 核心原则

### 2.1 锁定所有移动输入

每轮验证开始时必须把移动名称解析成不可变标识：

```text
GitHub active branch name -> exact commit SHA
CANN profile             -> exact version + installation path + version.info hash
machine profile          -> exact host + SoC + device id
Python environment       -> Python/torch/torch_npu/triton-ascend exact versions
```

测试过程中分支发生移动，不改变本轮输入；下一轮重新发现并锁定。

### 2.2 原生验证优先

| 平台 | 构建位置 | 构建参数 | 运行位置 |
|---|---|---|---|
| A2 | A2 服务器 | `--soc=ascend910b` | A2 |
| A3 | A3 服务器 | `--soc=ascend910_93` | A3 |
| A5 | A5 服务器 | `--soc=ascend950` | A5 |

每个平台的正式 PASS 必须来自该平台原生构建和原生运行。交叉编译只能作为补充证据，不能替代原生 PASS。

### 2.3 语义比较优先于二进制 hash

二进制 hash 只在相同源码、工具链、路径和环境下作为强判据。跨机器或跨 CANN 时，优先比较：

- 文件集合；
- 规范化 JSON；
- 动态符号表；
- 公共 ABI；
- kernel 变体与映射；
- 运行输出和变形关系。

### 2.4 每项都必须解释产物变化

每个验证项必须同时给出：

```text
expected_delta
actual_delta
unexpected_delta
verdict
```

“编译成功”但无法解释产物变化，不得判为 PASS。

## 3. 术语与维度拆分

### 3.1 两组不同的“在线/离线”

以下两组概念必须分开建模：

| 维度 | 取值 | 含义 |
|---|---|---|
| kernel 生成模式 | AOT / JIT | `build.sh --pkg` 预编 device kernel；`build.sh --jit` 不预编 kernel，由运行环境处理 JIT |
| 依赖获取模式 | online / network-disabled | third-party 依赖可联网获取，或只能使用本地 cache/内嵌 offline bundle |

因此需要分别验证：

- AOT + online；
- AOT + network-disabled；
- JIT + online；
- JIT + network-disabled。

如果某组合不受支持，必须在支持矩阵中明确声明，而不是运行时临时 SKIP。

### 3.2 编译期参数与运行期参数

不能把所有 API 参数都乘入重新编译矩阵。

| 参数类别 | 示例 | 覆盖方式 |
|---|---|---|
| 编译期算子范围 | `--ops` / `FLA_NPU_OPS` | 空、单算子、多算子、依赖闭包、非法名称 |
| 编译期 SoC | `--soc` / `FLA_NPU_SOC` | A2/A3/A5 各自原生执行 |
| 编译期 kernel 模板输入 | `--kernel_template_input` | 对算子声明的有限取值逐项构建 |
| 编译选项 | `-O0..O3`、OOM、experimental、bisheng flags | 参数契约全覆盖；昂贵构建采用 pairwise，必要时全组合 |
| 运行期枚举 | layout、dtype、bool/enum | 所有合法值运行验证 |
| 运行期 optional | optional tensor/attribute | `None`/存在；多参数采用 pairwise |
| 运行期数值 | chunk size、shape、长度 | 最小值、分支阈值前后、典型值、最大值、非法值 |

运行期参数清单必须从被测分支自己的 schema、`def.cpp` 和 tiling 条件生成，不能拿 `main` 的参数表套历史分支。

## 4. 支持矩阵

### 4.1 active branch

每轮开始时从上游仓库动态读取所有 active branch：

```text
https://github.com/flashserve/flash-linear-attention-npu
```

推荐使用 GitHub API：

```text
GET /repos/flashserve/flash-linear-attention-npu/branches?per_page=100
```

保存快照：

```json
{
  "discovered_at": "<ISO-8601>",
  "repository": "flashserve/flash-linear-attention-npu",
  "branches": [
    {"name": "main", "sha": "...", "protected": true}
  ]
}
```

“active”默认表示 GitHub 当前未删除的 branch。若项目另有生命周期规则，应在配置中显式排除 archived/实验分支，并记录理由。分支补丁无法应用、构建入口不存在或契约不兼容时，应记为明确的 branch compatibility FAIL，不能静默跳过。

### 4.2 每个 active branch 的三个源码状态

| 状态 | 定义 | 用途 |
|---|---|---|
| `upstream_base` | active branch 原始 SHA | 证明该分支自身的基线状态 |
| `candidate_base` | 同一 SHA 应用待验证构建脚本改动 | 判断构建改动是否引入回归 |
| `candidate_overlay` | `candidate_base` 加受控算子 fixture 修改，再用 `--ops` 构建 | 验证 scoped/overlay 变化范围与运行行为 |

需要进行两组比较：

1. `candidate_base` 对 `upstream_base`：构建脚本变化是否改变不应改变的产物或行为；
2. `candidate_overlay` 对 `candidate_base`：单算子或少量算子改动是否只影响预期层。

fixture 修改应在临时 worktree 中自动施加和撤销，不能污染分支源码。fixture 至少覆盖：

- 无源码修改的 scoped 控制组；
- host-only、接口不变；
- tiling-only；
- kernel-only；
- 公共 aclnn ABI 变化的负向用例；
- 新增算子的负向或完整 wheel 用例；
- 有复合依赖关系的算子。

### 4.3 SoC 与 CANN profile

机器与 CANN 版本由机器可读配置维护，例如：

```yaml
platforms:
  a2:
    host: root@a2-host
    soc: ascend910b
    python_env: zzy311new
    cann:
      - id: cann-9.1.0
        version: 9.1.0
        set_env: /path/to/cann-9.1.0/ascend-toolkit/set_env.sh
      - id: cann-9.2.0
        version: 9.2.0
        set_env: /path/to/cann-9.2.0/ascend-toolkit/set_env.sh

  a3:
    host: root@a3-host
    soc: ascend910_93
    python_env: <env>
    cann: []

  a5:
    host: user@a5-host
    soc: ascend950
    python_env: zzy
    cann: []
```

配置文件只记录声明支持或仍有用户使用的 CANN 安装。每个 profile 必须在执行前验证：

- `set_env.sh` 存在；
- `version.info` 与声明一致；
- `which bisheng`、`ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 指向该 profile；
- Python、torch、torch_npu、triton-ascend 满足该 CANN 组合要求；
- 前一个 profile 的环境变量和动态库路径没有残留。

### 4.4 同版本与跨版本 CANN

假设同一 SoC 机器登记了 N 个 CANN 版本：

- 每个版本必须执行 `CANN X build -> CANN X runtime`；
- 每份已构建产物再在该机器所有运行 CANN 上执行，形成完整有向矩阵 `X build -> Y runtime`；
- 构建产物生成 N 次，运行验证执行 N x N 次；
- 每次运行使用全新 shell 和进程，只 source 一个 CANN profile。

示例：A2 有 CANN 9.1 和 9.2：

```text
9.1 build -> 9.1 runtime
9.1 build -> 9.2 runtime
9.2 build -> 9.1 runtime
9.2 build -> 9.2 runtime
```

跨 CANN 结果必须按产品支持策略分类：

| 结果 | 含义 |
|---|---|
| PASS | 该有向组合在声明支持范围内且验证通过 |
| EXPECTED_UNSUPPORTED | 产品明确不承诺，验证也稳定拒绝；必须附支持声明 |
| FAIL | 声明支持但构建/加载/运行失败，或发生错误结果 |
| NOT_RUN | 资源或前置条件缺失；不能计入通过率 |

## 5. 六类守护契约

### 5.1 输入契约

构建入口必须明确声明并验证：

- 支持的 `--soc`、`--ops`、AOT/JIT、模板参数、优化参数；
- 每个参数默认值；
- 合法组合、互斥组合和依赖关系；
- 非法值在 CMake 配置和耗时构建前失败；
- CLI 与等价环境变量产生相同语义；
- 参数顺序不影响结果；
- 未识别参数不能被静默忽略。

长期建议让构建系统提供机器可读描述：

```bash
bash build.sh --describe-options --json
bash build.sh --list-ops --json
```

验证程序从机器可读输出动态生成参数用例，避免测试与脚本各维护一份列表。

### 5.2 构建过程契约

| 场景 | 必须满足 |
|---|---|
| 干净目录首次构建 | 成功并产生唯一目标产物 |
| 同目录重复构建 | 不混入旧产物；语义 manifest 一致 |
| 人工放入陈旧产物 | 陈旧文件不会被选择、打包或上传 |
| `-j1` 与并行构建 | 语义产物一致，无竞态 |
| 构建中途失败后重试 | 半成品不会被识别为成功产物 |
| 不同 active branch 连续构建 | 缓存、生成文件、版本号和产物不串分支 |
| online 与 network-disabled | 核心语义产物一致 |
| AOT 与 JIT | 各自布局正确，运行结果满足等价关系 |

同时记录：

- wall time；
- CPU/内存峰值；
- CXX 编译次数；
- opc 调用次数；
- 下载次数和域名；
- 产物数量与大小。

性能阈值应相对同一机器、同一 CANN、同一分支基线设置，跨机器的绝对耗时不直接比较。

### 5.3 产物契约

长期自动化目标是让所有构建统一生成 `artifact-manifest.json`：

```json
{
  "schema_version": 1,
  "provenance": {
    "branch": "main",
    "commit": "<sha>",
    "candidate_patch": "<sha-or-digest>",
    "soc": "ascend950",
    "build_cann": "9.2.0",
    "build_mode": "aot",
    "dependency_mode": "network-disabled",
    "command": ["bash", "build.sh", "..."]
  },
  "files": {},
  "op_api": {},
  "ops_info": {},
  "binary_info": {},
  "kernels": {},
  "tiling": {},
  "launcher": {},
  "environment_scripts": {}
}
```

该生成器落地后，manifest 必须包含：

| 层 | 记录内容 |
|---|---|
| L1 host | `libcust_opapi.so` 路径、大小、hash、ELF 属性、导出 aclnn 符号、公共头 ABI |
| L2 算子清单 | `aic-*-ops-info.json` 顶层算子键与规范化内容 |
| L3 kernel 索引 | `binary_info_config.json` 算子键、变体数、模板参数、二进制映射 |
| L4 实体 | tiling/proto 库和 kernel `.o` 路径、数量、大小、hash |
| wheel 级 | `libfla_npu_stable.so` hash、能力和 build stamp；overlay 中不得有第二份 launcher |
| 环境 | `config.ini`、生成的 `set_env.bash`、最终 vendor 顺序 |
| 安装边界 | 安装前后 base 的完整 manifest |

在生成器落地前，使用构建日志、文件清单、规范化 JSON、动态符号表、公共头签名和运行记录形成等价证据。比较 overlay 与 base 时，只比较目标算子切片和共享聚合产物，不能把 overlay 缺少非目标算子当成异常。

### 5.4 ABI 与接口契约

接口变化必须优先在编译期拦截，不能依赖运行时告警或自动回落。PR #878 当前落地的规则是：

1. 通过 `build.sh` 生成完整 custom 包时，强制执行 `op_abi_validate.py`，校验 stable adapter 与本次 OPP 公共 aclnn 头的参数数量、顺序及类型；用户不需要增加构建参数；
2. `--ops` scoped 包只检查同一算子集合；全量包检查全部可对应的 adapter/header；
3. Stable-only wheel 构建拒绝 `FLA_NPU_BUILD_STABLE_ABI=0`；直接运行 `build.sh --pkg` 也不以此跳过 adapter/header 门禁；不生成完整 custom 包的非打包、`--opkernel`、`--ophost`、`--opapi`、测试和 static 路径保持原行为；
4. overlay 安装前由 `_op_abi.py` 直接比较 overlay 与 base 的公共 aclnn 头；import 时再次比较，防止手工复制绕过安装器；
5. 同名 aclnn 原型不一致，或 overlay 出现 base wheel 未声明的新 aclnn 符号时，安装/import 必须硬错误，不能告警后继续，也不能自动回落 base/ctypes；
6. 首次 `_stable.load()` 加载实际 launcher 并通过严格 build stamp 后，若启用多个 host 库，必须从实际库（含 `FLA_NPU_STABLE_LIB` 覆盖路径）取得 `fla_npu_stable_overlay_order_version() >= 1`，否则禁止算子解析与执行。安装仅校验 ACLNN ABI，不能宣称 launcher 能力已通过；
7. 失败构建不得被标记、安装或上传为 PASS 产物。

当前 overlay 必须携带 `abi_manifest.json`；安装/import 核验规范化公共 aclnn 签名、host 库及头文件哈希。规划中的更广泛 `artifact-manifest.json` 不替代这个现有硬门禁。

Stable wrapper、torch schema、mutation 和 attribute 语义也必须纳入长期接口契约。不过这类检查不能错误宣称已由当前 public-header 比较完全覆盖。

自动检查无法可靠识别“类型不变但业务语义改变”。这类变化需要：

- schema/mutation/attribute 语义 manifest；
- 代码评审标签；
- 同一 Stable 后端的 baseline/candidate 对照；
- 针对性边界用例；
- 必要时强制完整 wheel 发布。

### 5.5 兼容性契约

每个 active branch 都要报告：

```text
patch_apply
clean_build
native_build
native_runtime
same_cann_runtime
cross_cann_runtime
aot_runtime
jit_runtime
offline_build
```

GitHub active branch 决定被测源码集合，但不自动表示每个分支支持所有 CANN。具体支持范围由版本化配置声明。

### 5.6 行为契约

代表算子不固定为某几个名字，而按能力类别从被测分支动态选择：

- 编译最快的普通算子；
- 有 optional tensor 的算子；
- 有 bool/enum/layout 分支的算子；
- 有多个 kernel 变体的算子；
- 有复合依赖关系的算子；
- Stable 公共入口（完整验收要求所有 37 个算子；代表配方只证明所选集合）。

若某分支不存在某类别，报告 `NOT_APPLICABLE` 和发现依据。

必须守护的变形关系：

```text
AOT 输出 == JIT 输出
Stable baseline 输出 == Stable candidate 输出（同输入与 seed；语义不变的改动）
online 构建输出 == network-disabled 构建输出
启用 overlay：目标算子来自 overlay
启用 overlay：非目标算子回落 base
禁用 overlay：全部来自 base
卸载 overlay：输出与 base 基线一致
重复构建：语义 manifest 不变
-j1 与并行构建：语义 manifest 不变
```

固定 digest 作为本轮证据保存，但长期断言以关系为主。

## 6. Overlay 对 Base 的预期差异

每个 case 必须声明 `change_class`，并按下表判定：

| `change_class` | overlay 相比 base 的预期变化 | 禁止变化 |
|---|---|---|
| `no_change_control` | 目标算子集合缩小；目标算子的 ABI、索引语义和运行结果一致 | launcher、base 文件、非目标算子语义 |
| `host_impl` | `libcust_opapi.so` 内容变化 | 导出符号集合、公共头 ABI、kernel 变体 |
| `tiling_impl` | tiling/opmaster 相关产物变化 | 公共 aclnn ABI；无关 kernel |
| `kernel_impl` | 对应 kernel `.o` 与可能的映射 hash 变化 | host 公共 ABI、无关算子 |
| `kernel_template` | `binary_info_config` 变体和 kernel 文件集合按模板输入变化 | host 公共 ABI、launcher |
| `public_aclnn_abi` | 构建期检测到接口变化 | 产生可安装 overlay |
| `new_operator` | L1-L4 都新增 | 仅靠历史 wheel + overlay 获得完整公共入口 |
| `jit` | 不含预编 kernel `.o`；首调生成缓存，后续命中 | 把 AOT 残留 kernel 打入包中 |
| `network_disabled` | 核心 OPP 语义与 online 一致 | 访问网络；缺依赖后仍成功出包 |

共享聚合库发生二进制变化不等于所有算子都发生语义变化。报告必须进一步列出导出符号、算子键、变体和运行结果的归属。

## 7. 标准用例流程

每个 case 按以下顺序执行：

1. 解析 branch 名并锁定 SHA；
2. 创建隔离 worktree、构建目录、安装目录和 conda 环境；
3. source 唯一的 build CANN profile；
4. 记录环境快照并校验无其他 CANN 残留；
5. 构建 `upstream_base`；
6. 应用候选构建改动，构建 `candidate_base`；
7. 应用受控 fixture，构建 `candidate_overlay`；
8. 为三份产物生成 manifest；生成器尚未落地时，归档 §5.3 所述的等价证据；
9. 先执行 ABI/接口门禁，再执行产物差异判定；
10. 安装到 base 副本，不修改真实用户环境；
11. 对每个 runtime CANN 使用全新 shell/process；
12. 分别在 Stable baseline/candidate 上运行 overlay disabled/enabled/rollback；
13. 记录实际动态库映射、backend 和 vendor 顺序，不能只由环境变量推断；
14. 卸载 overlay，验证 base manifest 和运行输出恢复；
15. 生成机器可读 verdict。

## 8. 标准归档格式

每个 case 固定输出：

```text
<run-id>/<branch>/<soc>/<build-cann>/<mode>/<case>/
├── case.json
├── environment-build.json
├── environment-runtime-<cann>.json
├── build-upstream-base.log
├── build-candidate-base.log
├── build-overlay.log
├── upstream-base-manifest.json
├── candidate-base-manifest.json
├── overlay-manifest.json
├── base-regression-delta.json
├── overlay-delta.json
├── abi-report.json
├── runtime-stable-baseline-<cann>.json
├── runtime-stable-candidate-<cann>.json
├── rollback.json
└── verdict.json
```

`case.json` 至少包含：

```json
{
  "branch": {"name": "main", "sha": "..."},
  "change_class": "kernel_impl",
  "build_cann": "9.2.0",
  "runtime_cann": "9.1.0",
  "platform": "a5",
  "soc": "ascend950",
  "build_mode": "aot",
  "dependency_mode": "network-disabled",
  "op_scope": ["chunk_local_cumsum"],
  "compile_parameters": {},
  "runtime_profiles": []
}
```

`verdict.json` 不允许只含一个布尔值，至少包含：

```json
{
  "cli_contract": "PASS",
  "clean_build": "PASS",
  "repeat_build": "PASS",
  "parallel_build": "PASS",
  "artifact_contract": "PASS",
  "abi_contract": "PASS",
  "native_runtime": "PASS",
  "cross_cann_runtime": "PASS",
  "offline_build": "PASS",
  "aot_jit_equivalence": "PASS",
  "install_rollback": "PASS",
  "performance_regression": "PASS",
  "overall": "PASS"
}
```

`overall=PASS` 只在所有 required 项均为 PASS 时成立。`NOT_RUN`、`SKIP` 和 `EXPECTED_UNSUPPORTED` 不得冒充 PASS。

## 9. 分级执行策略

完整矩阵成本很高，按风险分级：

| 级别 | 触发时机 | 范围 |
|---|---|---|
| L0 | 每次提交 | 参数解析、静态检查、manifest 工具单测、ABI 负向测试 |
| L1 | 普通 PR | 当前分支；每个 SoC 的主 CANN；AOT/JIT；代表算子；Stable baseline/candidate |
| L2 | 修改 `build.sh`、CMake、setup、打包/安装或 offline bundle | 所有 SoC；所有登记 CANN 的同版本原生构建运行；online/network-disabled；重复/并行/失败重试 |
| L3 | nightly | 所有 active branch x SoC x 登记 CANN；同版本原生运行；核心参数 pairwise |
| L4 | release | L3 + 完整 `build CANN X -> runtime CANN Y` 有向矩阵 + 全部关键参数分支 |

构建脚本修改至少触发 L2，不能由普通“按改动文件选少量算子”逻辑进一步缩减。

### 9.1 降维规则

允许的降维方式：

- 等价类；
- boundary value；
- pairwise/正交组合；
- 复用同一构建产物执行多个 runtime CANN；
- 代表算子按能力类别动态选择。

不允许的降维方式：

- 用 A5 交叉编译代替 A2/A3 原生构建运行；
- 用一个 CANN 版本代替用户仍在使用的其他登记版本；
- 用编译成功代替运行验证；
- 用一个目标算子 PASS 代替非目标算子回落；
- 用 hash 变化代替行为变化；
- 用 `SKIP` 计入通过率。

## 10. 失败分类

| 类别 | 示例 |
|---|---|
| `DISCOVERY_FAILURE` | active branch/CANN profile/算子能力清单无法发现 |
| `PATCH_COMPAT_FAILURE` | 候选构建改动无法应用到 active branch |
| `CLI_FAILURE` | 非法参数未早失败，或合法参数被误拒绝 |
| `BUILD_FAILURE` | 编译、链接、打包失败 |
| `STALE_ARTIFACT` | 选择或打包了上一轮产物 |
| `UNEXPECTED_DELTA` | 产物变化超出 `change_class` 允许范围 |
| `ABI_FAILURE` | aclnn/schema/adapter 契约不一致 |
| `INSTALL_BOUNDARY_FAILURE` | overlay 安装修改了 base |
| `RUNTIME_FAILURE` | 调用失败、错误结果、后端或 vendor 来源错误 |
| `CROSS_CANN_FAILURE` | 声明支持的 build/runtime CANN 组合失败 |
| `OFFLINE_FAILURE` | 断网构建访问网络或 bundle 不完整仍出包 |
| `NONDETERMINISM` | 重复/并行构建语义 manifest 不一致 |
| `PERFORMANCE_REGRESSION` | 同环境构建时间、内存或编译次数超过阈值 |

失败报告必须给出复现命令、输入 SHA、环境快照和首个失败阶段。

## 11. 准入规则

### 11.1 普通 PR

- L0 必须全部通过；
- L1 required 项必须全部通过；
- 与改动无关的已知问题只能以带 issue/到期日的 waiver 存在。

### 11.2 构建系统 PR

- 至少完成 L2；
- `candidate_base` 不得出现未声明的产物差异；
- ABI 负向用例必须在编译期硬失败；
- 所有 SoC/CANN 同版本原生组合必须通过；
- 断网构建必须证明没有网络访问；
- base 安装边界必须保持不变。

### 11.3 Release

- L3 全部 required 项通过；
- L4 跨 CANN 矩阵按支持声明给出 PASS 或 EXPECTED_UNSUPPORTED；
- 所有 NOT_RUN 都必须在发布结论中显式列出，不能省略。

## 12. 实施路线

建议按以下顺序落地，保证每一步可独立使用：

1. 定义 `compatibility-matrix.yaml` 与 branch/CANN/machine 快照格式；
2. 实现统一 `artifact-manifest.json` 生成器；
3. 实现 manifest comparator 和 `expected_delta` 规则；
4. 把现有参数、环境、ABI 单测接入 L0；
5. 建立 A2/A3/A5 原生 runner 与 CANN profile 切换；
6. 接入 active branch nightly；
7. 接入 network-disabled 构建；
8. 接入跨 CANN 有向运行矩阵；
9. 增加历史趋势、性能阈值和 waiver 管理。

本准则的第一批落地场景可以是 OPP overlay，但工具和报告结构不得依赖某个固定算子或本次 PR 的目录名，才能继续守护后续构建脚本修改。
