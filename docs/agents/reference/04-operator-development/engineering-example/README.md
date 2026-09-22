# 标准工程示例（`engineering-example/`）

> 示例角色：`ENGINEERING-EXAMPLE`
>
> 示例版本：`V1`（与 [`../engineering-structure.md`](../engineering-structure.md) `V1` 同期）
>
> 本目录下的文件是**实际样例文件**：每一份文件开头都有该文件自己的「注意事项」注释块，路径按真实仓库
> 位置镜像组织，可整棵复制后改名使用。
>
> 它们位于 `docs/` 下，**不参与构建、不参与 wheel 打包**（根 `CMakeLists.txt` 按显式 OP 列表构建，
> `MANIFEST.in` 只收 `fla/`、`cmake/`、`common/`、`scripts/`、`torch_custom/fla_npu`）。示例代码里的
> `#include` 路径按"复制到真实位置之后"的形状写，直接在 `docs/` 下编译不成立。

## 1. 先选形态

一个算子不一定都要有 kernel。先判断这套计算能否由已有算子拼出来：

| 形态 | 目录特征 | 示例 |
| --- | --- | --- |
| A 独立实现 | 自己的 `def` + `op_kernel` + L0 + L2 齐全 | [`standalone/`](standalone/) |
| B 自研主体 + L0 组合入口 | 主体独立实现，另有 V2 之类组合入口，在 L0 层拼接其它算子 | [`l2-composition/`](l2-composition/) |
| C 只有 L2 | 没有新的 `def`/`op_kernel`，L2 直接调用别的算子的 L0 并拼接公开输出 | 同 [`l2-composition/`](l2-composition/)（差异见该目录 README） |

B 与 C 的 L2 写法相同，区别只在"本算子是否还有自己的 kernel"。形态规则见
[`../engineering-structure.md`](../engineering-structure.md) §5.4。

## 2. 形态 A 的完整文件清单

每个文件都能在 [`standalone/`](standalone/) 下找到实际样例（含该文件的注意事项）：

| 文件（相对 `fla/ops/ascendc/<模块>/<算子>/`） | 这个文件必须做什么 | 规范章节 |
| --- | --- | --- |
| `CMakeLists.txt` | 只做子目录遍历；`ENABLE_TEST/BENCHMARK` 决定是否进 `tests` | §2 |
| `README.md` | 能力、输入限制、输出布局的**唯一定义来源** | §7.10 |
| `docs/api.md` | 全部公开接口、返回码、可选输出语义、布局规则 | §3.4、§5.1 |
| `docs/design.md` | 方案详设；开头记录规则版本 | §7.10 |
| `op_host/CMakeLists.txt` | 四段结构：注册来源、`add_modules_sources`、编译选项、`ENABLE_TEST` | §3.1 |
| `op_host/<算子>_def.cpp` | `Input/Output/Attr` + 三 SoC config；**输出全部 `REQUIRED`** | §3.2 |
| `op_host/<算子>_tiling.h` | TilingData 字段 + `REGISTER_TILING_DATA_CLASS` + 输入/属性索引枚举 | §3.3 |
| `op_host/<算子>_tiling.cpp` | 校验 → TilingData → `GET_TPL_TILING_KEY` → `SetTilingKey/SetBlockDim` → workspace | §3.3、§4.2 |
| `op_host/<算子>_tiling_processor.h` | 任务切分、offset、workspace 区域表（可被 host UT 直接调用） | §3.3、§3.6 |
| `op_host/<算子>_output_mask.h` | 可选输出档位与掩码常量 + 互斥/全覆盖 `static_assert` | §5.1 |
| `op_host/arch22/<算子>_tiling_impl.h` | A2/A3 专用 tiling 常量与 tile 选择 | §3.5 |
| `op_host/arch35/<算子>_tiling_impl.h` | A5 专用 tiling 常量与 tile 选择 | §3.5 |
| `op_host/op_api/<算子>.h` / `.cpp` | L0 内部 exec：固定输出槽位数组，按档位写内部张量 | §3.4 |
| `op_host/op_api/aclnn_<算子>.h` / `.cpp` | L2：可空输出描述符、档位推导、非法组合拦截、`GetWorkspaceSize`/`Launch` 成对 | §3.4、§5.1 |
| `op_host/tests/CMakeLists.txt` + `<算子>_tiling_processor_test.cpp` | tiling 分支与档位单测（`ENABLE_TEST` 时编译） | §3.6 |
| `op_kernel/<算子>.cpp` | 唯一入口：模板参数、架构选择、args 组装、分派 | §4.1、§4.3 |
| `op_kernel/<算子>_struct.h` | 设备侧与 host 一致的 TilingData 结构与常量 | §4.3 |
| `op_kernel/<算子>_tiling_key.h` | **必须**：`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` | §4.2 |
| `op_kernel/<算子>_<stage>.h` | 按 Stage 拆分的实现头 | §4.3 |
| `op_kernel/arch22|arch35/<算子>_cube.h`、`_vec.h` | 平台专用实现，文件名与根目录同名 | §4.1 |
| `tests/README.md` | 算子自带脚本/数据的索引（ATK 资产放 `tests/atk/<算子>/`） | §6 |
| `torch_custom/fla_npu/csrc/src/stable_<算子>.cpp` | `kSchema_` + `run_` + 一条 `FLA_STABLE_EXEC`（一算子一文件） | §5.3 |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py`（追加） | 真签名 wrapper：入参名、默认值、返回 tuple | §5.3 |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py`（追加） | `_ASCENDC_OPS` 公开名；原地参数登记 | §5.3 |
| `tests/atk/<算子>/README.md` + 三份 JSON + yaml + gen + executor | 精度/性能/`_mss` 三类用例来源不同 + TilingKey 覆盖表 + 验收结果 | §6.1 |

需要注意的三个高频错误：`_tiling_key.h` 不是可选件；`_policy.h`、`_common.h` 这类文件按实际需要建，
不要为了对齐目录树而空建；`arch22` 与 `arch35` 都可以是平台专用实现目录，不是"只有 A5 才分平台"。

## 3. 复制步骤

1. 复制 `standalone/fla/ops/ascendc/demo/example_scan/` 到目标位置，把 `demo` 换成所属模块；
2. 全局替换三个名字：目录/文件名 `example_scan`、op 类名 `ExampleScan`、Python 名 `npu_example_scan`；
3. 按文件头的「注意事项」逐条核对，**不要只改名字**：def 的输出集合、`output_mask` 档位、tiling key
   模板参数、kernel args 顺序都随算子改变；
4. 复制 `standalone/torch_custom/` 与 `standalone/tests/` 两棵镜像树到真实路径，按其中的追加说明改
   已有文件（`stable_ops.cpp`、`_stable.py`、`__init__.py`）；
5. 用 [`../engineering-structure.md`](../engineering-structure.md) §8 的清单自查，再跑 §6 的三层看护。

## 4. 与真实蓝本映射

| 示例文件 | 真实蓝本（可直接对照） |
| --- | --- |
| `op_host/CMakeLists.txt`、`_def.cpp`、`_tiling.h/.cpp` | `fla/ops/ascendc/kda/chunk_kda_fwd_prepare/` |
| `_output_mask.h`、L0/L2 `op_api/` | 同上（`PREPARE_*_OUTPUT_MASK`、`std::array<const aclTensor *, 13>`） |
| `_tiling_key.h`、`_<stage>.h`、入口分派 | `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/`、`fla/ops/ascendc/kda/chunk_kda_fwd/` |
| `arch22|arch35` 拆分 | `chunk_fwd_h`（两平台各一份）、`chunk_bwd_dv_local`（根目录默认 + `arch35/` 差异） |
| 组合入口（形态 B/C） | `fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/chunk_kda_fwd_v2.cpp` |
| `torch_custom/.../stable_*.cpp` | `torch_custom/fla_npu/csrc/src/stable_kda_gate_cumsum.cpp` |
| `tests/atk/<算子>/` | `tests/atk/chunk_kda_fwd/`、`tests/atk/README.md` |
