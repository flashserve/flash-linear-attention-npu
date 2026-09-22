# 标准工程示例

> 示例角色：`ENGINEERING-EXAMPLE`
>
> 示例版本：`V1`（与 [`../工程结构规范.md`](../工程结构规范.md) `V1` 同期）
>
> 蓝本算子：[`fla/ops/ascendc/kda/chunk_kda_fwd_prepare/`](../../../../../fla/ops/ascendc/kda/chunk_kda_fwd_prepare/)
>
> 用途：把 [`工程结构规范.md`](../工程结构规范.md) 的每条规则落到具体文件上。写新算子时按本文逐文件对照；
> 检视时按本文检查"该文件是否长成这个样子"。
>
> 说明：本文示例是**结构示意**，不参与构建。需要可编译的完整算子模板（含 `op_graph`、`op_kernel_aicpu`）
> 时使用仓库自带的 [`examples/add_example/`](../../../../../examples/add_example/)；需要真实业务形态时直接读蓝本算子。

## 1. 一页总览

```text
新增一个算子 = host 4 类文件 + kernel 1 入口 N 实现头 + L2 1 组接口 + 调用层 3 处 + 看护 3 类资产

fla/ops/ascendc/<模块>/<算子>/
    op_host/<算子>_def.cpp              输出全 REQUIRED，不写 optional 输出
    op_host/<算子>_tiling.cpp/.h        校验 -> TilingData -> SetTilingKey
    op_host/<算子>_tiling_processor.h   切分/offset/workspace
    op_host/<算子>_output_mask.h        可选：输出档位与掩码常量
    op_host/op_api/<算子>.h/.cpp        L0 内部 exec
    op_host/op_api/aclnn_<算子>.h/.cpp  L2 公开接口，可选输出在这里表达与拼接
    op_kernel/<算子>.cpp                唯一入口：架构选择 + tiling key 分派
    op_kernel/<算子>_struct.h           TilingData
    op_kernel/<算子>_common.h           地址解析与公共工具
    op_kernel/<算子>_<stage>.h          Stage 实现
    op_kernel/arch35/...                仅 A5 差异实现，文件名与根目录同名

torch_custom/fla_npu/
    csrc/src/stable_<算子>.cpp          kSchema_<算子> + run_<算子>（一算子一文件）
    csrc/src/stable_ops.cpp             include 一行 + 注册两行
    fla_npu/ops/ascendc/_stable.py      真签名 wrapper
    fla_npu/ops/ascendc/__init__.py     _ASCENDC_OPS 公开名（+ MUTATED_ARGUMENTS）

tests/atk/<算子>/                       README + 三份 JSON + yaml + gen + executor
```

## 2. 蓝本对照：每个文件看什么

| 文件 | 规范点 | 蓝本位置 |
| --- | --- | --- |
| `op_host/CMakeLists.txt` | 四段结构、`--cce-auto-sync=off`、平台分支、`ENABLE_TEST` 进 tests | `chunk_kda_fwd_prepare/op_host/CMakeLists.txt` |
| `<算子>_def.cpp` | 输入 `REQUIRED/OPTIONAL`、输出全部 `REQUIRED`、属性默认值、三 SoC config | `chunk_kda_fwd_prepare_def.cpp` |
| `<算子>_infershape.cpp` | 仅 shape 推导；不承担校验 | `chunk_kda_fwd_prepare_infershape.cpp` |
| `<算子>_tiling.cpp/.h` | 只做校验 / TilingData / `SetTilingKey` | `chunk_kda_fwd_prepare_tiling.cpp` |
| `<算子>_tiling_processor.h` | 任务切分、offset、workspace 区域表 | `chunk_kda_fwd_prepare_tiling_processor.h` |
| `<算子>_output_mask.h` | 可选输出档位、互斥与全覆盖 `static_assert` | `chunk_kda_fwd_prepare_output_mask.h` |
| `op_api/<算子>.h/.cpp`（L0） | 固定输出槽位数组，按档位写内部张量 | `op_api/chunk_kda_fwd_prepare.h`（`std::array<const aclTensor *, 13>`） |
| `op_api/aclnn_<算子>.h`（L2） | 可空输出指针、`GetWorkspaceSize`/`Launch` 成对、`visibility("default")` | `op_api/aclnn_chunk_kda_fwd_prepare.h` |
| `op_api/aclnn_<算子>.cpp`（L2） | 由非空组合推导档位、非法组合报 `outputMask`、缺席输出内部承接 | `op_api/aclnn_chunk_kda_fwd_prepare.cpp` |
| `op_kernel/<算子>.cpp` | 架构 `#if`、tiling key 分派、Stage 分派 | `chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare.cpp` |
| `op_kernel/arch22|arch35/*` | 同名实现头，差异只在被拆分的文件 | `op_kernel/arch22|arch35/chunk_kda_fwd_prepare_{cube,vec}.h` |
| `op_kernel/<算子>_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 实例枚举 | `chunk_fwd_h/op_kernel/chunk_fwd_h_tiling_key.h` |
| `op_host/tests/` | tiling 分支与档位单测 | `chunk_kda_fwd_prepare/op_host/tests/` |
| `tests/atk/<算子>/` | 三份 JSON 来源不同 + TilingKey 覆盖表 + 验收记录 | `tests/atk/chunk_kda_fwd/`、`tests/atk/README.md` |
| `csrc/src/stable_<算子>.cpp` | `kSchema_` + `run_` + 一条 `FLA_STABLE_EXEC` | `torch_custom/fla_npu/csrc/src/` 下任一 `stable_*.cpp` |
| `_stable.py` / `__init__.py` | wrapper 与公开名 | `torch_custom/fla_npu/fla_npu/ops/ascendc/` |

## 3. 关键骨架

以下片段给出"必须出现的结构"，参数表按本算子实际接口展开。

### 3.1 `op_host/<算子>_def.cpp`：输出全部 REQUIRED

```cpp
namespace ops {
class ExampleOp : public OpDef {
public:
    explicit ExampleOp(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType(kDt).Format(kNd).UnknownShapeFormat(kNd).AutoContiguous();
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        // 输出：一律 REQUIRED。调用方能不能不传，是 L2 的事。
        this->Output("y").ParamType(REQUIRED).DataType(kDt).Format(kNd).UnknownShapeFormat(kNd);
        // 反向需要的保存值同样 REQUIRED；L2 用可空指针表达"本次不导出"。
        this->Output("q_hat").ParamType(REQUIRED).DataType(kDt).Format(kNd).UnknownShapeFormat(kNd);

        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("scale").AttrType(REQUIRED).Float(1.0F);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true) /* 其余标志跟随相邻算子 */;
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(ExampleOp);
} // namespace ops
```

对照：`chunk_kda_fwd_prepare_def.cpp` 的 13 个输出全部 `REQUIRED`。历史算子
（`chunk_kda_fwd`、`chunk_fwd_h`、`causal_conv1d_bwd`、`chunk_kda_bwd`）里出现的
`Output(...).ParamType(OPTIONAL)` 不要再照抄。

### 3.2 `op_host/<算子>_output_mask.h`：档位是显式枚举

```cpp
enum ExampleOutputSlot : uint32_t { EXAMPLE_OUT_Y, EXAMPLE_OUT_Q_HAT, EXAMPLE_OUT_K_HAT, EXAMPLE_OUTPUT_COUNT };
enum ExampleOutputMask : uint32_t {
    EXAMPLE_MASK_Y = 1U << EXAMPLE_OUT_Y,
    EXAMPLE_MASK_Q_HAT = 1U << EXAMPLE_OUT_Q_HAT,
    EXAMPLE_MASK_K_HAT = 1U << EXAMPLE_OUT_K_HAT,
};
constexpr uint32_t EXAMPLE_REQUIRED_OUTPUT_MASK = EXAMPLE_MASK_Y;
constexpr uint32_t EXAMPLE_OPTIONAL_OUTPUT_MASK = EXAMPLE_MASK_Q_HAT | EXAMPLE_MASK_K_HAT;
constexpr uint32_t EXAMPLE_SAVE_OUTPUT_MASK = EXAMPLE_REQUIRED_OUTPUT_MASK | EXAMPLE_OPTIONAL_OUTPUT_MASK;

static_assert((EXAMPLE_REQUIRED_OUTPUT_MASK & EXAMPLE_OPTIONAL_OUTPUT_MASK) == 0);
static_assert((EXAMPLE_REQUIRED_OUTPUT_MASK | EXAMPLE_OPTIONAL_OUTPUT_MASK) == ((1U << EXAMPLE_OUTPUT_COUNT) - 1U));
```

档位名用语义命名（`none/forward/save/recompute`），不要用位组合拼表达式散落在 L2 里。

### 3.3 `op_host/op_api/aclnn_<算子>.h`（L2）：可选输出在这里

```cpp
__attribute__((visibility("default")))
aclnnStatus aclnnExampleOpGetWorkspaceSize(
    const aclTensor *q,
    const aclIntArray *cuSeqlensOptional,
    const char *layout,
    double scale,
    const aclTensor *yOut,           // 必选输出：L2 显式校验非空
    const aclTensor *qHatOut,        // nullptr = 本次不导出
    const aclTensor *kHatOut,        // nullptr = 本次不导出
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnExampleOp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

头文件顶部注释必须写：与 V1 的关系（若为 V2）、支持范围、每个可空输出的缺席语义。

### 3.4 `op_host/op_api/aclnn_<算子>.cpp`（L2）：档位推导与拼接

```cpp
static uint32_t GetOutputMask(const ExampleParams &params)
{
    uint32_t mask = EXAMPLE_MASK_Y;
    if (params.qHatOut != nullptr) { mask |= EXAMPLE_MASK_Q_HAT; }
    if (params.kHatOut != nullptr) { mask |= EXAMPLE_MASK_K_HAT; }
    return mask;
}

static aclnnStatus CheckOutputMask(uint32_t mask)
{
    CHECK_COND(mask == EXAMPLE_REQUIRED_OUTPUT_MASK || mask == EXAMPLE_SAVE_OUTPUT_MASK,
               ACLNN_ERR_PARAM_INVALID,
               "输出 nullptr 组合只支持 none/save 两档，当前 outputMask=0x%x。", mask);
    return ACLNN_SUCCESS;
}
```

缺席的输出现场分配内部张量（或直接跳过导出），保证 kernel 侧看到的槽位齐全；
非空且布局不同时做一次明确 `ViewCopy`，并在设计文档写清这次拷贝的代价。

### 3.5 `op_kernel/<算子>.cpp`：架构选择 + 分派

```cpp
#include "<算子>_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/<算子>_cube.h"
#include "arch35/<算子>_vec.h"
#else
#include "arch22/<算子>_cube.h"
#include "arch22/<算子>_vec.h"
#endif

extern "C" __global__ __aicore__ void example_op(GM_ADDR q, /* ... */ GM_ADDR y, GM_ADDR qHat, GM_ADDR kHat, GM_ADDR workspace, GM_ADDR tiling)
{
    // 只做地址解析 + 按 tiling 档位/Stage 分派，具体计算在 <算子>_<stage>.h
}
```

分阶段多次 launch 时用私有 `stage` 属性在同一入口内分派，不新增 L0 原型（参考 `chunk_kda_fwd` 的 `KDA_STAGE_*`）。

### 3.6 `op_kernel/<算子>_tiling_key.h`：模板实例枚举

```cpp
#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(ExampleOp,
    ASCENDC_TPL_DTYPE_DECL(D_T, 10 /*bf16*/, 30 /*fp32*/),
    ASCENDC_TPL_UINT_DECL(V_DIM, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, 128),
    ASCENDC_TPL_BOOL_DECL(USE_GK, 0, 1),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T, 10), ASCENDC_TPL_UINT_SEL(V_DIM, ASCENDC_TPL_UI_LIST, 128), ASCENDC_TPL_BOOL_SEL(USE_GK, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(D_T, 10), ASCENDC_TPL_UINT_SEL(V_DIM, ASCENDC_TPL_UI_LIST, 128), ASCENDC_TPL_BOOL_SEL(USE_GK, 1)),
    /* ... 其余 dtype/V_DIM 组合，用宏逐层展开而不是手抄笛卡尔积 */
);
#endif
```

实例内部用 `if constexpr` 去掉不需要的分支；TilingKey 只标场景族，不标平台。

### 3.7 `csrc/src/stable_<算子>.cpp`：一条宏下发

```cpp
kSchema_ExampleOp =
    "npu_example_op(Tensor q, Tensor? cu_seqlens, int layout, float scale) -> (Tensor, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>> run_example_op(
    Tensor q, std::optional<Tensor> cu_seqlens, int64_t layout, double scale)
{
    auto y = allocate_like(q);
    std::optional<Tensor> q_hat = std::nullopt;   // 由 Python 侧的保留策略决定要不要请求
    std::optional<Tensor> k_hat = std::nullopt;
    FLA_STABLE_EXEC(q_meta, cstr(kExampleOpLayoutNames, layout), scale,
                    out_tensor(meta_of(y)), optional_out_tensor(q_hat), optional_out_tensor(k_hat));
    return {y, q_hat, k_hat};
}
```

顺序必须与 `kSchema_ExampleOp`、`aclnnExampleOpGetWorkspaceSize` 完全一致；返回的可选槽走 boxed optional。

### 3.8 `_stable.py` 与 `__init__.py`

```python
# fla_npu/ops/ascendc/_stable.py
@_op("npu_example_op")
def npu_example_op(q, cu_seqlens=None, layout="BSND", scale=1.0, return_saved=False): ...

# fla_npu/ops/ascendc/__init__.py
_ASCENDC_OPS = (..., "npu_example_op")
# 仅当算子会原地改写输入时才登记：
MUTATED_ARGUMENTS = {..., "npu_example_op": ("state",)}
```

新增关键字参数只能追加在末尾并带默认值；默认值等于历史行为。

### 3.9 `tests/atk/<算子>/`：三份用例来源不同

```text
tests/atk/example_op/
|-- README.md              # 输入限制、标杆来源、SoC、TilingKey 覆盖表、验收结果
|-- atk_example_op.json    # 逻辑分支覆盖（gen_cases 生成后筛选补充）
|-- atk_example_op_perf.json  # 用户模型 case，只跑 NPU
|-- atk_example_op_mss.json   # 按全部可达 TilingKey 人工构造
|-- example_op.yaml
|-- gen_example_op.py
`-- executor_example_op.py
```

`README.md` 里的 TilingKey 覆盖表必须有"实际选择证据"列（host tiling UT 或运行时记录），
不能只按输入条件推断。

### 3.10 `op_host/tests/<算子>_tiling_processor_test.cpp`

```cpp
TEST_F(ExampleTilingTest, OutputMaskSave)
{
    // 断言档位推导与 static_assert 的互斥/全覆盖关系，覆盖每个可达 key 与档位组合
}
```

## 4. 常见错误对照

| 错误写法 | 问题 | 正确写法 |
| --- | --- | --- |
| `Output("y").ParamType(OPTIONAL)` 表达"调用方可选" | def 与 L2 两份事实；后续加可选输出必须改 def（ABI） | 输出 `REQUIRED`；L2 用可空指针 + 档位掩码 |
| 在 `def` 里新增输出表达 V2 的新能力 | 触发 ABI 重检，V1 调用方受影响 | 新开 `aclnn<Op>V2`，新增输出只加在 V2 形参尾部 |
| `op_kernel/arch35/` 复制整套算子 | 两份实现必然漂移，同步/性能修复要改两处 | 只拆差异文件，文件名与根目录同名 |
| TilingKey 里编码平台 | key 组合爆炸，覆盖表无法维护 | key 只标场景族；平台用 `__CCE_AICORE__` 与 host 侧 arch 目录 |
| 模板实例手抄笛卡尔积 | 漏实例、组合不可审计 | `ASCENDC_TPL_*` 声明 + 宏逐层展开 |
| L2 里判断 `disable_recompute` | 策略下沉，L2 语义被框架细节污染 | 策略留在 Python/legacy，L2 只认输出指针 |
| Python 侧缓存 stream 指针 | 多线程/多 stream 下用错 stream | 每次 `_current_stream_ptr()` |
| 报错只写"参数错误" | 定位不到触发条件 | 打印实际 `Kdim/Vdim`、`layout`、`outputMask` 等实参值 |
| 测试只跑 `--task run` | 不产生精度结论 | 精度走 `--task accuracy`，反向走 `--task run` + `expected_return_code` |
| 只按输入条件声称覆盖了 TilingKey | 无法证明实际选中该 key | 补 host tiling UT 或运行时记录作为证据 |

## 5. 与本规范的对应

| 本文小节 | 规范章节 |
| --- | --- |
| §2、§3.1、§3.5 | [`../工程结构规范.md`](../工程结构规范.md) §2、§3.1、§4.1 |
| §3.2、§3.3、§3.4 | §3.2、§5.1 |
| §3.6、§3.7 | §4.2、§5.3 |
| §3.8、§3.9、§3.10 | §5.3、§6、§3.6 |

## 6. 使用方式

1. 新增算子：按 §1 建立目录，逐文件对照 §2 与 §3 填写，最后用 [`../工程结构规范.md`](../工程结构规范.md) §8
   的清单自查。
2. 迭代既有算子：先按 §4 的对照表检查本次改动是否踩到错误写法，再按规范 §5.2 判断该不该新开 V2。
3. 检视：把被检视文件与 §2 对照表逐行比对；发现的偏差按仓库代码检视输出规范写成检视意见，并标注
   "违反工程结构规范 §x.y"。
