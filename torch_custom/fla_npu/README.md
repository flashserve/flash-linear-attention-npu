# fla_npu 适配层

`torch_custom/fla_npu` 把 Ascend C 算子变成可调用的 Python 接口。上层只用短名（不带 `npu_` 前缀），
由 `fla_npu/ops/ascendc/__init__.py` 的 `_strip_npu_prefix()` 统一导出：

```python
from fla_npu.ops.ascendc import chunk_fwd_o

out = chunk_fwd_o(...)
```

## 1. 新增算子适配

一次适配 = **新建 1 个算子文件 + 1 行 include + 2 行注册 + 1 个 Python wrapper + 1 行 public 名**。
**不需要先写一份 ctypes 适配**：ctypes 只是回退后端，没有它算子照样交付，这也是新算子的默认
形态（[接入指南 §8](../../docs/architecture/适配层接入指南.md)）。

### 1.1 交付件

```text
torch_custom/fla_npu/
├── csrc/src/stable_<op>.cpp          # 新建：文件名 = 算子名去掉 npu_
├── csrc/src/stable_ops.cpp           # 改：include 一行 + 注册两行
└── fla_npu/ops/ascendc/
    ├── _stable.py                    # 改：加一个真签名 wrapper
    └── __init__.py                   # 改：public 名加一行
```

public 函数名要和 schema 里的算子名一致，只改两个地方：

```python
# fla_npu/ops/ascendc/_stable.py
def npu_<op>(...):                      # 函数名 = schema 里的算子名
    ...

# fla_npu/ops/ascendc/__init__.py
_ASCENDC_OPS = (
    ...,
    "npu_<op>",                         # 加一行，公开名和短名都跟着导出
)
```

这样 `from fla_npu.ops.ascendc import <op>` 与 `npu_<op>` 都能用（短名自动去掉 `npu_` 前缀）。

### 1.2 交付件内容规范

| 文件 | 规范 |
| --- | --- |
| `stable_<op>.cpp` | **一个算子一个文件，用宏写，不写手写入口**：`kSchema_<op>` 形参 === `run_<op>` 形参 === `FLA_STABLE_EXEC` 实参 === aclnn 头文件顺序（`stream` 固定在最后）；申请输出（取维度用 `SIZE_OF(meta, dim)`，报错会带上实参名 + 调用点行号 + 实际 shape）+ 一条 `FLA_STABLE_EXEC`，算子私有的名表 / helper 也放这里 |
| `stable_<前缀>_common.cpp` | 只放**被 ≥2 个算子共用**的 helper，文件名带共享前缀（当前是 `stable_causal_conv1d_common.cpp`、`stable_fwd_h_common.cpp`）；在 include 列表里排在用它的算子之前 |
| `stable_ops.cpp` | 两件事：include 各算子文件（共享文件在前、其余按算子名排序）+ `m.def(kSchema_<op>)`、`m.impl("<op>", &boxed_adapter<run_<op>>)` 两行注册 |
| `_stable.py` | 真签名 wrapper（不要 `*args` / `**kwargs`），位置参数顺序与 schema 形参一致；字符串用 `_char_code`、host 数组用 `_host_ints`、stream 用 `_current_stream_ptr()` |
| `__init__.py` | `_ASCENDC_OPS` 加一行 public 名；算子会原地写参数时再登记 `MUTATED_ARGUMENTS`（必要时 `MUTATION_FLAGS`）。没有 ctypes 回退不用声明任何东西 |

参数类型对照、模板、门禁命令、设备回归矩阵和常见坑见
[适配层接入指南](../../docs/architecture/适配层接入指南.md)；「为什么必须用宏」（boxed kernel 的
输入所有权契约、手写入口漏引用导致 191 MiB 泄漏的事故）见
[适配层设计](../../docs/architecture/适配层设计.md) §6。

## 2. 常见问题

**写完适配，运行期报 dispatcher 找不到实现。** 所有适配文件由 `stable_ops.cpp` include 进同一个编译单元，
没人 include 的文件等于没编译——新文件必须加进 include 列表（`stable_coverage.py` 查这条）。

```c++
// stable_ops.cpp
#include "stable_causal_conv1d_common.cpp"   // 共享 helper 在前
#include "stable_causal_conv1d_fn.cpp"       // 其余按算子名排序
```

**缓存 stream 后复用。** vLLM 是多线程多 stream，每个 worker 线程有自己的 stream。512 上就是把取到的
stream pointer 存成了进程级缓存：另一个线程复用时，kernel 被投到不属于它的 stream 上，表现为非法地址
访问与执行顺序错乱，512 那次崩溃就是这么来的。适配代码不要自己取 stream，也不要把值存下来：

```python
# 对：每次调用现场取，作为最后一个实参
def npu_<op>(...):
    return _op("npu_<op>")(..., _current_stream_ptr())

# 错：存成进程级变量再复用
_STREAM = _current_stream_ptr()
```

**取 stream 的 accessor 和下发方式不配对。** 下发走 torch_npu 任务队列（vLLM `EXEC_NPU_CMD` 同路）时
要用不排空队列的 accessor，顺序由队列保证；内联直投时要用会排空队列的那把，否则 kernel 会插到已入队
任务的前面。配错在空队列上看不出来，在 vLLM worker 上会变成每次调用约 1 ms。这段逻辑固定在
`_stable.py::_current_stream_ptr()` 里，算子适配不需要感知，两个逃生阀是
`FLA_NPU_STABLE_STREAM=accessor` 与 `FLA_NPU_STABLE_LAUNCH=inline`。

**非连续输入被就地转连续。** 张量的 sizes / strides / storage offset 原样交给 `aclCreateTensor`
即可，适配层不额外做转连续操作：自己补一份连续拷贝会白白付出约 0.1 ms/次的代价。

```python
# 对：如实交出去，让算子按 strides 寻址
npu_causal_conv1d_update(..., conv_state, ...)

# 错：在适配层补一份连续拷贝
npu_causal_conv1d_update(..., conv_state.contiguous(), ...)
```

**报 `size_of dim N out of range`。** 这是输出 shape 规则按错的 rank 取维度（常是把 4-D 输入当 3-D 传）。
报错点名了**是哪一张张量**、**在算子哪一行读的维度**、以及它自己的维数和形状：

```text
fla_npu(stable): size_of dim 1 out of range at csrc/src/stable_causal_conv1d_bwd.cpp:74 (tensor weight_meta, ndim=1, shape=[1536])
```

四段各是什么：

- `dim 1`：要从这张张量上读第 1 维（从 0 开始数）。
- `tensor weight_meta`：读的是算子里的 `weight_meta`（schema 形参 `weight`）——调用点写的是
  `SIZE_OF(weight_meta, 1)`，宏把实参文本带进了消息。
- `ndim=1` / `shape=[1536]`：**这张张量**只有 1 维、长度 1536，所以读第 1 维越界。这是维数，不是「第几个输入 / 输出」。
- `at csrc/src/stable_causal_conv1d_bwd.cpp:74`：读这一维的那一行。

调用点只写 `SIZE_OF(meta, dim)`，名字和行号都由宏带上，不用额外传任何参数；实参里带逗号时自己加一层括号。
没传的可选入参直接点名：`(tensor g_meta is None / undefined)`。名字和行号都只来自**宏展开的那一行**，所以
`layout_math.h` 的 helper 一律不读维度，只回答"该读第几轴"，取维写回算子自己那行——`SIZE_OF(q_meta,
layout_math::token_axis(layout))` 报出来的就是 `q_meta`，而不是 helper 的形参名。

**换了新产物却没生效。** launcher 由 `torch.ops.load_library()` 在 torch 初始化之后加载，`fork`
出来的子进程要重新加载；构建戳（`_stable_hash.py` 的 `SOURCE_HASH` 与 `.so` 内嵌哈希）不一致时加载
直接报错。那是防「跑了旧产物还不自知」，重新编译而不是绕过：

```bash
python3 torch_custom/fla_npu/csrc/build_stable.py --out /tmp/libfla_npu_stable.so --no-debug-probe
```

**装了多个版本 / 多张不同 SoC 的机器。** `flash-linear-attention-npu` 同名包互覆盖，并存要用独立 venv。

**发布机比目标机新。** 适配层与 OPP 的 host 侧库都在构建机上编，构建机更新时目标机会在 `import fla_npu`
时报 `GLIBCXX_3.4.x not found`——pip 的 manylinux 标签只承诺 glibc，看不出这条。发布前用
`tools/stable_abi_audit.py --lib` 查一次水位。

## 3. 测试要求

- **host 下发性能**：不劣化，不引入毫秒级开销。
- **编译期不绑定版本**：产物不绑 torch / torch_npu / Python 版本，任意满足最低要求的版本都能直接用同一个 wheel。
- **接口兼容性**：`fla_npu.ops.ascendc.xxx` 的公开签名、原地语义与返回值保持兼容。
