# Phase6 A2/A3 内部实现

本目录只承载 `Ascend910B`（A2）和 `Ascend910_93`（A3）的 Phase6 私有实现。代码从
冻结提交 `chw@e0fb8942336efcf3b2e13d7463e63b5b5d38f341` 的独立算子路径复制，公共
`ChunkGatedDeltaRuleFwd` 入口不再通过同级算子目录获取实现。

维护约束：

- `arch22` 不包含或引用 `arch35` 文件；A5 代码不得在此目录修改。
- A2/A3 精度修复只进入本目录，并以冻结的 A2 双 500 ATK 基线回归。
- 独立算子仍可保留公开注册，但 Phase6 不依赖其源码路径或构建产物。
- 公共 ABI、输入输出顺序和 tiling 序列化布局不因内部化改变。

## Phase6 FP32 Solve

- DAV_2201（A2）统一从 KKT epilogue 产生的 `aWorkspace` 直接进入私有
  `solve_tri_fp32` pipeline，所有 `BT=64/128`、定长/变长模式都使用 head-first
  `[B,H,T,W]` 寻址；A2 不再经过 TND staging。
- 变长任务仍由 `cu_seqlens` 解码，dense 任务仍使用固定长度算术；块内列偏移继续使用
  局部 `row % BT`，不改变任务描述或 `BT=128` 的额外 merge 层级。
- `solve_layout_staging.h` 及其地址变量只在非 220 编译分支保留，A3 fallback 行为不变；
  A5 不使用本目录实现。
- 私有 pipeline 不改变 host tiling、workspace trailer、公开 ABI 或输出顺序。`Run` 自身
  保留首部/阶段间/末尾同步及 ready/free 闭环。
