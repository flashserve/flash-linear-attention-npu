# experimental —— 实验算子目录

本目录存放**尚未稳定、随时可能增删**的算子。它们与稳定树（`fla/ops/ascendc/`）**结构同构**，
由同一个开关 `--experimental` 二选一进入构建：**开启时只编本目录，关闭时完全不参与**。

设计说明见 [`docs/architecture/experimental 构建适配设计.md`](../docs/architecture/experimental%20构建适配设计.md)。

## 目录约定

镜像稳定树的仓库相对路径，即 `experimental/` 后面接原本的完整路径：

```
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/          ← 稳定树
│
└── 实验版放在 →
experimental/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── chunk_fwd_o_def.cpp
└── op_kernel/
```

**为什么必须镜像**：`<op>_depends` 里写的是仓根相对路径（如
`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h`），实验模式下框架统一在它前面拼
`experimental/`。只有路径同构，拼接结果才真实存在；否则依赖会被**静默跳过**（不报错）。

## 三条硬性约束

| 约束 | 原因 |
| --- | --- |
| 不要在本目录**根层**放 `CMakeLists.txt` | 发现逻辑用 `GLOB_RECURSE …/experimental/CMakeLists.txt` 递归匹配；根层那份会被当成一个名叫 `experimental` 的算子 |
| `experimental/<...>/common` 若是分叉副本，必须是稳定树的**超集** | 框架在实验模式下只复制实验树那份到 `src/common`；稳定树独有的头不会进来 → 编译期缺头，且不报「路径不存在」 |
| 实验算子只能依赖**实验树内**的算子 | 前缀会把依赖指到实验树；实验树里没有 → 静默丢依赖 |

## 使用

```bash
# 编一个实验算子
bash build.sh --pkg --experimental --soc=ascend910b --ops=<算子名>

# 编全部实验算子
bash build.sh --pkg --experimental --soc=ascend910b
```

开启后走的是**实验包**，与稳定包是两个独立产物，不会合并。

## 已知限制

- 与 `--pkg-type=rpm/deb` 互斥（另注：`--pkg-type=all` 会绕过该检查，请自行避免）。
- 与 `--static` 未做适配。
- Python / wheel 侧（`torch_custom/fla_npu/`）目前不感知 `--experimental`。
- 当前 CI 没有实验 job，这条链尚未在 CI 上被验证。
