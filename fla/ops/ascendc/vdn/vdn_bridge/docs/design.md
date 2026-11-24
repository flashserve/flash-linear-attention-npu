# VdnBridge 设计

方案设计规则版本：V2

工程结构参考 ENGINEERING-STRUCTURE V1。每个任务拥有一个 `(frame, head)` 的输出矩阵，全部 AIV 以同一算法跨步遍历任务。Host 根据平台 AIV 核数及 `C*H` 设置 blockDim，传递实际 F、H、start、C。两个模板键分别对应 FP32 输出和 BF16 输出。帧数、头数均是动态尺寸。

设备从两侧状态库读取两个矩阵，对 `prefix[lo]` 使用 `exp(logs[f+1]-logs[lo])`，对 `suffix[hi+1]` 使用 `exp(logs[hi+1]-logs[f])`。每次处理 64 行，按列广播两组 128 通道权重，先 FP32 相加，再按模板写回。每个输出只由一个 AIV 写入。临时数据位于 UB，workspace 为 0 字节。

## Stage 0 · Vector 窗口合并

每核先从 GM 加载四个 128 元素的 FP32 对数前缀向量，计算左右权重并在两次行块迭代间复用。每个 64×128 行块读 prefix、suffix 各 32 KiB。MTE2→V 事件保护广播乘和加；BF16 模板在加法后按 RINT 转换。V→MTE3 事件保护输出写回，每行块写出 FP32 32 KiB 或 BF16 16 KiB。下一行块重用缓冲区前等待完成写回。

UB [0,32 KiB) 和 [32,64 KiB) 保存两侧行块，[64,66 KiB) 保存四个小向量。从 66 KiB 起按输出元素宽度预留一个行块，BF16 占 16 KiB，FP32 预留 32 KiB。87 帧、7 头且 frame_tile=64 时，两次调用分别覆盖 448 和 161 个有效 frame/head；在 48 个 AIV 上分别每核处理 9–10 和 3–4 项。补齐帧不进入此阶段。
