# OPAPI 符号归属与调用诊断

## 修复目标与边界

`dlsym(vendor_handle, name)` 同时查找 vendor 和其依赖。overlay 未实现 Add，
但依赖 CANN `libopapi_math.so` 时，旧 runtime 会提前命中 CANN Add，跳过
FLA base。修复保持 `overlay → base → CANN`，overlay 及未验证库只接受自己
定义的 ACLNN 接口；已验证的末位 base 沿用原 base/CANN 查找。本修复的运行时
解析不要求新接口出现在 base 表中。
安装层允许新增接口及替换 overlay 的策略仍由 PR #1132 处理；本 PR 独立于它，
不修改安装准入规则。

影响范围是全部 Stable AscendC adapter（公共 exec 路径及 RecurrentKda 专用路径），
以及仓内全部 AscendC ACLNN/kernel 入口的可选诊断。不改变公开签名、计算公式、
TilingKey、数据布局、workspace、同步或 CPU 标杆。

## 符号选择

1. 包加载器先验证 base 的 `abi_manifest.json`、二进制/头文件哈希及完整的两阶段记录，
   再由实际加载的 Stable 库登记该 base 的绝对路径。缺少或无效的 manifest 不授予
   快速路径，仍用严格归属检查；这不是新增的 base 接口准入门禁。
   不以库数量或单独的环境变量开关推断已验证身份。校验在包加载阶段执行一次；
   其成本与接口首次解析成本分别测量，不能视为整个启动过程没有新增工作。
2. 仅当已登记的 base 是解析列表最后一项时，保留原 base/CANN 的 `dlsym` 查找，
   不再逐接口调用 `dlinfo/dladdr1`。base 自身的两阶段完整性由上述包校验证明；
   base 依赖中的 CANN 接口沿用原回退行为。登记必须先于 native runtime 初始化，
   不支持进程运行中更换 OPP 文件或切换 base。
3. 对 overlay 和未验证的库，用 `dlinfo(RTLD_DI_LINKMAP)` 取得当前句柄的 ELF 对象，
   `dladdr1(RTLD_DL_LINKMAP)` 取得候选函数的实际 ELF 对象，比较对象身份。
   不比较 `.so` 文件名，不受同名文件、软链接或相对路径影响。
   只有当前 vendor 自身的 ACLNN 函数可以命中；依赖提供的同名函数继续向下查找。
   ACL descriptor 等非 ACLNN 基础符号保留原依赖搜索行为。
4. 正式执行使用成对解析接口；同一 vendor 必须同时提供 GetWorkspaceSize 和执行函数。
   只提供一个阶段时明确报错，避免静默混用不同 executor 契约。
5. 所有 custom vendor 都未提供时才显式查 CANN，并验证阶段来源一致。
6. 缓存解析结果，严格路径的 ELF 身份核对只发生在首次解析，不进入每次 kernel 的执行逻辑。

## 解析性能验证

在 A5 主机上，对同一个真实 base/CANN 库和 `aclnnRecurrentGatedDeltaRuleV2`，
交替执行主线、原严格修复、已验证 base 快速路径。单线程绑定同一 CPU，15 个独立进程
取中位数；每进程预热后重复 100 万次缓存查询。首次解析计时排除库加载，缓存计时
包含适配器使用的字符串构造和取得两阶段地址，不调用 ACLNN/kernel，也不执行打印。

Stable 构建脚本实际使用 `-O2`；相同 `-O2` 编译下：

| 指标 | 主线 | 严格修复 | 已验证 base 快速路径 |
| --- | ---: | ---: | ---: |
| 首次解析 | 6.610 μs | 13.670 μs | 5.018 μs |
| 缓存命中，单次取得两阶段 | 63.071 ns | 28.905 ns | 28.517 ns |
| 4 线程并发摊销耗时 | 188.827 ns | 95.431 ns | 93.684 ns |

另按此前的 `-O3` 微基准设置对比：主线/严格版/快速路径首次分别为
5.919/15.433/6.550 μs，缓存命中为 77.093/31.159/31.172 ns。
这组首次结果仍有小幅回退，不能推广为所有编译配置、主机和场景都零退化。
NPU kernel 的既有优化等级没有调整。

快速路径还要求一次包加载校验：校验 1,925,784 字节的 host 库、对应头文件哈希及
76 个符号的成对记录，本机文件缓存已热时 15 次中位数为 **4.128 ms**。
base 路由只需要这些证据，不重复解析头文件签名；overlay ABI 校验仍保留完整签名检查。
该新增加载成本不包含在上表中，因此不声明进程完整启动时间没有回退。
失效的 manifest 回退严格路径，不允许通过修改环境变量直接关闭归属检查。

## 调用诊断与性能

提供默认关闭的构建开关 `FLA_NPU_OP_TRACE`。开启时，手写和生成的 ACLNN
入口都输出 `[FLA][ACLNN]`，kernel 入口由 block 0 输出 `[FLA][KERNEL]`。
kernel 标记来自实际 device 代码，host 符号归属不能代替这项证据。
源文件路径和函数名用于区分 FLA 与 CANN，同步完成后检查 device 打印。

关闭时宏在预处理阶段展开为空，不保留环境变量读取、条件分支、字符串、计数或打印调用。
不修改优化级别，保持原 `-O3` kernel 构建选项；开启打印仍有开销，不用于性能测量。
ELF 地址诊断只能证明 host 入口；profiling/加载路径可作辅助证据，不能单独证明
某次调用执行了 FLA kernel。

## 验证计划

- 原生 Linux ELF 回归：依赖遮蔽、overlay/base/CANN 优先级、加载顺序、路径别名、
  非 ACLNN 依赖、缺失符号、新增接口、两阶段不完整和缓存行为。
- 诊断开关 ON/OFF：检查所有仓内入口覆盖；关闭时比较优化后的代码/标记，
  开启时检查 ACLNN 与实际 kernel 的输出。
- A5/CANN 9.2.0：构建并安装隔离环境产物，用重名 RecurrentGatedDeltaRule 验证
  符号路径和实际 kernel 标记；按已有测试入口执行受影响回归。
- 未运行的平台/算子明确记录，不将 host 模拟或符号地址检查当作 NPU 精度验收。

方案检视：本次只修改公共解析和编译期诊断，不涉及算子 Stage/资源重新设计；
满足原接口和两阶段 executor 生命周期约定。正式测试结果随 PR 更新。

## 使用方法

构建开关必须在编译时设置，运行时 export 同名变量不会改变已经生成的二进制：

```bash
# 全部算子：保留正常优化等级，启用诊断。也可用 --ops=... 缩小编译范围。
FLA_NPU_OP_TRACE=1 bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu

# 重新构建正式包，关闭新增诊断；build.sh 会显式重置 CMake 缓存中的开关。
FLA_NPU_OP_TRACE=0 bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu
```

按开发者指南安装完整匹配的 wheel/OPP，并重启 Python 进程；初始化 CANN 前准备
`ASCEND_CUSTOM_OPP_PATH`。通过现有 ATK/公开接口调用算子，同步设备后检查：

```text
[FLA][ACLNN] aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize file=...
[FLA][ACLNN] aclnnRecurrentGatedDeltaRuleV2 file=...
[FLA][KERNEL] recurrent_gated_delta_rule file=...
```

ACLNN 每次进入都打印，kernel 的 block 0 打印（混合核场景可来自多个执行单元）。
打印入口只证明执行进入该层，不证明整个算子成功；仍须检查同步结果、返回码和精度。
设备打印的输出/落盘方式遵循 CANN 的 printf 配置；不能因只看 stdout 没有输出就
断言未执行。正式性能测量必须重新构建关闭诊断的包。

手写 ACLNN 和 kernel 在源码入口放置宏；op_build 生成的 ACLNN 由
`scripts/util/instrument_aclnn_trace.py` 在生成后插入同一宏。未提供仓内实现的
基础 CANN 算子（例如 FastGelu）不会伪造 `[FLA]` 标记。

原理参考：[Linux dlinfo](https://man7.org/linux/man-pages/man3/dlinfo.3.html)、
[Linux dladdr](https://man7.org/linux/man-pages/man3/dladdr.3.html)、
[Ascend C printf](https://asc.gitcode.com/api/Utils-API/tuning_interface/printf.html)。
printf 开启时有运行开销；`-O3` 不会消除必须发生的 I/O。默认宏在预处理阶段删除
新增诊断，既不切换到 `-O0`，也不依赖运行时判断来实现关闭。

## 构建开关的传递

| 项目 | 行为 |
| --- | --- |
| shell 输入 | `FLA_NPU_OP_TRACE` 未设置或为空时为 `0`；仅接受 `0`、`1`，其它值退出并报错 |
| CMake | 同名 option 默认 `OFF`；直接 CMake 可传 `-DFLA_NPU_OP_TRACE=ON`，`build.sh` 则每次按环境值显式设置 ON/OFF |
| host | ON 时定义 `FLA_NPU_OP_TRACE=1`，覆盖手写和 op_build 生成的 ACLNN |
| kernel | ON 时追加 `-DFLA_NPU_OP_TRACE=1 -DASCENDC_DUMP=1`，保留原优化和同步选项 |
| 子构建 | `config.cmake → prepare.sh → CMake` 传递同一值，避免 host 开启而预编译 kernel 仍关闭 |
| 选项合并 | 多条 `ALL` 编译选项按顺序累加，避免后续诊断选项覆盖已有 `NOT_DYNAMIC_COMPILE` 等选项 |
| 打包 | kernel 公共头放在既有 `ascendc/common` 复制/安装路径，不依赖构建机器上的绝对 include 路径 |
| 回退 | 重新构建 `FLA_NPU_OP_TRACE=0` 的匹配包并重启调用进程；运行时修改环境变量不生效 |

该开关不选择算法、平台或兼容后端。诊断包的设备端 printf 使用 CANN 的调试缓冲区，
因此开启时除打印时间外，还可能增加调试资源占用。关闭时只删除本 PR 新增的诊断，
不会强制关闭算子原有的其它调试功能。

## A5 验证记录

基线为 main `959d6b198db8dba79dd0d928f9af2025b19f62ec`，使用隔离 Python 环境和
CANN 9.2.0；原 Python 环境及 CANN 安装未修改。

- Linux 原生 ELF 测试直接编译生产 `runtime.cpp`，验证依赖遮蔽、完整两阶段选择、
  新接口无需 base 记录、同文件名/软链接及 CANN 回退。
- 真实 CANN `libopapi_math.so` 的 DT_NEEDED 复现场景中，安装校验通过的 Demo overlay
  不包含 Add；修复后 Add 两阶段都选中 FLA base，实际标记函数分别返回 `1100/1101`。
- RecurrentGatedDeltaRule、CausalConv1d 与 RecurrentKda 各执行既有 ATK 第 0 条精度用例并通过。
  CausalConv1d 第一次在 ATK 显存预填充阶段 OOM；关闭 `ATK_GM_INIT_MODE` 后通过，
  不将初始化失败记作算子执行失败或精度通过。
- 独立 Python 进程复用上述三个 ATK executor、调用公开 Stable 入口并同步设备，
  在默认 CANN 日志配置下都观察到 `[FLA][ACLNN]` 和真实 `[AIV Block 0/…] [FLA][KERNEL]`。
  多进程测试的输出捕获可能看不到设备 printf；复现诊断时使用独立进程并正常退出，
  给 CANN 异步打印输出留出时间。固定等待时间不是验收标准，实际标记和同步成功才是。
- 最终全量诊断包完成 34 个顶层算子的编译/打包；191 个 kernel 对象均含诊断标记，
  逐一与安装目录比较 SHA-256 全部一致，host 库也一致。上述三个算子在该最终包上复验打印通过。
- 显式列出全部 `--ops` 时，现有末尾筛选检查会因组合算子
  `chunk_gated_delta_rule_bwd` 没有独立 ACLNN 调用点而报错；不将该整条命令记为成功。
  对最终包头文件单独执行不带 `--ops` 的完整检查：34 个 ACLNN 调用点，0 mismatch/缺失头文件。
  手写 adapter 检查同步支持 `op_api()`，避免 RecurrentKda 从检查范围中遗漏。
- 新增 16 项原生 ELF/诊断测试及 74 项 Stable gate/runtime/build policy 测试通过。
- base 快速路径更新后，原生解析、包验证、Stable、安装/overlay 和诊断回归合计
  162 项通过。链接器包装计数验证已登记 base 不执行 `dlinfo/dladdr1`；
  未验证单库、非末位登记、overlay 依赖遮蔽及不完整两阶段仍受严格检查。
  失效二进制/头文件哈希或不完整 manifest 只拒绝快速路径资格，不新增 base 准入拦截。
- 重建 Stable launcher 并安装匹配的 Python loader 后，三个既有 ATK 第 0 条精度用例
  再次通过；三个入口均取得实际 host/kernel 标记，FastGelu 显式前后向与 CANN 逐位一致。
- 关闭诊断后重新编译两算子包：host 库及 6 个 kernel 对象均无新增标记，
  实际执行 RecurrentGatedDeltaRule/CausalConv1d 并同步成功，无新增打印；非法开关值按预期拒绝。
- 关闭宏时的最小 host/kernel 宏样例在 GCC `-O3` 下与不含宏的样例汇编逐字节一致。
  这是新增宏关闭路径的证据，不等同于全部 NPU 算子的性能实测结论。

以上是定向诊断与回归记录，不替代 A2/A3/A5 全部受影响算子的正式 ATK 验收及仓库合入 CI。
