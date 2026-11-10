# OPAPI 符号归属与解析性能

## 修复目标与边界

`dlsym(vendor_handle, name)` 同时查找 vendor 和其依赖。overlay 未实现 Add，
但依赖 CANN `libopapi_math.so` 时，旧 runtime 会提前命中 CANN Add，跳过
FLA base。修复保持 `overlay → base → CANN`，overlay 及未验证库只接受自己
定义的 ACLNN 接口；已验证的末位 base 沿用原 base/CANN 查找。本修复的运行时
解析不要求新接口出现在 base 表中。
安装层允许新增接口及替换 overlay 的策略仍由 PR #1132 处理；本 PR 独立于它，
不修改安装准入规则。

影响范围是全部 Stable AscendC adapter（公共 exec 路径及 RecurrentKda 专用路径）。
不改变公开签名、计算公式、
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

## 验证范围

原生 Linux ELF 回归编译生产 runtime，覆盖依赖遮蔽、overlay/base/CANN 优先级、
预加载、同名文件、软链接、新接口、两阶段不完整、缓存和 CANN 两阶段归属。
已验证 base 通过链接器包装计数确认不调用 dlinfo/dladdr1；旧包、无效 manifest、
手工库、错误登记位置及重复句柄不绕过严格检查。Python loader 和包校验另有回归。

手写 adapter 的头文件检查同步识别 op_api()，保证 RecurrentKda 继续进入检查范围。
新接口不需要出现在旧 base 表中；本次不修改 overlay 安装准入政策。
