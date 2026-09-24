# ChunkKdaBwd V2 ATK 精度工程

本目录归档 `npu_chunk_kda_bwd(..., implementation="optimized")` 的临时交付矩阵。
四个执行文件为 `atk_chunk_kda_bwd.json`、`chunk_kda_bwd.yaml`、
`gen_chunk_kda_bwd.py` 和 `executor_chunk_kda_bwd.py`。

## 标杆与比较方式

NPU 运行实际算子，CPU 使用独立解析反向公式，以 FP64 计算、FP32 输出作为唯一 golden。
JSON、YAML 和生成器统一使用仓库的 `mixed_tolerance_bm`，没有覆盖比较器的默认阈值。
CPU 不根据 ATK benchmark 角色切换到低精度参考；低精度公式仅保留用于 CPU 自检。
输入从固定种子在 CPU 生成，各节点使用相同的量化输入。saved 模式使用相同保存缓存；
recompute 模式的 CPU golden 独立重建缓存，不读取 DUT 输出或缺省缓存。

CPU 自检包含短序列逐 token autograd 对照、dense/packed 等价性、可选输出及重计算缓存污染检查。
修改比较器不等于已完成混合容差的值域校准或正式精度验收。

## 输入与覆盖

- Ascend950；K=V=128、chunk_size=64，主输入 BF16。
- beta、raw_g、A_log 覆盖 BF16/FP32；可选 bias 和 Q/K L2 反向。
- dense 与 packed 各 100 条；saved 与 recompute 各 100 条。
- recompute 要求 H 是 8 的倍数且不超过 256，各非空序列长度是 64 的倍数。
- initial_state/dht 不提供，dh0=None；可见输出为 dq/dk/dv/db/dg/dA 及可选 dbias。
- 用例序号 0–199；确定性使用全部 200 条，内存检测使用序号 0–49。
- 精确 shape、dtype、seed、属性、原始 case id 和预期 Finalize key 保存在生成器及 JSON 的 case_spec 中。

每组连续四条 `4*i..4*i+3` 分别为 dense saved、dense recompute、packed saved、packed recompute。
Finalize 预期 key：dense 无 L2=1、packed 无 L2=2、dense L2=3、packed L2=4；
全矩阵及前 50 条均包含这四类。生成器会检查这些覆盖约束；预期 key 不替代实际运行证据。
历史交付版本的前 50 条 sanitizer 记录覆盖四个 key，新归档没有重新采集 key 证据。

矩阵保留此前按双标杆结果筛选的 100 saved + 100 recompute 用例；没有为混合容差再次筛选、
缩小值域或调整随机种子。它不包含完整负向测试和每个逻辑 case 的三个种子，不能称为全域验收。
本次仅归档精度矩阵，尚未补齐仓库正式验收要求的独立 `_perf.json`、`_mss.json` 及对应映射。

## 执行

准备当前构建安装的 wheel、CANN 和 ATK。仓库统一脚本要求 ATK >=26.8.8；
26.7.8 在此比较器上会报 `KeyError: 'mixed_tolerance_bm'`，不可绕过版本门槛当作通过。

在仓库根目录运行：

```bash
python tests/atk/chunk_kda_bwd/gen_chunk_kda_bwd.py --check
python tests/atk/chunk_kda_bwd/executor_chunk_kda_bwd.py --self-test
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -soc=ascend950 \
  -npu_device_id=1 -scope=accuracy
```

精度命令不指定范围，执行 JSON 全部 200 条。如手工调用 ATK，`-s 0 -e 200` 的结束边界不包含在内。
用例生成使用 `GEN_CASES_DTYPE_NUMBERS=200 GEN_CASES_EXTRA_NUMBERS=0 GEN_CASES_SEED=20260921`
配合统一脚本的 `-scope=gen_cases`；生成器的 `--check` 校验提交 JSON 与固定矩阵一致。
在独立 perf/MSS 用例包补齐前，不使用统一脚本的 `-scope=all`。

## 2026-09-22 验证记录

- CPU 标杆自检、JSON/生成器一致性检查通过。
- A5 私人环境 ATK 26.7.8 的混合容差命令在加载标准时失败，尚未进入 DUT 精度比较。
- 随后使用服务器上可访问的 ATK 26.9.8 wheel 创建独立环境，保持相同 JSON、输入与默认容差：
  200 条均执行成功，**83 条精度通过、117 条精度失败**，不是正式精度验收通过。
- saved 通过 69/100，recompute 通过 14/100。失败输出为 dA（83 条）、db（71 条）、
  dbias（18 条），同一 case 可有多个失败输出；dq/dk/dv/dg 的混合容差检查全部通过。
- 同环境以原版 fwd JSON/executor 运行第一个 case（id=250），混合容差 1/1 Pass。
- ATK wheel SHA256：`f1ea03ad9310943d2dceef7428ba10325599a9f6c8fa9e79891ca1cd58c01b09`；
  被测算子 wheel SHA256：`c3e71043f93bcac53c0c61c9be7d4f8b9345fc834011883ff9f14ab66b624bda`，
  已核对安装的四个 Finalize 对象与该 wheel 一致。
- 用户反馈此前双标杆精度、确定性和内存检测完成且无问题；该反馈作为历史记录，
  不改记为本次混合容差通过，也不据此宣称 sanitizer 零 WARNING。
- 既有精度失败仍待解决；本次没有改 kernel、输入值域或输出 dtype。
  CPU golden 的 ATK 传输统一为 FP32，与仓内单标杆工程一致。

ATK 日志、数据、Excel 和 profiling 产物保留在验证目录，不提交到仓库。
