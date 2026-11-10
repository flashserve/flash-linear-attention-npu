# PR #1096 对齐 #1073 开发记录

- 基线：远程 main `606ffff7`（已包含 #1073），原 PR head `6d037ac7`。
- 目标：保留 A5 Engine 优化；状态调用与 main 对齐，移除 wrapper 克隆/递归及 ACLNN 非原位预拷贝。
- 设计：初态只读、最终状态按需独立分配；未写输出槽位未定义；输入数值和版本计数不变；所有模式拒绝 None。
- 影响范围：RecurrentKDA 私有 ACLNN 路径、文档和 mutation 回归；wrapper/Stable launcher 使用 main 原实现。
- 验证计划：CPU ABI/入口测试；隔离环境构建安装 A5 wheel；非零初态原位/非原位数值对照、输出开关、None/grad/version 契约；ATK 精度回归。
- CPU：12 项 ABI/header 测试和 14 项 Stable runtime 测试通过；`git diff --check` 通过。
- A5 环境：A5_ci / Ascend950PR，物理卡7映射逻辑0，CANN 9.2.0-beta.2，Python 3.11.17，torch 2.9.0+cpu，torch_npu 2.9.0.post6，ATK 26.9.8。
- 隔离目录：`/home/fazhenyao123/runs/pr1096-align1073-20261110`，源码在 `source/`，独立 venv 在 `venv/`。未覆盖共享环境。
- 构建：`FLA_NPU_SOC=ascend950 FLA_NPU_OPS=recurrent_kda python -m pip wheel --no-build-isolation --no-deps . -w ../dist`，并在独立 venv 内安装该 wheel；构建安装通过。
- 产物一致性：`verify_libcust_opapi_md5.py --run-package build_out/fla_npu_linux-x86_64.run` 通过，op_api MD5 `a1361b17caf57e60dc13a5a09b4778d8`，72个 kernel 对象全部匹配。wheel SHA256 `608db5f378b4cb3b10642102e9f31017d33db3d2adc3147bc7a351913e87259e`。
- 设备 mutation 回归：执行 `regression_mutation_contract.py` 的 `scenario_recurrent_kda()`，非零初态原位/非原位 out 与有效状态逐值相同，输入数值/版本、输出 alias/shape/dtype、requires-grad、四种标志组合的 None 拒绝全部通过。没有运行该脚本中的 GDR 场景。
- ATK 基础精度：`bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend950 -npu_device_id=0 -scope=accuracy`，1/1通过；日志 `atk.log`。
- ATK 长序列回归：更新后的 executor 生成200例（保留 case id/种子，将历史 None 零初态改为显式零张量），覆盖 BSND/TND、V/K-first、FP32/BF16 State、V128/V256、长序列、indexed/重复槽、空序列、非连续 view 和状态输出组合。使用安装态 Stable 入口，开启 `--gm_init_flag`，200/200通过，日志 `longseq-mixed.log`，报告由仓库 `check_atk_result.py` 独立检查通过。
- 长序列标准说明：当前 ATK 不支持生成器默认的 `cv_fused_double_benchmark`，第一次执行在比较阶段报 KeyError，不能计为通过。第二次仅在运行目录生成的 JSON 中沿用仓库基础包的 `mixed_tolerance_bm` 标准，未修改仓库标准或阈值；上述200例为 CPU 单标杆回归，不是双标杆验收。
- 边界：未写入的非原位输出槽位不参与数值比较，整个输入状态池单独检查不变；未把未定义输出当作零值或输入副本。
- 尚未完成：当前 head 的双平台 NPU CI、A2/A3 设备回归、完整 PR 性能/内存/确定性和 Example ST 验收。本记录是本轮契约对齐回归，不表示上述正式验收已通过；因此保留开发期记录。
