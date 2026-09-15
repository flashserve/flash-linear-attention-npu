# flash-linear-attention-npu 算子测试工程

> 本文件是仓上算子测试工程的统一使用入口, 定义测试资产结构、工具版本、算子索引和执行命令.

## 1. 测试工程简介

本仓算子测试采用 ATK (Ascend Test Kit) 原生工程和命令行使用方式. 开发者直接使用 `atk case`, `atk node ... task` 等 ATK 命令生成和执行用例, 不在仓内建设另一套测试框架.

### 四类算子测试资产

每个算子只维护以下四类文件, 集中放在 `test/<op_name>/` 目录下:

| 文件 | 职责 |
| --- | --- |
| `atk_<op_name>.json` | 看护用例 JSON, 保存已评审的典型/边界/负向/性能等 case |
| `<op_name>.yaml` | 泛化 YAML, 声明参数域/关系/生成策略/精度标准 |
| `gen_<op_name>.py` | 泛化约束生成 gen, 将 YAML 约束转换为 ATK CaseGen 可消费的规则 |
| `executor_<op_name>.py` | 执行适配 executor, 构造输入 tensor, 调用 DUT, 提供 golden/benchmark |

### 目录树

```text
test/
├── readme.md                    # 本文件, 全仓测试统一入口
├── spec.md                      # 测试工程方案说明书
├── chunk_bwd_dqkwg/
│   ├── atk_chunk_bwd_dqkwg.json
│   ├── chunk_bwd_dqkwg.yaml
│   ├── gen_chunk_bwd_dqkwg.py
│   └── executor_chunk_bwd_dqkwg.py
└── ... (其他算子按相同结构组织)
```

### 支持的 SOC

| SOC | 代号 | 说明 |
| --- | --- | --- |
| A2 | `ascend910b` | 已验证 |
| A3 | `ascend910_93` | 已验证 |
| A5 | `ascend950` | 已验证 |

### 支持的调用路径

| route | DUT | 说明 |
| --- | --- | --- |
| `ascendc` | `fla_npu.ops.ascendc.<op_name>` | Python 稳定入口, ctypes 直调 aclnn |
| `aclnn` | aclnn GetWorkspaceSize + execute 两段式接口 | aclnn 原生接口 |
| `direct_launch` | Ascend C `<<<>>>` 直调 | 暂未实现, 待 ATK 社区提供通用 backend |

### 结果规则

- 进入执行矩阵的用例必须实际执行, 最终只能成功或失败.
- 环境/设备/工具/executor/标杆/结果文件或解析失败均按失败处理.
- 不产生跳过数/预期失败数或独立错误数.
- 不能通过缩小 case 范围/修改阈值/删除失败 case 制造通过结论.

## 2. 工具版本表

| 工具 | 需要记录的内容 |
| --- | --- |
| ATK | 官方仓库 [AECG/atk](https://gitcode.com/AECG/atk), 锁定 tag 或完整 commit, ATK version, Python 版本 |
| CT | 正式获取地址待补充 (未检索到可公开核验的发行地址) |
| CANN | 已验证版本范围 (待补充) |
| 驱动/固件 | 已验证版本范围 (待补充) |
| Python | 3.8+ |
| SOC | A2/A3/A5 验证情况如上表 |

工具升级必须单独评审, 并重新执行代表性固定 case.

## 3. 算子索引

### chunk_bwd_dqkwg

| 项目 | 内容 |
| --- | --- |
| 算子名 | `chunk_bwd_dqkwg` |
| 目录链接 | [test/chunk_bwd_dqkwg/](./chunk_bwd_dqkwg/) |
| 公开 API | `fla_npu.ops.ascendc.npu_chunk_bwd_dqkwg` |
| aclnn 接口 | `aclnnChunkBwdDqkwg` |
| 支持 route | `ascendc`, `aclnn` (`direct_launch` 暂未实现) |
| 支持 SOC | `ascend910b`, `ascend910_93`, `ascend950` |
| 看护 JSON case 数量 | 17 |
| 泛化 profile | dtype: fp16/bf16; g_dtype: fp32/bf16/fp16; K=128; V={128,256}; chunk_size={64,128}; B=[1,128]; T=[1,32768]; HK=[1,64]; n_ratio={1,2}; 定长/变长; is_mix={True,False} |
| golden 来源 | CPU fp32 参考实现 (`chunk_bwd_dqkwg_cpu`) |
| benchmark 来源 | CPU fp64 参考实现 (`chunk_bwd_dqkwg_cpu`, `benchmark=True`) |

## 4. 快速使用

### 4.1 ATK 获取和安装

```bash
git clone https://gitcode.com/AECG/atk.git <atk_source_dir>
git -C <atk_source_dir> checkout <locked_tag_or_full_commit>

python3 -m venv <atk_venv_dir>
source <atk_venv_dir>/bin/activate
python -m pip install --upgrade pip
python -m pip install -r <atk_source_dir>/atk/ATK-dev/requirements.txt
python -m pip install <atk_source_dir>/atk/ATK-dev

atk --version
```

### 4.2 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <atk_venv_dir>/bin/activate

# 确认已安装与目标 SOC 匹配的算子包
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python scripts/check_packaged_wheel_api.py

# 确认环境变量
echo $ASCEND_CUSTOM_OPP_PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
```

### 4.3 泛化用例生成

```bash
cd test/chunk_bwd_dqkwg

atk case \
  -f ./chunk_bwd_dqkwg.yaml \
  -p ./gen_chunk_bwd_dqkwg.py \
  -dt 1 \
  -en 0 \
  -s 20260815
```

生成后检查: ATK case schema 校验成功, 合法与非法用例数量符合 profile, coverage report 覆盖必选 dtype/layout/边界/SOC/功能分支.

### 4.4 全量精度测试

```bash
cd test/chunk_bwd_dqkwg

atk node --backend npu --devices 0 -o ./atk_output \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy \
  -p ./executor_chunk_bwd_dqkwg.py \
  -sp \
  -to 2000
```

### 4.5 单 case 定位

```bash
cd test/chunk_bwd_dqkwg

# 例如执行 case 0
atk node --backend npu --devices 0 -o ./atk_output_single \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy \
  -p ./executor_chunk_bwd_dqkwg.py \
  -s 0 \
  -e 1 \
  --save_data output \
  -sp \
  -to 2000
```

### 4.6 三条调用路径

通过 case ID 白名单选择 route 对应的 case:

```bash
cd test/chunk_bwd_dqkwg

# ascendc route case IDs: 0-6, 9-16
atk node --backend npu --devices 0 \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[0,1,2,3,4,5,6]' \
  -sp \
  -to 2000

# aclnn route case IDs: 7, 8
atk node --backend npu --devices 0 \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[7,8]' \
  -sp \
  -to 2000
```

`direct_launch` 路径待 ATK 社区提供通用 backend 后再补充.

### 4.7 负向用例

```bash
cd test/chunk_bwd_dqkwg

# 负向 case IDs: 9, 10, 11, 14
atk node --backend npu --devices 0 task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task run \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[9,10,11,14]' \
  -sp \
  -to 2000
```

必须同时检查: ATK 总任务数/success/failed, 每条 case 的 expected return code, executor 捕获的异常类型/返回码/关键错误信息.

### 4.8 NaN 脏数据

```bash
cd test/chunk_bwd_dqkwg

# NaN 脏数据 case IDs: 14
atk node --backend npu --devices 0 -o ./atk_output_nan \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[14]' \
  --save_data output \
  -sp \
  -to 2000
```

### 4.9 确定性

```bash
cd test/chunk_bwd_dqkwg

# 确定性 case IDs: 12
atk node --backend npu --devices 0 \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy_dc \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[12]' \
  -sp \
  -to 2000
```

### 4.10 精度复检

```bash
cd test/chunk_bwd_dqkwg

atk node --backend npu --devices 0 -o ./atk_output_recheck \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task accuracy_lt \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[<failed_case_ids>]' \
  --loop_nums 50 \
  --disable_id_seed \
  -mt 64 \
  -to 2000
```

`accuracy_lt` 不使用 `-sp`, 避免阻塞多轮调度. 白名单整体加引号, 例如 `-wl '[61,96,97]'`.

### 4.11 性能测试

```bash
cd test/chunk_bwd_dqkwg

# 性能 case IDs: 13
atk node --backend npu --devices 0 -o ./atk_output_performance \
  node --backend cpu task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task performance_device \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[13]' \
  --performance_data 20,100,80 \
  --save_data profile \
  -sp \
  -to 2000
```

`--performance_data` 的 warmup/采集次数/统计次数由测试团队按资源冻结.

### 4.12 mssanitizer

前置检查:

```bash
# 确认算子使用 sanitizer 选项编译
nm <operator_object_file> | grep sanitizer
```

执行方式 (以 memcheck 为例):

```bash
cd test/chunk_bwd_dqkwg

ATK_BIN=$(command -v atk)

mssanitizer --tool=memcheck --log-file ./mssanitizer_memcheck.log -- \
  "$ATK_BIN" node --backend npu --devices 0 task \
  -c ./atk_chunk_bwd_dqkwg.json \
  --task run \
  -p ./executor_chunk_bwd_dqkwg.py \
  -wl '[<sanitizer_case_ids>]' \
  -ms \
  -msl ./mssanitizer_memcheck.log \
  -sp \
  -to 2000
```

其他工具只替换 `--tool` 和日志名:

```bash
mssanitizer --tool=racecheck --log-file ./mssanitizer_racecheck.log -- <atk_command>
mssanitizer --tool=initcheck --log-file ./mssanitizer_initcheck.log -- <atk_command>
mssanitizer --tool=synccheck --log-file ./mssanitizer_synccheck.log -- <atk_command>
```

### 4.13 CT 精度可视化

> CT 正式获取地址待补充.

精度失败 case 先用 ATK `--save_data output` 保存 DUT 和标杆输出, 再执行:

```bash
ct viz <atk_saved_output_or_case_dir>
```

双标杆聚合分析:

```bash
ct dual analyze <atk_recheck_result.xlsx>
```

## 5. 看护 JSON case 说明

### chunk_bwd_dqkwg case 索引

| ID | name | route | tag | 说明 |
| --- | --- | --- | --- | --- |
| 0 | `aclnn.chunk.bwd.dqkwg.fp16.dense.small` | ascendc | accuracy, regression | fp16 定长小规模 |
| 1 | `aclnn.chunk.bwd.dqkwg.bf16.dense.small` | ascendc | accuracy, regression | bf16 定长小规模 |
| 2 | `aclnn.chunk.bwd.dqkwg.fp16.dense.chunk128` | ascendc | accuracy, boundary | fp16 chunk_size=128 边界 |
| 3 | `aclnn.chunk.bwd.dqkwg.bf16.dense.v256` | ascendc | accuracy, boundary | bf16 V=256 边界 |
| 4 | `aclnn.chunk.bwd.dqkwg.fp16.varlen.small` | ascendc | accuracy, boundary | fp16 变长场景 |
| 5 | `aclnn.chunk.bwd.dqkwg.fp16.gva.nratio2` | ascendc | accuracy, boundary | fp16 GVA n_ratio=2 |
| 6 | `aclnn.chunk.bwd.dqkwg.bf16.varlen.tail` | ascendc | accuracy, boundary, dirty_data | bf16 变长非整除尾块 |
| 7 | `aclnn.chunk.bwd.dqkwg.fp16.aclnn.route` | aclnn | accuracy, regression | fp16 aclnn 路径 |
| 8 | `aclnn.chunk.bwd.dqkwg.bf16.aclnn.route` | aclnn | accuracy, regression | bf16 aclnn 路径 |
| 9 | `aclnn.chunk.bwd.dqkwg.negative.k.invalid` | ascendc | negative | K=64 非法 |
| 10 | `aclnn.chunk.bwd.dqkwg.negative.v.invalid` | ascendc | negative | V=64 非法 |
| 11 | `aclnn.chunk.bwd.dqkwg.negative.hv.not.divisible` | ascendc | negative | HV 不是 HK 整数倍 |
| 12 | `aclnn.chunk.bwd.dqkwg.fp16.determinism` | ascendc | determinism, regression | fp16 确定性 |
| 13 | `aclnn.chunk.bwd.dqkwg.bf16.performance` | ascendc | performance | bf16 性能 |
| 14 | `aclnn.chunk.bwd.dqkwg.fp16.dirty_data.nan` | ascendc | dirty_data, negative | fp16 NaN 脏数据 |
| 15 | `aclnn.chunk.bwd.dqkwg.fp16.large.batch` | ascendc | regression | fp16 大 batch |
| 16 | `aclnn.chunk.bwd.dqkwg.bf16.varlen.large` | ascendc | regression, boundary | bf16 变长大规模 |

## 6. 注意事项

1. **资产集中**: 所有 ATK 单算子测试资产只能出现在 `test/` 目录, 不得在算子实现目录下新增同类资产.
2. **四类文件**: 每个算子只维护看护 JSON, 泛化 YAML, 约束生成 gen, executor 四类文件, 不得为精度/性能/Sanitizer/NaN/确定性/复检分别复制 executor 或 case 文件.
3. **文档一致性**: YAML 和 gen 中的 dtype/shape/layout/属性/可选输入/平台差异/非法组合必须与算子 README/设计文档/API 文档一致.
4. **direct_launch**: 当前缺少 ATK 通用 backend, `direct_launch` 路径暂未实现. 待 ATK 社区提供通用 backend 后再补充.
5. **产物清理**: ATK 生成的 `result/`, `atk_output/`, 日志, XLSX, profiling 和 sanitizer 产物不得提交.
6. **结果判定**: shell 返回码为 0 但 ATK 报告存在 failed case 时, 整体仍为失败.
7. **性能判定**: 不使用 Python wall time 直接下结论, 以 ATK profiling/CI 结果为准.
8. **精度失败**: 不能通过收窄输入 range/跳过 case/降低覆盖强度/放宽阈值来制造通过结论, 应先定位误差来源.
