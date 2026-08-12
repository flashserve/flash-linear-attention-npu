# ChunkKdaFwd A5 + GPU 分布式 ATK 归档

本文归档已经验证过的执行拓扑：在 A5 节点本机发起 ATK 任务，A5 作为 NPU DUT，
GPU 位于另一台机器的 Docker 容器中并通过 ATK server 提供 Torch FP64 真值和
同精度 Triton 对照。所有地址、端口、设备号和安装路径均使用占位符。

通用版本、case 范围、精度标准和复检规则见 [`../README.md`](../README.md)。

## 1. 已验证拓扑

```text
A5 host
  atk task
  `-- local NPU, physical device 6 -> logical device 0
  `-- HTTP -> GPU host:<GPU_HOST_PORT>
                    `-- Docker container:9090
                          `-- physical GPU 6 -> logical device 0
                          `-- atk server + CUDA Torch + Triton
```

任务在 A5 上发起，因此 NPU node 不填写 `--host`/`--port`。GPU node 必须填写
GPU 宿主机可达地址和宿主机映射端口，不要填写容器内部 IP。

两边必须使用同一版 ATK、同一提交的本目录测试资产和同一个 case JSON。A5 只需要
NPU 运行时和 `fla_npu.ops.ascendc`；GPU 容器只需要 CUDA Torch、Triton 和配置的
KDA Triton callable，不需要安装 CANN 或 NPU wheel。

## 2. 固定变量

后续命令使用以下占位符：

```bash
export REPO_ROOT=<repo_root>
export ATK_ENV=<atk_26_7_8_environment>
export GPU_HOST=<gpu_host_ip_or_name>
export GPU_HOST_PORT=<published_host_port>
export GPU_CONTAINER=<gpu_container_name>
export GPU_REPO_ROOT=<repo_root_in_container>
export TRITON_KDA_ROOT=<fla_org_or_compatible_triton_source_root>
```

推荐在两边都把仓库放到各自固定路径；路径本身不要求相同。ATK server 与 A5 发起命令
都从各自仓库的 `test/chunk_kda_fwd` 目录启动，输出路径使用相对路径。

## 3. GPU Docker 准备

优先在 Docker 边界只暴露物理 GPU 6，并将容器 9090 映射到宿主机端口：

```bash
docker run --rm -it \
  --name "$GPU_CONTAINER" \
  --gpus '"device=6"' \
  -p "${GPU_HOST_PORT}:9090" \
  -v <gpu_host_repo>:"$GPU_REPO_ROOT" \
  -w "$GPU_REPO_ROOT/test/chunk_kda_fwd" \
  <gpu_image> bash
```

若复用已有容器，先检查 GPU 和端口。运行中的容器不能补加端口映射；缺少映射时应按原镜像、
挂载和环境重建容器，或在安全策略允许时使用 host network。

```bash
docker inspect "$GPU_CONTAINER" --format '{{json .HostConfig.DeviceRequests}}'
docker port "$GPU_CONTAINER" 9090/tcp
docker exec "$GPU_CONTAINER" nvidia-smi -L
```

容器只暴露物理 GPU 6 时，它在容器内是逻辑设备 0。若容器仍能看到所有卡，则先设置
`CUDA_VISIBLE_DEVICES=6`，ATK 中仍使用重新编号后的 `--devices 0`。

## 4. GPU 容器环境与服务

进入容器后，激活 ATK 26.7.8 环境并检查 CUDA/Triton：

```bash
source "$ATK_ENV/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$GPU_REPO_ROOT:$TRITON_KDA_ROOT:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd "$GPU_REPO_ROOT/test/chunk_kda_fwd"
atk --version
python -c 'import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
python -c 'from fla.ops.kda.chunk_fwd import chunk_kda_fwd; print(chunk_kda_fwd)'
```

预期 ATK 为 `26.7.8`、`torch.cuda.is_available()` 为 `True`、可见卡数为 1。
若使用其他兼容实现，先设置：

```bash
export KDA_ATK_TRITON_CALLABLE=<python_module>:<chunk_kda_fwd_callable>
```

随后在前台启动 GPU server：

```bash
atk server \
  --host 0.0.0.0 \
  --port 9090 \
  --devices 0 \
  --name gpu_reference \
  --output_path ./atk_output/gpu_server \
  --plugin_path ./executor_chunk_kda_fwd.py \
  --timeout 8000
```

不要退出这个终端。日志应显示监听 `0.0.0.0:9090`，而不是只监听
`127.0.0.1:9090`。

## 5. A5 环境

在 A5 上加载 ATK、CANN、当前构建的 OPP 和仓内 Python 包。这里只暴露物理 NPU 6，
所以后续 ATK 使用逻辑设备 0：

```bash
source "$ATK_ENV/bin/activate"
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONPATH="$REPO_ROOT/torch_custom/fla_npu:$REPO_ROOT:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=<writable_cache_dir>
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd "$REPO_ROOT/test/chunk_kda_fwd"
atk --version
python -c 'import fla_npu; from fla_npu.ops.ascendc import chunk_kda_fwd; print(fla_npu.__file__)'
npu-smi info -i 6
```

若没有设置 `ASCEND_RT_VISIBLE_DEVICES=6`，则 `node --devices` 必须传物理编号 6；两种
写法只能选一种，不要设置可见卡后仍传 6。

## 6. 两端一致性与网络预检

在 A5 和 GPU 容器内分别执行，结果必须一致：

```bash
atk --version
sha256sum \
  atk_chunk_kda_fwd.json \
  chunk_kda_fwd.yaml \
  gen_chunk_kda_fwd.py \
  executor_chunk_kda_fwd.py
```

从 A5 检查 GPU 宿主机映射端口：

```bash
timeout 5 bash -c "</dev/tcp/${GPU_HOST}/${GPU_HOST_PORT}"
curl -fsS "http://${GPU_HOST}:${GPU_HOST_PORT}/openapi.json" >/dev/null
```

端口不可达时依次检查：容器内 server 是否仍在运行、是否监听 `0.0.0.0:9090`、
`docker port` 是否存在、宿主机防火墙和 A5 到 GPU 宿主机的路由。

## 7. 单 case 三路烟测

在 A5 的 `test/chunk_kda_fwd` 目录执行。远端 GPU 不与 A5 共享文件系统，因此
`--syc_dataset` 必须保留；分布式任务不要使用 `-sp`，使用 `-mt 1` 限制并发和显存：

```bash
export KDA_ATK_TRACE_SEED=1

atk node --name npu_dut --backend npu \
    --devices 0 \
    --output_path ./atk_output/kda_remote \
  node --name gpu_reference --backend gpu \
    --host "$GPU_HOST" \
    --port "$GPU_HOST_PORT" \
    --devices 0 \
    --is_compare true \
    --output_path ./atk_output/kda_remote \
  task \
    -c ./atk_chunk_kda_fwd.json \
    --task accuracy \
    --bm_device gpu \
    -p ./executor_chunk_kda_fwd.py \
    -s 250 \
    -e 251 \
    --save_data output \
    --syc_dataset \
    -mt 1 \
    -to 2000

unset KDA_ATK_TRACE_SEED
```

case 250 是 A5 `H=96, T=8192, chunk_size=64` 模型 case，GPU FP64 真值需要较多显存。
烟测前应确认目标卡空闲；显存不足时不要换卡并发跑，也不要改输入范围，先释放该卡上的其他
任务。只验证链路时可从 JSON 选择同属 A5 正向范围的小 shape case。

同一 case/seed 的日志必须出现三种角色：

```text
NPU: benchmark=False high_precision=False triton_control=False
GPU truth: benchmark=True high_precision=True triton_control=False
GPU control: benchmark=False high_precision=False triton_control=True
```

`gpu_benchmark` 目录保存 Torch FP64 真值；普通 `gpu_*` 目录保存同输入 dtype 的 Triton
对照；`npu_*` 目录保存 A5 DUT 输出。

## 8. 全量和单 case 定位

A5 正向范围为 `250-449`，ATK 的 `-e` 是开区间：

```bash
# 将烟测命令中的范围替换为：
-s 250 -e 450
```

定位某一条失败 case 时保留 `--save_data output`，并分析三路结果：

```bash
python ./analyze_atk_saved_outputs.py \
  ./atk_output/kda_remote \
  --case-id <case_id>
```

只验证 NPU 固定输入的二进制确定性，不需要启动 GPU server：

```bash
python ./stress_npu_determinism.py \
  --device 0 \
  --case-id <case_id> \
  --repeats 100
```

## 9. 常见失败

| 现象 | 原因与处理 |
| --- | --- |
| NPU 报 `No module named 'fla_npu'` | A5 的 `PYTHONPATH` 未包含 `torch_custom/fla_npu`，或当前 OPP/Python 包不是同一提交。按第 5 节重新加载。 |
| GPU 返回 `input.bin is not exists` / HTTP 404 | 远端节点看不到 A5 本地数据。确认任务带 `--syc_dataset`，两端 output path 可写，server 未切换工作目录。 |
| A5 进程尝试调用 CUDA，报 `_cuda_setDevice` | 分布式任务误加了 `-sp`，GPU node 被当成本地 backend 执行。移除 `-sp`，使用 `-mt 1`。 |
| ATK 版本相同但 executor 行为不同 | 版本一致不等于测试资产一致。比较第 6 节四个文件的 SHA256。 |
| GPU FP64 真值 OOM | 物理 GPU 6 未空闲或存在其他容器/进程。先释放资源；H96 长序列保持 `-mt 1`。 |
| 设备号不可用 | 物理设备经 `ASCEND_RT_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` 或 Docker 映射后会重新编号；ATK 使用映射后的逻辑编号。 |
| 能连接端口但任务立即失败 | 检查 GPU server 启动终端中的 Python traceback、CUDA Torch/Triton 导入和 callable 签名；发起端的 404 只是远端失败的包装。 |

结果归档和公开 PR/issue 只记录测试项、case 范围、通过/失败结论和必要的非敏感错误摘要，
不得记录服务器地址、账号、绝对路径、容器名、token 或内部日志路径。
