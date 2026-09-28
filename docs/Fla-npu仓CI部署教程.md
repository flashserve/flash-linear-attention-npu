# NPU CI 部署教程

本文面向第一次接触 GitHub Actions self-hosted runner 的维护者，目标是部署本仓库按需触发的 A2 + A5 双平台 NPU CI。一次触发会并行创建 A2、A5 两个设备 job；每个平台内部按顺序执行分项检查，任一平台缺席或失败都不能通过汇总门禁。

示例服务器路径使用 `/workspace/flash-linear-attention-npu-ci`。示例仓库地址使用 `https://github.com/flashserve/flash-linear-attention-npu`。

## 先看重点

下面这些点最容易踩坑，建议先读完再操作。

1. **Docker 容器必须带 `--privileged`。**
   本仓 `ci/run_ci_container.sh` 默认已经启用 `CI_DOCKER_PRIVILEGED=true`，最终会执行带 `--privileged` 的 `docker run`。如果没有它，容器里常见现象是 `npu-smi` 不可用，或者 `torch_npu.npu.device_count()` 返回 `0`。

2. **宿主机物理 NPU 和容器内逻辑 NPU 不是同一个编号。**
   脚本会在宿主机上自动选择一张物理卡，例如 `ASCEND_RT_VISIBLE_DEVICES=2`。进入容器后，这张卡通常映射成逻辑设备 `0`。所以 CI 中传给 PyTorch example 的设备号默认是 `CI_CONTAINER_DEVICE=0`。

3. **Example ST 必跑，shape 由用例文件统一管理。**
   CI 默认执行 `ci/example_st_cases.json` 中 `enabled=true` 的用例。当前 `case1_current_default` 保持原始 shape；后续 GVA、`Vdim=256` 等场景通过新增用例显式配置 `B`、`T`、`chunk_size`、`query_head`、`value_head`、`Kdim`、`Vdim` 等 shape 字段，以及 `gate_source`、`gate_function`、`initial_state`、`output_final_state`、`qk_l2norm` 等行为字段。

4. **A2、A5 使用各自验证过的 CANN/PyTorch 组合。**
   A2 保持 CANN 9.1.0、PyTorch 2.7.1、`torch_npu` 2.7.1.post5；A5 使用 CANN 9.1.0、PyTorch 2.12.0、`torch_npu` 2.12.0。不要在两个平台之间复用 wheel 或 CANN 镜像。

5. **`triton-ascend` 和社区版 `triton` 不要共存。**
   A2 镜像安装 `triton-ascend==3.2.1`，A5 镜像安装 `triton-ascend==3.2.1`，并清理社区版 `triton`。如果容器里又装了社区版 `triton`，可能出现 `torch_npu`/`triton` namespace 重复注册问题。

6. **runner registration token 很短时效，且只能由 GitHub 管理员生成。**
   如果下载 runner 很慢，token 可能在真正注册前过期。过期后回到 GitHub 页面重新生成一个 token，再重新执行注册脚本即可。

7. **runner 标签必须和 workflow 完全匹配。**

   | 平台 | 必需标签 |
   | --- | --- |
   | A2 | `linux,arm64,npu,flash-linear-attention-npu` |
   | A5 | `linux,x64,npu,ascend950,flash-linear-attention-npu-a5` |

   A5 必须使用独立产品标签，不能只注册成通用 `npu` runner，否则无法证明一次触发确实覆盖了两个平台。

8. **NPU CI 不会因为 PR 新建、重新打开、push 新 commit 自动执行。**
   PR 新建、重新打开、push 新 commit 后，会自动出现 7 个 pending 分项状态，描述为“未执行”，用于阻止未验证 commit 合入。真正执行 NPU CI 仍需要仓库 Admin 权限账号点击 GitHub Actions 按钮，或在 PR 评论里发送 `/run-npu-ci quick` / `/run-npu-ci full` 触发。PR 更新 commit 后，旧 commit 上的 CI 成功状态不会再满足合入门禁。

9. **分支保护里的状态检查名称要完全一致。**

   | 顺序 | 必需状态检查 | 可执行分项的 `CI_STAGE` |
   | --- | --- | --- |
   | 01 | `NPU CI / A2+A5 / 01 环境、wheel 与运行时契约` | `environment-contracts` |
   | 02 | `NPU CI / A2+A5 / 02 全量 OPP 构建` | `opp-package` |
   | 03 | `NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局` | `standalone-layout` |
   | 04 | `NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配` | `torch-adapter` |
   | 05 | `NPU CI / A2+A5 / 05 GDR Example/ST` | `gdr-example-st` |
   | 06 | `NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装` | `scoped-overlay` |
   | 07 | `NPU CI / A2+A5 / 07 报告与 commit 校验` | 不适用，由汇总 job 执行 |

   每个平台按 01 到 06 顺序执行。前置分项失败后，后续分项会标记为未执行，不会继续运行；失败诊断会保留关键错误，并给出带对应 `CI_STAGE` 的复现命令。07 由双平台汇总 job 最后执行并发布。除这 7 项外，还应把自动执行的 `CI 契约测试` 配置为必需检查，用于验证 CI 代码本身。所有名称都必须精确匹配；名字写错、大小写不同或空格不同，GitHub 都会认为没有通过。

10. **`weinachuan` 强行合入 bypass 不是写在代码里自动生效的。**
    需要 GitHub 管理员用 `scripts/github/apply_branch_protection.sh` 应用到 GitHub 分支保护规则。

11. **多个 NPU CI 同时触发时有两层保护。**
    同一 PR 的触发先通过 workflow concurrency 串行化，再由 `prepare` 根据带 `A2+A5` 契约的状态去重。每台 runner 宿主机还会用 `/tmp/fla-npu-ci-npu-<id>.lock` 对物理 NPU 加锁，避免多个任务抢同一张卡。

12. **CI 容器内日志默认只保留 7 天。**
    每次 NPU CI 容器启动时都会执行 `ci/cleanup_ci_logs.sh`，删除超过 7 天的 CI/Ascend/NPU 日志。这个策略只管理 CI 容器和仓库工作目录里的日志，不等同于 GitHub Actions 网页日志保留时间。

13. **A5 源码通过 GitHub API 归档下载。**
    A5 runner 不依赖 `github.com` 的 Git smart-HTTP：workflow 会从 `api.github.com` / `codeload.github.com` 下载精确 test-merge commit 和可信 CI commit，并校验 test-merge 的父提交必须对应 prepare 固定的 base/head。归档解包前会拒绝绝对路径、路径穿越、重复成员、链接和特殊文件。A5 网络策略至少需要允许这两个 HTTPS 域名。prepare 会读取目标分支当前 tip 并固定为不可变 SHA；评论触发由默认分支分发到选定的目标分支，可信 CI 脚本统一使用该分支的 `github.workflow_sha`，因此维护分支继续使用自身规则，CI 自身改动也能在合入前验证。

    归档解包与用例文件读取的安全边界测试仅依赖 Python 和仓库 requirements，可在 Linux 环境执行：

    ```sh
    python3 ci/tests/test_workflow_archive_security.py
    ```

## 流程图

PR 新建或更新默认状态流程：

```mermaid
flowchart TD
    A["PR 新建、重开或 push 新 commit"] --> B["NPU CI 默认状态 workflow 运行"]
    B --> C{"当前 head commit 是否已有执行结果"}
    C -- "有通过、失败或运行中状态" --> D["保留已有 NPU CI 状态"]
    C -- "没有执行结果" --> E["写入 7 个分项 pending 状态"]
    E --> F["机器人评论：NPU CI 未执行"]
    F --> G["提示可请求触发的 CI 账号和 /run-npu-ci 命令"]
```

评论触发总流程：

```mermaid
flowchart TD
    A["仓库 Admin 在 PR 评论 /run-npu-ci"] --> B["默认分支评论分发 workflow"]
    B --> C{"触发人是否具备仓库 Admin 权限"}
    C -- "否" --> D["直接失败：无权触发"]
    C -- "是" --> E["选择 ci_ref；未指定时使用 PR base.ref"]
    E --> F["workflow_dispatch 到目标分支 ci.yml"]
    F --> G["准备 NPU CI：解析 PR 和 head commit"]
    G --> H{"当前 commit 是否已通过 NPU CI"}
    H -- "是" --> I["更新机器人评论：已通过，不重复运行"]
    H -- "否" --> J{"当前 commit 是否已有 NPU CI pending/running"}
    J -- "是" --> K["更新机器人评论：已在运行，不重复启动"]
    J -- "否" --> L["写入 7 个分项 pending 状态"]
    L --> M["A2 runner / ascend910b"]
    L --> N["A5 runner / ascend950：API 下载并校验源码归档"]
    M --> O["A2 顺序执行 01-06；失败后跳过后续"]
    N --> P["A5 顺序执行 01-06；失败后跳过后续"]
    O --> Q["上传带本次 run 身份的分项和精度结果"]
    P --> Q
    Q --> R["GitHub-hosted finalize 校验并发布 7 个最终状态"]
```

同一 PR + 同一 commit 重复触发流程：

```mermaid
flowchart LR
    A["第一次评论"] --> B["创建 pending 状态"]
    B --> C["runner 开始或排队运行"]
    D["第二次评论"] --> E["prepare 读取当前 head commit 状态"]
    E --> F{"已有 active pending?"}
    F -- "是" --> G["只更新同一条机器人评论"]
    G --> H["不启动新的 runner job"]
    F -- "否" --> I["允许重新触发"]
```

runner 宿主机 NPU 加锁流程：

```mermaid
flowchart TD
    A["runner 进入 ci/run_ci_container.sh"] --> B["扫描 npu-smi，生成候选 NPU 顺序"]
    B --> C["尝试 flock /tmp/fla-npu-ci-npu-<id>.lock"]
    C --> D{"是否拿到锁"}
    D -- "是" --> E["设置 ASCEND_RT_VISIBLE_DEVICES=<id>"]
    E --> F["启动 Docker 容器运行编译和 Example ST"]
    F --> G["Docker 退出"]
    G --> H["释放 NPU 锁"]
    D -- "否" --> I{"是否还有候选卡"}
    I -- "是" --> C
    I -- "否" --> J["等待 CI_NPU_LOCK_RETRY_SECONDS 后重新扫描"]
    J --> B
```

## 你需要准备什么

- A2、A5 各一台可运行 self-hosted runner 的 Ascend NPU 服务器，例如 `<a2-ci-host>` 和 `<a5-ci-host>`
- A2 服务器是 Linux/aarch64，A5 服务器是 Linux/x86_64；两者都已安装匹配的 Ascend 驱动，宿主机能执行 `npu-smi info`
- 两台服务器都已安装 Docker，并且 runner 服务账号可以执行 `docker`
- GitHub 仓库管理员权限
- 一个能访问 GitHub 的网络环境
- 本仓库代码
- CI Docker 镜像：A2 使用 `fla-npu-ci:9.1.0-910b`，A5 使用 `fla-npu-ci:9.1.0-950`

## 第 1 步：登录服务器并准备目录

以下目录命令需要在 A2、A5 runner 上分别执行。示例路径只是占位，实际部署应使用各主机允许写入且容量充足的目录。

```sh
ssh <npu-ci-host>
```

创建 CI 工作目录：

```sh
mkdir -p /workspace/flash-linear-attention-npu-ci
cd /workspace/flash-linear-attention-npu-ci
```

如果普通用户没有权限创建 `/workspace`，请让服务器管理员创建目录，或使用：

```sh
sudo mkdir -p /workspace/flash-linear-attention-npu-ci
sudo chown -R "$USER":"$USER" /workspace/flash-linear-attention-npu-ci
```

## 第 2 步：放置本仓库代码

有 GitHub 访问权限时，直接 clone：

```sh
cd /workspace/flash-linear-attention-npu-ci
git clone https://github.com/flashserve/flash-linear-attention-npu.git context
cd context
git checkout main
```

如果服务器不能访问 GitHub，也可以从一台能访问 GitHub 的机器把仓库目录同步到：

```text
/workspace/flash-linear-attention-npu-ci/context
```

确认关键脚本存在：

```sh
ls ci/run_ci_container.sh ci/setup_self_hosted_runner.sh ci/detect_npu.sh
```

## 第 3 步：检查宿主机 NPU

先确认宿主机能看到 NPU：

```sh
npu-smi info
```

再用本仓脚本检查自动选卡结果：

```sh
cd /workspace/flash-linear-attention-npu-ci/context
bash ci/detect_npu.sh --summary
```

你应该能看到类似输出：

```text
Detected NPU devices:
  - id=0 name=910B3 health=OK free=0 soc=ascend910b
  - id=2 name=910B3 health=Alarm free=1 soc=ascend910b
Selected NPU: id=2 name=910B3 health=Alarm free=1 soc=ascend910b
```

A5 的 `npu-smi` 设备行可能在设备号后带独立分隔符，例如 `| 0 | Ascend950PR | OK | ...`。`ci/detect_npu.sh` 同时支持这种格式和 A2 的紧凑格式；A5 摘要中的 `soc` 应为 `ascend950`。可以在无 NPU 的开发机上运行解析回归测试：

```sh
bash ci/tests/test_detect_npu.sh
```

`health=Alarm` 不一定会阻止 CI，因为默认 `CI_REQUIRE_HEALTHY_NPU=false`。如果你希望只允许健康卡运行，可以在 runner 环境里设置 `CI_REQUIRE_HEALTHY_NPU=true`。

查看 runner 会按什么顺序尝试加锁：

```sh
bash ci/detect_npu.sh --candidates
```

CI 会优先尝试健康且空闲的 NPU，然后尝试空闲但健康状态异常的 NPU。真正占用前还会拿 `/tmp/fla-npu-ci-npu-<id>.lock` 文件锁，拿不到锁时会继续尝试下一张卡。

## 第 4 步：准备 Docker 镜像

两个平台的镜像不能互换：

| 平台 | 镜像 | Dockerfile | 关键版本 |
| --- | --- | --- | --- |
| A2 | `fla-npu-ci:9.1.0-910b` | `ci/Dockerfile` | CANN 9.1.0、Python 3.12、PyTorch 2.7.1 |
| A5 | `fla-npu-ci:9.1.0-950` | `ci/Dockerfile.ascend950` | CANN 9.1.0、Python 3.12、PyTorch 2.12.0 |

推荐在 runner 维护窗口从受信任的默认分支构建或加载镜像，再启动 runner。构建命令如下：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

# A2 / arm64
docker build --platform=linux/arm64 \
  -t fla-npu-ci:9.1.0-910b \
  -f ci/Dockerfile .

# A5 / x86_64
docker build --platform=linux/amd64 \
  --network=host \
  -t fla-npu-ci:9.1.0-950 \
  -f ci/Dockerfile.ascend950 .
```

如果通过 tar 包分发，加载后必须检查准确 tag 和架构：

```sh
docker load -i <ci-image.tar>
docker image inspect <ci-image> \
  --format 'id={{.Id}} arch={{.Architecture}} created={{.Created}}'
```

A5 job 固定设置 `CI_REQUIRE_PRELOADED_IMAGE=true`。若 `fla-npu-ci:9.1.0-950` 不存在，`ci/run_ci_container.sh` 会立即失败，不会在 PR job 中临时联网构建；同时设置 `CI_REBUILD_IMAGE=true` 也会被拒绝。镜像升级应在维护窗口完成构建、自检和原子换 tag，再触发 CI。A2 未设置该开关，保留原有的缺失时自动构建行为。

A5 Dockerfile 对 PyTorch 和 `torch_npu` wheel 做 SHA256 校验。默认 Python 包源可通过 `PYPI_INDEX_URL` build arg 覆盖，但不能跳过 wheel 版本和校验和约束。

A5 matrix 固定设置 `CI_TMPDIR=/tmp/fla-npu-ci`。A5 Triton/BiShengIR 编译器要求临时目录位于容器 `/tmp` 下；不要修改这个容器路径。`ci/run_ci_container.sh` 会默认把 runner 工作区内的 `.ci-tmp/container-tmp` bind mount 到该路径，使临时数据与 runner 工作区使用同一数据盘；job 结束后会校验真实路径并清理这个目录。手工运行时可用 `CI_HOST_TMPDIR` 覆盖宿主机临时目录，但 `CI_TMPDIR` 仍须是绝对容器路径。

## 第 5 步：验证容器能看到 NPU

先跑一个轻量检查。这里重点验证 `--privileged`、驱动挂载和容器内逻辑设备号：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

# 以下以 A5 为例；A2 改为 fla-npu-ci:9.1.0-910b。
export CI_IMAGE=fla-npu-ci:9.1.0-950

eval "$(bash ci/detect_npu.sh --env)"

mount_args=()
for path in \
  /usr/local/dcmi \
  /usr/local/bin/npu-smi \
  /usr/local/Ascend/driver/lib64 \
  /usr/local/Ascend/driver/version.info \
  /etc/ascend_install.info; do
  if [ -e "$path" ]; then
    mount_args+=(-v "$path:$path")
  fi
done

docker run --rm \
  --privileged \
  --network host \
  --ipc host \
  "${mount_args[@]}" \
  -e ASCEND_RT_VISIBLE_DEVICES="$NPU_SELECTED_DEVICE" \
  "$CI_IMAGE" \
  bash -lc 'npu-smi info && python3 - <<PY
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())
PY'
```

期望看到：

```text
torch_npu device_count: 1
```

如果这里是 `0`，优先检查：

- 命令里是否有 `--privileged`
- 宿主机 `npu-smi info` 是否正常
- `/usr/local/Ascend/driver/lib64`、`/usr/local/bin/npu-smi` 等路径是否存在
- `ASCEND_RT_VISIBLE_DEVICES` 是否选到了真实存在的 NPU

然后跑本仓 CI 入口做一次端到端检查：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

# A2
CI_IMAGE=fla-npu-ci:9.1.0-910b \
CI_SOC=ascend910b \
CI_MODE=quick \
CI_RUN_EXAMPLE_ST=true \
CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK=true \
CI_RUN_SCOPED_WHEEL_INSTALL_CHECK=true \
bash ci/run_ci_container.sh

# A5：必须提前准备镜像
CI_IMAGE=fla-npu-ci:9.1.0-950 \
CI_SOC=ascend950 \
CI_REQUIRE_PRELOADED_IMAGE=true \
CI_MODE=quick \
CI_RUN_EXAMPLE_ST=true \
CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK=true \
CI_RUN_SCOPED_WHEEL_INSTALL_CHECK=true \
bash ci/run_ci_container.sh
```

这个命令会：

- 自动扫描宿主机 NPU
- 用 `--privileged` 启动容器
- 挂载 `third_party` 缓存
- 按顺序检查环境、wheel 配置及运行时契约
- 为目标 SOC 构建全量 OPP run 包
- 检查独立 `torch_custom` wheel 与 OPP 安装布局
- 安装 OPP 并构建 PyTorch 适配
- 执行 `ci/example_st_cases.json` 中启用的 GDR Example/ST 用例，仅覆盖容器内逻辑设备号 `--device 0`
- 检查 `chunk_fwd_o` 局部 run 包覆盖 wheel OPP 的安装流程

上述分项串行执行。一个分项失败后，脚本会停止实际测试并把后续分项标记为未执行；不会为了收集更多结果继续执行依赖已失败的步骤。失败日志和汇总诊断会给出类似下面的复现命令：

`quick` 和 `full` 都只在第 02 项构建当前平台对应的目标 SOC，不会把其他安装或测试混入构建状态。未指定 `ops` 时，`full` 会在第 04 项追加全部 legacy PyTorch 适配测试。

```sh
CI_STAGE=opp-package \
CI_MODE=quick \
CI_SOC=ascend950 \
FLA_NPU_SOC=ascend950 \
CI_IMAGE=fla-npu-ci:9.1.0-950 \
CI_DOCKERFILE=ci/Dockerfile.ascend950 \
CI_REQUIRE_PRELOADED_IMAGE=true \
bash ci/run_ci_container.sh
```

`CI_STAGE` 可取 `environment-contracts`、`opp-package`、`standalone-layout`、`torch-adapter`、`gdr-example-st` 或 `scoped-overlay`。脚本会先执行该分项所需的前置分项，再执行目标分项；应优先使用失败诊断中给出的完整命令。

本地排障时可以只跑某个用例：

```sh
CI_EXAMPLE_CASE_FILTER=case1_current_default \
CI_MODE=quick \
CI_RUN_EXAMPLE_ST=true \
bash ci/run_ci_container.sh
```

正式 GitHub NPU CI 不暴露用例子集选择，默认跑全部 `enabled=true` 用例，避免只跑部分用例却满足合入门禁。

## 第 6 步：生成 self-hosted runner token

这一步必须由 GitHub 仓库管理员操作。

1. 打开仓库页面：`https://github.com/flashserve/flash-linear-attention-npu`
2. 点击顶部 `Settings`
3. 左侧点击 `Actions`
4. 点击 `Runners`
5. 点击 `New self-hosted runner`
6. 选择操作系统 `Linux`
7. A2 选择架构 `ARM64`，A5 选择架构 `X64`
8. 页面会显示一段安装命令，其中有一个 `--token xxxxxxxxx`，复制这个 token

不要把 token 发到公开聊天、PR、Issue 或日志里。它过期后重新生成即可。

## 第 7 步：注册 runner

回到 NPU 服务器执行：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <把刚才复制的 registration token 放这里>
```

上面的默认值用于 A2。A5 必须显式使用独立 runner 名称和完整标签：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

RUNNER_ROOT=/workspace/actions-runner/flash-linear-attention-npu-a5 \
RUNNER_NAME="$(hostname)-flash-linear-attention-npu-a5" \
RUNNER_LABELS=linux,x64,npu,ascend950,flash-linear-attention-npu-a5 \
bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <把刚才复制的 registration token 放这里>
```

脚本默认会把 runner 安装到：

```text
/workspace/actions-runner/flash-linear-attention-npu
```

默认 runner name 是：

```text
<hostname>-flash-linear-attention-npu
```

默认 labels 是：

```text
linux,arm64,npu,flash-linear-attention-npu
```

A5 labels 必须是：

```text
linux,x64,npu,ascend950,flash-linear-attention-npu-a5
```

如果脚本用 root 运行，会尝试安装并启动系统服务。完成后回到 GitHub 页面：

```text
Settings -> Actions -> Runners
```

确认 runner 状态是绿色 `Idle` 或 `Online`。

## 第 8 步：runner 没变绿怎么查

先在服务器上看 runner 目录：

```sh
cd /workspace/actions-runner/flash-linear-attention-npu
ls
```

如果已经安装成服务：

```sh
sudo ./svc.sh status
```

尝试重启：

```sh
sudo ./svc.sh stop
sudo ./svc.sh start
sudo ./svc.sh status
```

查看日志：

```sh
journalctl -u 'actions.runner.*' -n 200 --no-pager
```

如果没有安装服务，可以前台运行看日志：

```sh
cd /workspace/actions-runner/flash-linear-attention-npu
./run.sh
```

常见原因：

- registration token 过期：回到 GitHub 重新生成 token，再执行 `ci/setup_self_hosted_runner.sh`
- runner 下载太慢：等下载完成后重新生成 token，再执行注册
- labels 不匹配：A2 设置 `RUNNER_LABELS=linux,arm64,npu,flash-linear-attention-npu`；A5 设置 `RUNNER_LABELS=linux,x64,npu,ascend950,flash-linear-attention-npu-a5`
- 服务器不能访问 GitHub：检查 DNS、代理、防火墙
- runner 已注册但离线：重启 runner 服务

重新注册示例：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

RUNNER_LABELS=linux,arm64,npu,flash-linear-attention-npu \
bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <新的 registration token>
```

脚本会使用 `--replace` 替换同名 runner。

## 第 9 步：配置分支保护和 weinachuan bypass

这一步也需要 GitHub 仓库管理员权限。

推荐使用脚本应用分支保护：

```sh
cd /workspace/flash-linear-attention-npu-ci/context

export GITHUB_TOKEN=<具有仓库 Administration 写权限的 GitHub token>
bash scripts/github/apply_branch_protection.sh main
```

这个脚本会给 `main` 配置：

- 必需状态检查：本教程“先看重点”第 9 项列出的 7 个精确 context
- PR 至少 2 个 approval
- 需要 Code Owners review；ABI 敏感路径的 code owner 是 `weinachuan`
- Admin 也必须遵守分支保护
- push 新 commit 后旧 review 失效
- 需要最后一次 push 不是由审批人自己完成
- `weinachuan` 具备 PR bypass allowance
- 禁止 force push
- 禁止删除分支

如果你要手动在 GitHub 页面配置：

1. 打开仓库 `Settings`
2. 左侧点击 `Branches`
3. 找到 `Branch protection rules`
4. 编辑或新建保护规则，Branch name pattern 填 `main`
5. 勾选 `Require a pull request before merging`
6. Required approvals 填 `2`
7. 勾选 `Require review from Code Owners`
8. 勾选 `Dismiss stale pull request approvals when new commits are pushed`
9. 勾选 `Require approval of the most recent reviewable push`
10. 勾选 `Require status checks to pass before merging`
11. 勾选 `Require branches to be up to date before merging`
12. 添加本教程“先看重点”第 9 项列出的全部 7 个必需检查
13. 勾选让 administrators 也遵守分支保护，或不要让普通 Admin 具备全局 bypass
14. 在 bypass 或 pull request bypass allowance 中添加用户 `weinachuan`
15. 保存规则

如果 GitHub 页面里搜不到某个 status check，通常是因为该 check 还没有在仓库里跑过。可以先让 workflow 或状态检查执行一次，或者直接用脚本通过 API 设置。

## 第 10 步：验证触发方式

本仓 NPU CI 不会被 PR 自动执行。PR 新建、重开或 push 新 commit 时，GitHub 会自动给当前 head commit 写入 7 个 pending 分项状态，描述为“未执行”。这些状态会出现在 PR checks 中，用来提醒该 commit 还没有完成双平台 NPU CI。

如果 PR push 了新 commit，默认状态会重新写到新的 head commit 上，描述为“commit 已变化，请重新触发”。旧 commit 的 NPU CI success 不会给新 commit 使用。

机器人也会在 PR 评论区写入或更新一条提示，说明可以请求仓库 Admin 权限账号使用 `/run-npu-ci quick` / `/run-npu-ci full` 触发。

仓库 Admin 权限账号可以用两种方式手动触发真正的 NPU CI。

触发后，GitHub 机器人会在 PR 评论区写入一条 NPU CI 状态评论。刚触发时显示“已开始”，A2、A5 都结束后由 `finalize` 更新同一条评论为“通过”或“失败”，并给出双平台分项汇总。全部通过时只显示平台结论和精度用例计数，不重复展开成功 Tensor 的逐项指标；失败时才显示真实编译错误、运行异常、失败精度指标和带 `CI_STAGE` 的复现命令。每个平台的分项按顺序执行，前置失败后续标记为未执行。如果 Actions 页面能看到 workflow 已触发，但 PR 下没有机器人评论，检查仓库 `Settings -> Actions -> General` 中的 `Workflow permissions` 是否允许 workflow 请求写权限。本 workflow 通过 job 级 `permissions` 申请 PR 读取、issue comment 写入和 commit status 写入权限。

方式一：GitHub Actions 按钮。

1. 打开仓库 `Actions`
2. 选择左侧 `NPU CI`
3. 在 `Run workflow` 的 `Branch` 下拉框选择 CI 定义分支
4. 点击右侧 `Run workflow`
5. 填写：
   - `pr_number`: PR 编号，例如 `23`
   - `ci_mode`: `quick` 或 `full`
   - `ops`: 可选，逗号分隔的算子列表，仅用于编译定向诊断；填写后只执行第 01、02 项并发布 `NPU CI / A2+A5 / 定向诊断`，不会执行 GDR 精度，也不会写入或覆盖 01-07 正式门禁状态
6. 点击绿色 `Run workflow`

方式二：在 PR 评论区发送命令。

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
/run-npu-ci quick ci_ref=release-v2
```

带 `ops=` 的命令只顺序执行第 01 项环境契约和第 02 项指定算子 OPP 构建，并发布 `NPU CI / A2+A5 / 定向诊断`；它不执行 GDR 精度或后续安装分项，也不会写入或覆盖 01-07 正式门禁状态。用于合入门禁的运行必须省略 `ops`。

评论触发由默认分支上的 `NPU CI 评论分发` workflow 处理：没有指定 `ci_ref` 时自动使用 PR 的目标分支 `pr.base.ref`，指定后使用该分支的 `.github/workflows/ci.yml`。目标分支必须存在对应的 `workflow_dispatch`。只有仓库 Admin 权限账号可以触发；分发 workflow 和实际 NPU CI 都会调用 GitHub API 校验请求人权限。手动运行时，`Branch` 下拉框就是实际使用的 CI 定义分支。

PR 新建或 push 新 commit 后，当前 head commit 会先出现默认状态：

```text
NPU CI / A2+A5 / 01 环境、wheel 与运行时契约 pending
NPU CI / A2+A5 / 02 全量 OPP 构建 pending
NPU CI / A2+A5 / 03 torch_custom wheel 与 OPP 布局 pending
NPU CI / A2+A5 / 04 OPP 安装与 PyTorch 适配 pending
NPU CI / A2+A5 / 05 GDR Example/ST pending
NPU CI / A2+A5 / 06 chunk_fwd_o 局部覆盖安装 pending
NPU CI / A2+A5 / 07 报告与 commit 校验 pending
未执行：请维护者评论 /run-npu-ci quick
```

如果是 push 新 commit，上述 7 个状态会重新变为 pending，并显示“未执行：commit 已变化，请重新触发 /run-npu-ci quick”。

维护者触发执行后，上述 7 个状态仍会保持 pending，但描述会变成类似“等待执行 (quick,all,A2+A5)”。

成功后，上述 7 个状态都会变成 success。发生失败时，真正失败的分项会显示关键原因，依赖它的后续分项显示未执行；PR 评论和 Actions Summary 给出对应 `CI_STAGE` 复现命令。

PR 评论区也会出现或更新一条机器人评论，包含触发人、模式、Example ST 要求和 Actions run 链接。

只有当前 head commit 的同一次 Actions run 已让 7 个 A2+A5 分项状态全部通过，重复触发才会跳过，不再占用 NPU。旧版状态或单平台状态不会命中该去重条件。

如果当前 head commit 已经有 NPU CI 处于排队或运行中，重复评论也会跳过，不会再启动新的 self-hosted runner job。机器人评论会更新为“已在运行”，并指向已有 Actions run。

不同 PR 或不同 commit 的 NPU CI 可以同时被触发。每次触发都会分别调度一个 A2 job 和一个 A5 job；真正到各 runner 宿主机执行时，`ci/run_ci_container.sh` 会为选中的物理 NPU 加文件锁，默认锁文件是 `/tmp/fla-npu-ci-npu-<id>.lock`。如果所有候选 NPU 都被锁住，job 会等待并重试。

锁相关环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CI_NPU_LOCK_DIR` | `/tmp` | NPU 锁文件所在目录 |
| `CI_NPU_LOCK_WAIT_SECONDS` | `14400` | 等待空闲 NPU 锁的最长时间，`0` 表示一直等待 |
| `CI_NPU_LOCK_RETRY_SECONDS` | `10` | 所有 NPU 都被锁住时的重试间隔 |

日志清理相关环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CI_LOG_CLEANUP_ENABLED` | `true` | 是否在 CI 容器启动时清理历史日志 |
| `CI_LOG_RETENTION_DAYS` | `7` | 仅保留最近多少天的日志 |
| `CI_LOG_CLEANUP_DIRS` | 空 | 额外清理目录，多个目录用 `:` 分隔 |

默认会清理这些目录中超过保留天数的文件：仓库内 `log/`、`logs/`、`log_ut/`、`output/log*/`、`build/log*/`、`build_out/log*/`，以及容器内常见的 Ascend/NPU 日志目录。

## 第 11 步：合入前怎么判断是否满足门禁

合入 PR 前确认两件事：

1. 当前 head commit 上本教程“先看重点”第 9 项列出的 7 个状态全部是 success
2. GitHub 分支保护要求的 2 个 approval 已经满足

如果 PR 又 push 了新 commit，需要重新跑 NPU CI。旧 commit 的成功结果不能给新 commit 使用。

`weinachuan` 如果需要紧急强行合入，可以走 GitHub branch protection bypass，但前提是第 9 步已经把 bypass allowance 配好。

## 第 12 步：常见问题

### Actions 页面显示 queued，不开始跑

优先检查：

- runner 是否绿色 `Idle` 或 `Online`
- A2 runner labels 是否包含 `linux,arm64,npu,flash-linear-attention-npu`
- A5 runner labels 是否包含 `linux,x64,npu,ascend950,flash-linear-attention-npu-a5`
- workflow 的 `runs-on` 是否被改过
- runner 服务是否在运行

### Runner 页面没有绿色状态

优先检查：

- registration token 是否过期
- `setup_self_hosted_runner.sh` 是否执行到 `./config.sh`
- 服务器是否能访问 `github.com`
- runner 服务是否启动成功

### 容器里看不到 NPU

优先检查：

- `CI_DOCKER_PRIVILEGED` 是否是 `true`
- `docker run` 日志里是否有 `--privileged`
- 宿主机 `npu-smi info` 是否正常
- 容器内 `torch_npu.npu.device_count()` 是否为 `1`
- 是否把宿主机物理设备号当成容器内逻辑设备号使用。容器内通常应使用 `--device 0`

### A5 报预置镜像缺失

A5 不会在 PR job 中自动构建镜像。确认维护窗口已经构建或加载准确 tag，并确认 runner 服务使用的 Docker daemon 与检查命令一致：

```sh
docker image inspect fla-npu-ci:9.1.0-950 \
  --format 'id={{.Id}} arch={{.Architecture}} created={{.Created}}'
```

输出架构必须是 `amd64`。不要通过关闭 `CI_REQUIRE_PRELOADED_IMAGE` 临时绕过；应先修复 runner 的镜像或 Docker daemon 配置。

### 报 `Invalid device ID`

通常是容器内传了宿主机物理卡号。例如宿主机选择了卡 `2`，容器里应该传 `--device 0`，不要传 `--device 2`。

### 报 `torchnpugen` 缺失

说明镜像里的 `torch_npu` 不对。A2 应确认 Ascend PyTorch `v26.1.0-beta.1` 对应的 `torch_npu` wheel；A5 应确认 `torch_npu==2.12.0` 的 CPython 3.12 x86_64 wheel。不要混用平台、Python ABI 或 PyTorch 版本，也不要重新拉 `op-plugin` 编译。

### 报 `triton` namespace 或重复注册问题

检查是否同时安装了社区版 `triton` 和 `triton-ascend`。本仓 A2 CI 镜像只保留 `triton-ascend==3.2.1`，A5 CI 镜像只保留 `triton-ascend==3.2.1` 提供的 `triton` 模块。

### `.run` 安装后 Python 调用找不到自定义 op

检查日志里是否设置了 custom OPP 的 op_api lib：

```text
Custom OPP op_api lib: /usr/local/Ascend/.../opp/vendors/fla_npu_transformer/op_api/lib
```

如果没有，需要确认 `.run` 包安装成功，并且 `LD_LIBRARY_PATH` 包含这个目录。

### 评论触发没有反应

检查：

- 评论是否发在 PR 页面，不是普通 Issue 页面
- 命令是否以 `/run-npu-ci` 开头
- 发送评论的人是否具备仓库 Admin 权限
- `NPU CI 评论分发` workflow 是否启用
- PR 目标分支或 `ci_ref` 指定分支是否存在 `.github/workflows/ci.yml`
- 目标分支的 `ci.yml` 是否声明 `workflow_dispatch`
- Actions 页面是否先出现 `NPU CI 评论分发`，随后出现新的 `NPU CI` run

### PR 创建后没有 7 个 NPU CI 分项的未执行状态

检查：

- `NPU CI 默认状态` workflow 是否启用
- PR 是否是新建、重开、ready for review，或刚 push 了新 commit
- workflow 是否有 `statuses: write` 权限
- 组织或仓库策略是否允许 workflow 申请 `statuses: write` 权限
- Actions 页面是否出现新的 `NPU CI 默认状态` run

### Actions 已触发，但 PR 下没有机器人评论

检查：

- `Settings -> Actions -> General -> Workflow permissions` 是否允许 workflow 请求写权限
- 组织或仓库策略是否允许 workflow 申请 `issues: write` 和 `statuses: write` 权限
- workflow 日志里是否还有 `Resource not accessible by integration`

### PR 已合入但 Checks 历史里还有红色失败记录

旧版自定义检视门禁会在审批不足时让 Actions job 失败，GitHub 会保留这些历史 check run。当前版本已经删除该自定义门禁，后续不再产生这类等待审批的红色 job。

已经产生的历史红色 check run 不能被后续 commit status 覆盖删除。若必须清理旧记录，需要仓库管理员在 GitHub `Actions` 页面删除对应的旧 workflow run，或使用 GitHub API 删除旧 run。

### 多个 NPU CI 同时触发怎么办

同一 PR 的同一 head commit 如果已经有 NPU CI 在排队或运行，重复评论只会更新机器人评论，不会再启动新任务。

不同 PR 或不同 commit 同时触发时，GitHub Actions 可能把它们分配给不同 self-hosted runner。runner 宿主机会用 `flock` 给物理 NPU 加锁，避免多个任务抢同一张卡。默认锁文件如下：

```text
/tmp/fla-npu-ci-npu-0.lock
/tmp/fla-npu-ci-npu-1.lock
```

如果所有候选 NPU 都被锁住，job 会输出：

```text
[CI] All detected NPU devices are locked; retrying in 10s.
```

这不是失败，表示正在等待其他 NPU CI 释放卡。如果等待超过 `CI_NPU_LOCK_WAIT_SECONDS`，job 才会失败。

### CI 容器日志怎么保留

CI 容器启动时会自动执行：

```sh
bash ci/cleanup_ci_logs.sh
```

默认只保留 7 天内的日志。需要调整时，在 runner 环境或 workflow 环境变量里设置：

```sh
export CI_LOG_RETENTION_DAYS=7
```

如果某次排障需要临时保留所有容器内日志，可以设置：

```sh
export CI_LOG_CLEANUP_ENABLED=false
```

注意：GitHub Actions 网页上的 run 日志保留时间由 GitHub 仓库或组织设置控制，不受 `ci/cleanup_ci_logs.sh` 影响。

### GitHub Actions 输出分层

每个平台 job 将输出分成两层：

- Job Summary 只显示平台、SOC、执行结论和精度通过计数。失败时补充去重、限长并脱敏后的编译错误、运行异常、精度异常和复现命令。
- 完整构建与测试输出保留在 Actions 原始日志的折叠区中。普通编译 warning、include 展开和构建进度不会重复出现在 Summary 或 PR 评论中；需要深入排障时再展开查看。

平台 artifact 保留结构化执行结果、精度报告和关键诊断。成功报告中的逐 Tensor 指标仍可从精度 JSON 读取，但不会填满 PR 评论。精度报告只有在全部选中 case 已结束，或剩余 case 已明确记为 `not_run` 后才标记完整；进程中断留下的部分报告不会被计为精度通过。CI 不使用 `-w` 或其他方式关闭编译告警，避免丢失有价值的原始诊断。

## 参考链接

- GitHub self-hosted runners 文档：<https://docs.github.com/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners>
- GitHub branch protection 文档：<https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches>
- GitHub `GITHUB_TOKEN` 权限文档：<https://docs.github.com/actions/security-for-github-actions/security-guides/automatic-token-authentication>
