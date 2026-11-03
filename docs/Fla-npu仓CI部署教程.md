# NPU CI 部署教程

本文面向第一次接触 GitHub Actions self-hosted runner 的维护者，目标是部署本仓库按需触发的 A2 + A5 双平台 NPU CI。一次触发会并行创建 A2、A5 两个设备 job；每个平台内部按顺序执行分项检查，任一平台缺席或失败都不能通过汇总门禁。

本文用 `CI_ROOT` 统一表示宿主机 CI 部署根目录。以 runner 服务账号登录每台主机后，先设置：

```sh
export CI_ROOT="${CI_ROOT:-$HOME/ci}"
```

选择只含英文字母、数字、`/`、`.`、`_`、`-` 的绝对路径，供后面的 shell 和 systemd 示例使用。示例仓库地址使用 `https://github.com/flashserve/flash-linear-attention-npu`。

`CI_ROOT` 是教程和部署命令使用的变量。设置它不会自动改变 `ci/run_ci_container.sh` 的默认值或 workflow：实际部署时，须让各 job 的绝对路径配置、临时目录清理校验与同一部署位置一致。一键部署工具也须显式传入 `--ci-root "$CI_ROOT"`。

## 部署约定

完整 CI 流程、6 项门禁、文件分类、触发名单和 Admin 确认规则统一见 [仓库规则：NPU CI 门禁](repository-rules.md#npu-ci-门禁)。仅修改 `.md`、`docs/` 或 `LICENSES/` 自动放行；其他正式请求按差异选择包准备方式，并验证选中算子的 ATK 精度。本教程说明 runner、镜像、底包、网络和排障配置。

- A2/A5 使用同一条 Git 源码准备路径：固定 base、拉取并核对 PR head、合并 PR；首次直连，失败后才启用宿主机代理重试。两台主机均需能直连或通过代理访问 GitHub Git 服务。
- 两个平台一律运行 PR 合并后的 `ci/`，不叠加其他版本的 CI 目录。非 Markdown 的 `ci/` 或 `.github/` 修改必须由 CI 名单内的仓库 Admin 带 `ci_modified=true` 确认，重跑人也须满足相同权限。
- runner 的 GitHub 控制连接可以使用代理。源码首次直连步骤会清空继承的 HTTP/HTTPS/ALL/Git 代理并设置 `NO_PROXY=*`；只有源码失败后才加载 `CI_PROXY_SCRIPT`。代理配置须在两台主机均提前部署。
- runner 运行用户须能使用 Docker 和 Ascend 设备。`ci/run_ci_container.sh` 默认带 `--privileged`，否则可能出现 `npu-smi` 不可用或 `torch_npu.npu.device_count()` 为 `0`。
- `/run-npu-ci ... device-id=N` 对应 `CI_DEVICE_ID=N`，约束宿主机物理卡并对该卡加锁；省略时自动择卡。容器通过 `ASCEND_RT_VISIBLE_DEVICES` 只暴露选中卡，Example/ST、PyTorch 和 ATK 使用逻辑设备 `0`。
- A2/A5 使用各自的镜像和 CANN/PyTorch 组合，不共用 wheel。A5 必须预置镜像，不能在 PR job 中临时构建。
- 两个平台均只保留 `triton-ascend==3.2.1` 提供的 `triton` 模块，避免与社区版 `triton` 共存；当前 Stable-only 链路不要求 torchnpugen 或重新编译 op-plugin。
- Example/ST 执行 `ci/example_st_cases.json` 中全部 `enabled=true` 用例；新增 shape 或行为必须在该文件显式配置。
- 同一 PR 的请求以 `cancel-in-progress: false` 排队，准备阶段再按 head、目标分支名、base/workflow SHA、模式、编译计划、ATK 集合和设备号去重；所有任务和手工验证必须共用 `${CI_ROOT}/tmp/npu-locks`，防止抢占同一张卡。
- 容器日志默认只保留 7 天；GitHub Actions 网页日志的保留时间由仓库/组织设置独立控制。

runner 标签必须精确匹配：

| 平台 | 必需标签 |
| --- | --- |
| A2 | `linux,arm64,npu,flash-linear-attention-npu` |
| A5 | `linux,x64,npu,ascend950,flash-linear-attention-npu-a5` |

A5 的独立产品标签用于确保一次门禁覆盖两个平台。注册 token 和分支保护配置仍要求仓库管理员操作；日常 CI 触发权限由默认分支 [`.github/npu-ci-users.json`](../.github/npu-ci-users.json) 独立管理。

## 你需要准备什么

- A2、A5 各一台可运行 self-hosted runner 的 Huawei Ascend NPU 服务器，例如 `<a2-ci-host>` 和 `<a5-ci-host>`
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

登录后，在该主机的 shell 会话中设置开头的 `CI_ROOT`；本地 SSH 之前设置的变量不会自动传入远端。

创建 CI 工作目录：

```sh
mkdir -p "${CI_ROOT}"
cd "${CI_ROOT}"
```

源码与 runner 安装在普通用户自己的目录中；编译临时文件和缓存使用下文配置的数据目录，无需创建宿主机 `/workspace`。

## 第 2 步：放置本仓库代码

有 GitHub 访问权限时，直接 clone：

```sh
cd "${CI_ROOT}"
git clone https://github.com/flashserve/flash-linear-attention-npu.git context
cd context
git checkout main
```

如果服务器不能访问 GitHub，也可以从一台能访问 GitHub 的机器把仓库目录同步到：

```text
${CI_ROOT}/context
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
cd "${CI_ROOT}/context"
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

CI 会优先尝试健康且空闲的 NPU，然后尝试空闲但健康状态异常的 NPU。真正占用前还会拿 `CI_NPU_LOCK_DIR` 下的 `fla-npu-ci-npu-<id>.lock` 文件锁，拿不到锁时会继续尝试下一张卡。手工运行与 Actions 必须使用同一锁目录。

## 第 4 步：准备 Docker 镜像

两个平台的镜像不能互换：

| 平台 | 镜像 | Dockerfile | 关键版本 |
| --- | --- | --- | --- |
| A2 | `fla-npu-ci:9.1.0-910b` | `ci/Dockerfile` | CANN 9.1.0、Python 3.12、PyTorch 2.7.1 |
| A5 | `fla-npu-ci:9.1.0-950` | `ci/Dockerfile.ascend950` | CANN 9.1.0、Python 3.12、PyTorch 2.12.0 |

推荐在 runner 维护窗口从受信任的默认分支构建或加载镜像，再启动 runner。构建命令如下：

```sh
cd "${CI_ROOT}/context"

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

### Daily 底包与 ATK 环境

`ci/prepare_ci_package.py` 实现四种准备方式：`full` 通过 `python scripts/build_wheel.py` 构建完整 wheel；`incremental` 安装 Daily 底包后只构建选中算子并安全叠加 OPP；`torch` 安装底包后只重建 torch_custom；`base` 仅安装底包。后续布局、适配、GDR 和 ATK 均使用该次安装，不重复编译。

底包沿用 [PR #1037](https://github.com/flashserve/flash-linear-attention-npu/pull/1037) 的发布源，默认 `CI_BASE_WHEEL_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/`。按 PR 目标分支选择 `main` 的 `0.dev...+main.<SHA>` 或 `vX.Y.Z` 的 `X.Y.Z.dev...`，按 SOC 选择 A2/A5 包，再按当前 Python、CPU wheel 标签匹配。安装前还校验 METADATA、Stable 库和目标 SOC 的 kernel/config。不会回退到 PyPI 或其他分支、平台；如维护分支无映射、缺包或 ABI 不兼容，应补发匹配 Daily，或使用 `/run-npu-ci full`。维护者可用 `CI_BASE_WHEEL_VERSION` 固定同一分支的某个 Daily 版本。

选中 ATK 算子的 runner 镜像还须预装可用的 ATK（`REQUIRED_ATK_VERSION` 默认最低版本为 26.8.8）、`taskset` 和对应依赖。仓库不假设公开 PyPI 上的同名包可用；维护者应获取受控 HTTPS 来源的 CPython 3.12 wheel，A2 使用 aarch64 wheel，A5 使用 x86_64 wheel，通过 Dockerfile 的 `ATK_WHL_URL`、`ATK_WHL_SHA256` 参数校验后安装，详见 [ATK 镜像依赖](../tests/atk/README.md#ci-镜像依赖)，例如：

```sh
# A2：替换为受控下载地址及对应 wheel 的完整 SHA256。
docker build --platform=linux/arm64 \
  --build-arg ATK_WHL_URL='<aarch64-ATK-wheel-URL>' \
  --build-arg ATK_WHL_SHA256='<64位SHA256>' \
  -t fla-npu-ci:9.1.0-910b -f ci/Dockerfile .

# A5：需在维护窗口构建并预加载到实际 runner 使用的 Docker daemon。
docker build --platform=linux/amd64 --network=host \
  --build-arg ATK_WHL_URL='<x86_64-ATK-wheel-URL>' \
  --build-arg ATK_WHL_SHA256='<64位SHA256>' \
  -t fla-npu-ci:9.1.0-950 -f ci/Dockerfile.ascend950 .
```

两个参数默认留空可保留原镜像构建行为，但没有 ATK 的旧镜像不能完成选中算子的 ATK 验证；入口会明确失败，提示管理员升级镜像。新增测试流程不会自动部署受控 ATK 依赖，A5 也不会在 PR job 内重建镜像。

CI 仅调用 `tests/atk/run_test_cpu.sh`，不直接读取 ATK 用例格式或调用 ATK 内部命令。该入口先检查每个选中算子有至少 200 个独立精度 case、perf/mss 各至少 1 个有效 case，再执行全部精度用例；perf/mss 在此只做资产检查。入口使用 `taskset` 将进程树限制到最多 16 个 CPU，并设置 ATK 最多 16 个任务、关闭自动绑核、数值库每任务 1 线程。镜像缺少 `taskset` 或无权设置 affinity 时应修复镜像/容器配置，不允许绕过限制。

A2/A5 job 统一设置 `CI_TMPDIR=/tmp/fla-npu-ci`。A5 Triton/BiShengIR 编译器要求临时目录位于容器 `/tmp` 下；不要修改这个容器路径。`CI_HOST_TMPDIR` 把 `${CI_ROOT}/tmp/container-<platform>-<run>-<attempt>` bind mount 到该路径，由 `ci/run_ci_container.sh` 创建本次目录。宿主机 `TMPDIR` 固定为 `${CI_ROOT}/tmp`，基础目录由部署脚本提前创建，供源码准备阶段使用。容器中的 `/workspace/repo` 和 `/workspace/gdr_accuracy_golden` 是挂载目标，不要求宿主机存在 `/workspace`。

`ci/run_ci_container.sh` 在宿主机注册 `EXIT` 收尾：通过 Docker `--cidfile` 记录本次创建的容器 ID，仅停止该容器，再用无网络 helper 容器将源码及本次 `CI_HOST_TMPDIR` 的所有权归还调用用户，最后释放设备锁。正常退出、失败、中断及 stale-commit 监控提前 `docker rm -f` 的路径均执行收尾；原始非零退出码保留，成功运行后的收尾失败也会使 CI 失败。目录准备与权限收尾都在入口脚本中完成，无需新增 workflow 步骤或宿主机 `sudo`。

workflow 清理步骤仅删除本次 `CI_HOST_TMPDIR`，并校验目录边界及符号链接。源码直接使用 PR 的 `ci/`，不再创建 CI 覆盖或用例暂存目录。静态 `TMPDIR` 根目录始终保留。

缓存固定使用 `CI_CACHE_ROOT=${CI_ROOT}/cache_root`；NPU 锁固定使用 `${CI_ROOT}/tmp/npu-locks`，跨运行共享，不能按 run 分开。单次运行的收尾保留共享缓存与锁文件，不递归修改它们的所有权。

Stage 耗时统一保存在每台服务器的 `${CI_ROOT}/results/ci-timings/stage-timings.csv`（workflow 通过 `CI_TIMING_ROOT` 配置）。总表每行对应一个 `(run_id, run_attempt, platform, stage)`，包含 commit、SOC、状态、退出码、UTC 起止时间和 `duration_seconds`，历史 CI 行持续保留。记录设备执行阶段，包括 `atk-accuracy`；GitHub-hosted 汇总校验不在此表内。每次 stage 状态变化即加锁更新总表、同步落盘并原子替换，失败也保留，工作区与临时目录清理不会删除它。被强制终止的 stage 保留 `running` 和开始时间，耗时留空；`skipped/not_run` 耗时也留空，不写成 0。启用前须在 A2、A5 分别创建下述可写目录；总表由各自服务器维护，不自动跨机合并。

A2/A5 的代理回退读取普通用户配置 `CI_PROXY_SCRIPT=${CI_ROOT}/vpn-proxy.sh`，调用 `proxyon`，校验变量不含 CR/LF，再通过 `GITHUB_ENV` 将代理地址与端口传递给后续步骤。管理员须提前在两台主机部署 `nelvko/clashctl` 到 `${CI_ROOT}/clashctl`，并将提供的 `vpn-proxy.sh` 适配副本放到上述位置：运行时配置和二进制路径指向该目录，日志写入 `${CI_ROOT}/tmp/mihomo.log`，权限允许 `ubuntu` 读取配置、执行二进制和写日志。脚本属于主机部署配置，不提交到仓库；workflow 不读取 `/root`、不修改 `HOME` 或全局 Git 配置。

A2/A5 workflow 固定设置 `CI_JOBS=16` 和 `CI_CPACK_JOBS=16`，全量与局部 OPP 构建均使用 `-j16`。`ci/run_checks.sh` 的默认并行度也为 16，并将该值导出为 PyTorch/Ninja 使用的 `MAX_JOBS`；CPack、CMake 和 Make 使用同一并行度设置，cmake/scripts/prepare.sh 也遵循 CMAKE_BUILD_PARALLEL_LEVEL，避免内部 prepare 阶段重新按 CPU 数放大并行度。`FLA_NPU_BUILD_ARGS` 追加 `-j16`，覆盖 wheel 打包内部再次调用 `build.sh` 时的并行度。

runner 使用普通 `ubuntu` 用户，workflow 全程不调用 `sudo`。[一键部署工具](https://github.com/alvazu/fla-npu-ci-deploy)负责创建基础数据目录及配置用户服务；在工具仓库中执行 `bash deploy.sh --platform a5 --ci-root "$CI_ROOT" --setup-host` 可显式启用首次主机准备，A2 将平台改为 `a2`。管理员须提前授予该用户 Docker daemon 访问权限、所列宿主机目录的读写权限及其父目录的遍历权限；手工准备时可执行以下命令，之后重新登录 `ubuntu`，使新组生效，再启动 runner 用户服务：

```sh
sudo groupadd -f fla-npu-ci
sudo usermod -aG docker,HwHiAiUser,fla-npu-ci ubuntu
sudo install -d -o ubuntu -g fla-npu-ci -m 2775 \
  "${CI_ROOT}/tmp" "${CI_ROOT}/cache_root" \
  "${CI_ROOT}/results" "${CI_ROOT}/results/ci-timings" \
  "${CI_ROOT}/actions-runner"
```

容器内部仍使用镜像默认用户执行全局 pip 和 OPP 安装；这与宿主机 runner 以普通用户运行是两个权限范围。手工调用 `ci/run_ci_container.sh` 时，也应显式设置 `TMPDIR`、`CI_TMPDIR`、`CI_HOST_TMPDIR`、`CI_CACHE_ROOT` 和 `CI_NPU_LOCK_DIR`，使用与 Actions 相同的数据目录和设备锁。

## 第 5 步：验证容器能看到 NPU

先跑一个轻量检查。这里重点验证 `--privileged`、驱动挂载和容器内逻辑设备号：

```sh
cd "${CI_ROOT}/context"

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
cd "${CI_ROOT}/context"

# 两个平台共用这些目录配置；与 Actions 共享设备锁和缓存。
export TMPDIR="${CI_ROOT}/tmp"
export CI_TMPDIR=/tmp/fla-npu-ci
export CI_HOST_TMPDIR="${CI_ROOT}/tmp/manual-container"
export CI_CACHE_ROOT="${CI_ROOT}/cache_root"
export CI_NPU_LOCK_DIR="${CI_ROOT}/tmp/npu-locks"

# A2
CI_IMAGE=fla-npu-ci:9.1.0-910b \
CI_SOC=ascend910b \
CI_MODE=quick \
CI_BUILD_KIND=full \
CI_RUN_EXAMPLE_ST=true \
bash ci/run_ci_container.sh

# A5：必须提前准备镜像
CI_IMAGE=fla-npu-ci:9.1.0-950 \
CI_SOC=ascend950 \
CI_REQUIRE_PRELOADED_IMAGE=true \
CI_MODE=quick \
CI_BUILD_KIND=full \
CI_RUN_EXAMPLE_ST=true \
bash ci/run_ci_container.sh
```

这个命令会：

- 自动扫描宿主机 NPU
- 用 `--privileged` 启动容器
- 挂载 `third_party` 缓存
- 按顺序检查环境、wheel 配置及运行时契约
- 通过 `python scripts/build_wheel.py` 构建并安装完整目标 SOC wheel
- 检查已安装 wheel、OPP 布局和 PyTorch 适配环境
- 执行 `ci/example_st_cases.json` 中启用的 GDR Example/ST 用例，仅覆盖容器内逻辑设备号 `--device 0`

上述命令显式使用 `CI_BUILD_KIND=full` 做镜像验收；Actions 会从 PR 文件差异注入执行计划。手工验证增量构建和 ATK 时，例如：

```sh
CI_IMAGE=fla-npu-ci:9.1.0-910b \
CI_SOC=ascend910b CI_MODE=quick CI_BASE_REF=main \
CI_BUILD_KIND=incremental CI_BUILD_OPS=chunk_fwd_o \
CI_ATK_OPS=chunk_fwd_o CI_DEVICE_ID=0 \
bash ci/run_ci_container.sh
```

仅验证 torch_custom 时使用 `CI_BUILD_KIND=torch`；底包验证使用 `CI_BUILD_KIND=base`，两者均清空 `CI_BUILD_OPS` 并填写 `CI_BASE_REF`。`CI_ATK_OPS` 为要完整验证的 ATK 算子列表，逗号分隔；为空则跳过 ATK。A5 同时设置对应 SOC、镜像和 `CI_REQUIRE_PRELOADED_IMAGE=true`。

正式 `quick`、`full` 都执行环境、已安装布局、适配环境和 GDR，并运行相同选中集合的 ATK；`full` 强制完整 wheel，且额外运行全部 PyTorch 适配测试。手工 `CI_RUN_TORCH_TESTS=true` 可在 quick 中启用该测试，正式 workflow 不设置这一覆盖。

上述分项串行执行。一个分项失败后，后续依赖分项标记未执行。失败日志和汇总诊断同时给出原因、修改建议及包含编译计划、ATK、分支和设备号的复现命令，例如：

```sh
CI_STAGE=opp-package \
CI_MODE=quick \
CI_BUILD_KIND=incremental CI_BUILD_OPS=chunk_fwd_o CI_BASE_REF=main \
CI_ATK_OPS=chunk_fwd_o CI_DEVICE_ID=0 \
CI_SOC=ascend950 \
FLA_NPU_SOC=ascend950 \
CI_IMAGE=fla-npu-ci:9.1.0-950 \
CI_DOCKERFILE=ci/Dockerfile.ascend950 \
CI_REQUIRE_PRELOADED_IMAGE=true \
bash ci/run_ci_container.sh
```

`CI_STAGE` 可取 `environment-contracts`、`opp-package`、`standalone-layout`、`torch-adapter`、`gdr-example-st` 或 `atk-accuracy`。脚本会先执行该分项所需的前置分项，再执行目标分项；应优先使用失败诊断中给出的完整命令。正式增量验证使用 `CI_BUILD_KIND=incremental` 和 `CI_BUILD_OPS`，并保留 `CI_STAGE=all`；命令 `ops=` 才使用仅 01/02 的诊断路径。

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

在 A2、A5 NPU 服务器上分别以 `ubuntu` 用户执行：

```sh
cd "${CI_ROOT}/context"
source "${CI_ROOT}/vpn-proxy.sh"
proxyon

RUNNER_ROOT="${CI_ROOT}/actions-runner" \
bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <把刚才复制的 registration token 放这里>
```

上面的默认值用于 A2。A5 必须显式使用独立 runner 名称和完整标签：

```sh
cd "${CI_ROOT}/context"
source "${CI_ROOT}/vpn-proxy.sh"
proxyon

RUNNER_ROOT="${CI_ROOT}/actions-runner" \
RUNNER_NAME="$(hostname)-flash-linear-attention-npu-a5" \
RUNNER_LABELS=linux,x64,npu,ascend950,flash-linear-attention-npu-a5 \
bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <把刚才复制的 registration token 放这里>
```

以上示例通过环境变量覆盖安装目录，两台主机分别安装到各自的：

```text
${CI_ROOT}/actions-runner
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

脚本自身的默认安装路径仍为 `/workspace/actions-runner/flash-linear-attention-npu`，部署时使用上面的 `RUNNER_ROOT` 覆盖。注册完成后，以 `ubuntu` 用户创建用户服务。下面使用可展开的 here-doc，在生成 unit 时写入绝对路径；systemd 运行时不依赖 shell 中的 `CI_ROOT`：

```sh
mkdir -p "$HOME/.config/systemd/user"
RUNNER_ROOT="${CI_ROOT}/actions-runner"
cat > "$HOME/.config/systemd/user/fla-npu-runner.service" <<EOF
[Unit]
Description=Flash Linear Attention NPU GitHub Actions runner

[Service]
WorkingDirectory=${RUNNER_ROOT}
ExecStart=/bin/bash -c 'source "${CI_ROOT}/vpn-proxy.sh"; proxyon; exec "${RUNNER_ROOT}/bin/runsvc.sh"'
Restart=always
RestartSec=5
KillMode=process
TimeoutStopSec=5min

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now fla-npu-runner.service
```

管理员另外执行一次 `sudo loginctl enable-linger ubuntu`，使用户服务在退出登录后继续运行并随主机启动。如果用户服务管理器在授予 Docker 等组权限前就已启动，管理员需安排重新启动该用户管理器或主机，使服务继承新的组权限；仅重启 runner 进程不能刷新已有用户管理器的组列表。完成后回到 GitHub 页面：

```text
Settings -> Actions -> Runners
```

确认 runner 状态是绿色 `Idle` 或 `Online`。

## 第 8 步：runner 没变绿怎么查

先在服务器上看 runner 目录：

```sh
cd "${CI_ROOT}/actions-runner"
ls
```

以 `ubuntu` 用户检查服务：

```sh
systemctl --user status fla-npu-runner.service
```

尝试重启：

```sh
systemctl --user restart fla-npu-runner.service
systemctl --user status fla-npu-runner.service
```

查看日志：

```sh
journalctl --user -u fla-npu-runner.service -n 200 --no-pager
```

如果没有安装服务，可以前台运行看日志：

```sh
cd "${CI_ROOT}/actions-runner"
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
cd "${CI_ROOT}/context"

RUNNER_ROOT="${CI_ROOT}/actions-runner" \
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
cd "${CI_ROOT}/context"

export GITHUB_TOKEN=<具有仓库 Administration 写权限的 GitHub token>
bash scripts/github/apply_branch_protection.sh main
```

这个脚本会给 `main` 配置：

- 必需状态检查：[仓库规则](repository-rules.md#npu-ci-门禁)列出的 6 个 NPU context 和 `CI 契约测试`
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
12. 添加[仓库规则](repository-rules.md#npu-ci-门禁)列出的 6 个 NPU context 和 `CI 契约测试`
13. 勾选让 administrators 也遵守分支保护，或不要让普通 Admin 具备全局 bypass
14. 在 bypass 或 pull request bypass allowance 中添加用户 `weinachuan`
15. 保存规则

如果 GitHub 页面里搜不到某个 status check，通常是因为该 check 还没有在仓库里跑过。可以先让 workflow 或状态检查执行一次，或者直接用脚本通过 API 设置。

## 第 10 步：验证触发方式

先确认触发用户名已加入默认分支的 [`.github/npu-ci-users.json`](../.github/npu-ci-users.json)。分发和执行准备各自读取默认分支名单，待测 PR 的名单变更不授予权限；普通代码验证无需仓库 Admin 权限。CI 文件变更还须名单内 Admin 带 `ci_modified=true` 确认，实际重跑人也须满足对应权限。完整规则见 [仓库规则](repository-rules.md#npu-ci-门禁)。

部署后的检查顺序：

1. 新建只修改 `.md`、`docs/` 或 `LICENSES/` 的 PR，确认 6 项自动 success 且没有 NPU job。重命名检查原、新路径；文件列表不完整时不能跳过，同一 commit 在其他 PR 的文档结果不能代替本 PR 检查。
2. 在普通代码 PR 上确认 6 项初始 pending；在包含非 Markdown 的 `ci/` 或 `.github/` 修改的 PR 上确认默认 failure 评论明确提示 Admin 追加 `ci_modified=true`。
3. 用名单用户评论 `/run-npu-ci quick`，CI 修改则由名单内 Admin 评论 `/run-npu-ci quick ci_modified=true`；Actions 手动输入使用下划线名称 `atk_ops`、`device_id`、`ci_modified`。确认新增评论列出全量/增量/torch/底包计划及 ATK 集合，并分别调度 A2/A5。
4. 验证普通单算子修改只编译对应算子并自动测试该算子 ATK；只有 ATK 算子目录修改时用底包运行 ATK；公共构建文件修改应全量。核对 artifact 中实际包准备方式和底包分支、SOC、CPU、Python、版本。
5. 使用 `/run-npu-ci quick atk-ops=chunk_fwd_o device-id=0`，确认手动 ATK 与自动集合合并、物理设备选择生效，容器中的测试仍使用逻辑设备 `0`。缺少 200 个精度 case、perf/mss 各 1 个时，日志须列出期望/实际数量和补齐建议；ATK 日志须显示最多 16 任务/16 CPU。
6. 在两个平台日志确认源码先直连、失败后代理，并一律使用 PR 合并后的 `ci/`。`full` 应强制完整 wheel，且比 `quick` 额外执行全部 PyTorch 适配测试；GDR 和同一 ATK 集合的验证范围保持相同。
7. 完成后确认新增独立结果评论及 6 个最终状态，ATK 显示为单独设备阶段并纳入第 06 项。缺少选中 ATK 报告不能通过；失败评论包含明确原因、修改建议及 `CI_STAGE` 复现命令。
8. 再次触发相同请求，确认先排队、保留原任务，再去重并新增评论。head、目标分支名、base/workflow SHA、模式、编译计划、ATK 集合或设备号不同均应重新验证。

正式门禁使用 `/run-npu-ci quick` 或 `/run-npu-ci full`，按需追加 `atk-ops=...`。`ops=...` 只执行显式算子的 01/02 增量编译诊断，不运行 ATK、不满足正式门禁，不能和 `full`、`atk-ops` 组合。评论默认使用 PR 目标分支的 `ci.yml`，`ci_ref=<branch>` 或 Actions 页面的 `Branch` 可选择实现相同输入契约的其他 CI 定义分支。

如果 Actions 有运行但 PR 没有评论，检查 `Settings` → `Actions` → `General` 的 workflow 写权限，以及组织策略是否允许 job 申请 `issues: write`、`statuses: write`。不同 PR 可同时调度；`ci/run_ci_container.sh` 通过共享物理 NPU 文件锁等待空闲卡：

锁相关环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CI_NPU_LOCK_DIR` | `/tmp`（脚本默认） | workflow 覆盖为 `${CI_ROOT}/tmp/npu-locks` |
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

合入 PR 前确认：

1. 当前 head commit 的 6 个 NPU context 和 `CI 契约测试` 全部 success
2. GitHub 分支保护要求的 2 个 approval 和 Code Owners review 已经满足

如果 PR push 新 commit 或更换目标分支，门禁会重新检查：只有 `.md`、`docs/` 或 `LICENSES/` 改动直接通过，其余需名单用户重新触发；CI 文件变更还须 Admin 确认。旧 commit 或旧 base 的结果不能放行新的请求。

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

### 旧镜像仍检查 torchnpugen

当前 Stable-only 构建和 CI 不再要求 torchnpugen。出现这类检查时，先确认使用的是当前提交的 Dockerfile、CI 脚本和镜像；不要为恢复已删除的生成链路重新拉取 op-plugin 编译。

### 报 `triton` namespace 或重复注册问题

检查是否同时安装了社区版 `triton` 和 `triton-ascend`。本仓 A2 CI 镜像只保留 `triton-ascend==3.2.1`，A5 CI 镜像只保留 `triton-ascend==3.2.1` 提供的 `triton` 模块。

### 安装后 Python 调用找不到自定义 op

检查日志里是否设置了 custom OPP 的 op_api lib：

```text
Custom OPP op_api lib: <已安装fla_npu>/opp/vendors/fla_npu_transformer/op_api/lib
```

如果没有，检查 `.ci-tmp/npu-ci-package.json` 与生成的环境文件，确认完整 wheel 已安装且在 Python 导入前加载其 `bin/set_env.bash`。增量模式还需启用 `fla_npu_overlay_transformer`，不能将部分 `.run` 直接覆盖完整底包；overlay ABI 不匹配时修复接口兼容性，或使用 `full` 构建完整 wheel。

### Daily 底包下载或匹配失败

核对 PR 目标分支是否为 `main` 或已有对应 Daily 的 `vX.Y.Z`，发布源中是否同时存在当前 SOC、CPU 架构、Python 版本对应 wheel。不能用 A2/aarch64 代替 A5/x86_64，也不能用 `main` 的包代替维护分支。应发布匹配底包，必要时设置同分支 `CI_BASE_WHEEL_VERSION`，或改用 `/run-npu-ci full`；不要增加其他 index 回退来掩盖缺包。

### ATK 缺失、资产不足或精度失败

日志会显示算子、失败阶段和修改建议。缺少 `atk` 时，管理员按上面的受控 wheel + SHA256 流程重建镜像，A5 预加载后再触发。缺少 `taskset`、无法设置 CPU affinity 或 ATK 版本过旧时，修复镜像依赖和容器权限。

用例检查失败时，按日志中的 expected/actual 在该算子 ATK 目录补齐至少 200 个独立精度 case、perf/mss 各 1 个有效 case，修复格式或重复 id。执行失败或精度不通过时，从 artifact 查看首个失败 case，修复 executor、CPU 标杆、环境或算子实现，再通过同一入口复现：

```sh
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -scope=accuracy -ci=true -npu_device_id=0 -soc=ascend910b
```

该命令需在已安装待测 wheel、加载正确环境且预装 ATK 的对应平台容器内运行。不得通过缩减 case 范围、关闭 CPU 限制或跳过选中 ATK 来满足门禁。

### 评论触发没有反应

检查：

- 评论是否发在 PR 页面，不是普通 Issue 页面
- 命令是否以 `/run-npu-ci` 开头
- 发送评论的人是否在默认分支 `.github/npu-ci-users.json` 名单中；文件是否为合法 JSON
- PR 是否修改非 Markdown 的 `ci/`、`.github/`；如是，触发人必须同时为仓库 Admin 并显式添加 `ci_modified=true`
- 是否错误组合了 `ops=` 与 `full` 或 `atk-ops`；需要 ATK 时使用正式 `quick atk-ops=...`
- `NPU CI 评论分发` workflow 是否启用
- PR 目标分支或 `ci_ref` 指定分支是否存在 `.github/workflows/ci.yml`
- 目标分支的 `ci.yml` 是否声明 `workflow_dispatch`
- Actions 页面是否先出现 `NPU CI 评论分发`，随后出现新的 `NPU CI` run

### PR 创建后没有 6 个 NPU CI 分项的未执行状态

只有 `.md`、`docs/` 或 `LICENSES/` 的 PR 直接显示 6 个 success。CI 文件变更在管理员确认前显示 failure；其他 PR 检查：

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

同一 PR 的后续请求先排队，保留当前正在执行的任务；进入准备阶段后，按 head、目标分支名、base/workflow SHA、模式、编译计划、ATK 集合和设备号去重，新增评论说明复用或重新运行的结论。

不同 PR 或不同 commit 同时触发时，GitHub Actions 可能把它们分配给不同 self-hosted runner。runner 宿主机会用 `flock` 给物理 NPU 加锁，避免多个任务抢同一张卡。workflow 使用的锁文件如下：

```text
${CI_ROOT}/tmp/npu-locks/fla-npu-ci-npu-0.lock
${CI_ROOT}/tmp/npu-locks/fla-npu-ci-npu-1.lock
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

平台 artifact 保留结构化阶段结果、包准备信息、GDR/ATK 精度报告和关键诊断。成功报告中的逐 Tensor 指标仍可从精度 JSON 读取，不会填满 PR 评论。精度报告只有在全部选中 case 已结束，或剩余 case 已明确记为 `not_run` 后才标记完整；进程中断、缺少选中算子的 ATK 报告或存在未执行 case 均不能计为通过。CI 不使用 `-w` 或其他方式关闭编译告警。

## 参考链接

- GitHub self-hosted runners 文档：<https://docs.github.com/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners>
- GitHub branch protection 文档：<https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches>
- GitHub `GITHUB_TOKEN` 权限文档：<https://docs.github.com/actions/security-for-github-actions/security-guides/automatic-token-authentication>
