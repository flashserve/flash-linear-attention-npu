"""pre_process_fwd_kernel_merged of ATK executor.

本算子是 CP（context parallel）场景下 GDN / KDA / DPLR 前向的 pre-process 融合算子：
一次调用处理**一个打包窗口**（窗口内可含多段序列，段边界由 `cu_seqlens` 给出），
输出 `hm[Nseq, HV, K, V+K]`（左 `[0,V)` 为 `h`，右 `[V,V+K)` 为 `m`）。

输入布局：**BNSD `[B,H,T,D]`**（与仓内其它 AscendC 算子一致），`B ≡ 1`。
CPU 标杆：本目录 `scripts/pre_process_fwd_kernel_merged_cpu.py`
（纯 PyTorch，token-major `[T,H,D]`），本文件只做布局搬运与逐段调用。

本 executor 支持三种 ATK 节点角色：

- **NPU DUT**（`device=npu`）：仓内 `fla_npu.ops.ascendc.pre_process_fwd_kernel_merged`；
- **高精度 golden**（`is_benchmark_task=True`，CPU 或远端 GPU）：FP64 小算子拼接；
- **同精度 benchmark**（`is_benchmark_task=False`，CPU 或远端 GPU）：契约版 FP32 标杆；
  GPU 节点上优先用上游 Triton kernel，不可用时回落 GPU torch 契约精度标杆。

**GPU 双标杆**（A5/A910 侧发起，远端 GPU server 承载参考节点）：同一 seed 下参考节点被
调用两次 —— FP64 真值（golden）与同精度标杆（benchmark），三路比较由
`cv_fused_double_benchmark` 完成。GPU 标杆用上游
`fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged` **直接按窗口启动**
（需要 `FLACPContext` / 进程组的编排层 `chunk_gated_delta_rule_fwd_h_pre_process`
本身不参与比较）。CPU 双标杆同样可用，但真值与同精度标杆都在 CPU 上跑，更慢。

精度口径（重要）：本算子的**验收基线是"契约版"标杆** —— `accum_dtype=fp32` +
三个舍入点开关全开。`scripts/pre_process_fwd_kernel_merged_cpu.py` 的模块文档写明：kernel 的 h/m 累加器是 FP32，
用 FP64 基准会让任何忠实实现平白多出 ~9.4e-3 的绝对偏差（与 H20 `ieee` 对齐时实测）。
因此 **FP64 结果（`high_precision=True`）只作参考侧灵敏度对照（ATK golden 节点）**，
与 DUT 同精度类的对照是约定容差下的混合容差比较；本工程的输入构造（模型同构分布）
与逐段调用约定不随之改变。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import (
    _calc_dtype,
    _case_spec,
    _finite_tuple,
    _marker_device,
    _orig_dtype,
)


OP_NAME = "pre_process_fwd_kernel_merged"
ACLNN_NAME = "PreProcessFwdKernelMerged"

K_DIM = 128
V_DIM = 128
BT = 64

# 上游 Triton kernel（token-major `[B,T,H,*]`）：`<python_module>:<callable>`。
# 该 kernel 是 `@triton.jit`，调用方式是 `kernel[grid](...)`，因此走 `_triton_hm` 直接启动。
_DEFAULT_TRITON_CALLABLE = "fla.ops.cp.chunk_delta_h:pre_process_fwd_kernel_merged"
_TRITON_ENV = "PPFM_ATK_TRITON"
_TRITON_CALLABLE_ENV = "PPFM_ATK_TRITON_CALLABLE"

# CPU 标杆放在本算子 ATK 目录的 scripts/ 下（与仓内其它算子的约定一致）。
_REFERENCE_PY = (
    Path(__file__).resolve().parent / "scripts" / "pre_process_fwd_kernel_merged_cpu.py"
)


def _cuda_device() -> torch.device:
    """GPU 参考节点使用的设备；远端 ATK GPU server 按 `CUDA_VISIBLE_DEVICES` 暴露。"""
    return torch.device("cuda")


def _load_triton_callable():
    """加载上游 GPU Triton kernel；不可用（未安装 / 导入失败 / 被禁用）时返回 None，
    由调用方回落到 GPU torch 契约精度标杆。"""
    if os.environ.get(_TRITON_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    target = os.environ.get(_TRITON_CALLABLE_ENV, _DEFAULT_TRITON_CALLABLE).strip()
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"{_TRITON_CALLABLE_ENV} 必须使用 '<python_module>:<callable>' 语法")
    try:
        module = importlib.import_module(module_name)
        callable_obj = getattr(module, attribute, None)
    except (ImportError, AttributeError):
        return None
    return callable_obj if callable(callable_obj) else None


def _load_reference():
    """加载本目录 scripts/ 下的 CPU 标杆。"""
    if not _REFERENCE_PY.is_file():
        raise FileNotFoundError(f"找不到 CPU 标杆：{_REFERENCE_PY}")
    spec = importlib.util.spec_from_file_location("_ppfm_reference", _REFERENCE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ppfm_reference"] = module
    spec.loader.exec_module(module)
    return module.pre_process_fwd_kernel_merged


def build_inputs(spec: dict[str, Any], device: torch.device, high_precision: bool = False) -> dict[str, Any]:
    """按 **token-major** 构造本算子的输入（与 `scripts/pre_process_fwd_kernel_merged_cpu.py` 同分布）。

    spec 字段：dtype / B(=1) / HK / HV / T / K(=128) / V(=128) / chunk_size(=64)
              / cu_seqlens(可选, list[int]) / gate(g|gk) / gate_dtype(fp32|bf16) / route / soc

    **数据分布必须是模型同构的**：
    `k` 归一化、`w = beta · k`（`beta ~ U(0, 0.02)`），使 `|Kw| << 1`、`m` 链良态。
    若改用满幅随机 `w`，`m = Π M_c` 会把 fp32 求和顺序的 1 ulp 差异放大到 O(1)
    （`|m| ~ 1e7`），那是**用例病态**、不是实现缺陷 —— 交付件里不允许出现这种用例。

    head 约定：`k` **恒为 `[T, HK, K]`**（即使 gk/GVA 路径，门控按 value head 给 `gk[T,HV,K]`）；
    `w/u/v` 在 `HV` 维。`HK` 与 `HV` 成倍数（`HV % HK == 0`）。
    """
    dtype_name = str(spec.get("dtype", "bf16")).lower()
    calc = _calc_dtype(dtype_name, high_precision)
    seed = int(spec.get("seed", 20260818))
    elem = _orig_dtype(dtype_name)

    HK = int(spec.get("HK", 1))
    HV = int(spec.get("HV", 1))
    T = int(spec.get("T", 1024))
    K = int(spec.get("K", K_DIM))
    V = int(spec.get("V", V_DIM))
    chunk_size = int(spec.get("chunk_size", BT))
    gate_kind = str(spec.get("gate", "g")).lower()
    gate_dtype = str(spec.get("gate_dtype", "fp32")).lower()

    cu_raw = spec.get("cu_seqlens")
    cu = [int(x) for x in cu_raw] if cu_raw else [0, T]

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def randn(shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def real_gate(shape_tail):
        """真实分布：每个 chunk 内做 cumsum 的负对数衰减（与标杆 self_test 同款）。"""
        nblk = -(-T // chunk_size)
        base = -0.013 / chunk_size * (1 + torch.rand(nblk, chunk_size, *shape_tail, generator=gen) * 0.5)
        return base.cumsum(1).reshape(nblk * chunk_size, *shape_tail)[:T].contiguous()

    # 模型同构数据：k 归一化、w = beta * k（beta ~ U(0, 0.02)），保证 |Kw| 远小于 1、m 链良态。
    k = torch.nn.functional.normalize(randn((T, HK, K)), dim=-1).to(elem)
    v = randn((T, HV, V)).to(elem)
    beta = torch.rand((T, HV, 1), generator=gen, dtype=torch.float32) * 0.02
    head_of_k = torch.arange(HV) // (HV // HK)
    w = (beta * k[:, head_of_k].float()).to(elem)
    u = v.clone()

    inputs: dict[str, Any] = {
        "k": k.to(calc),
        "v": v.to(calc),
        "w": w.to(calc),
        "u": u.to(calc),
        "cu_seqlens": [int(x) for x in cu],
        "chunk_size": chunk_size,
        "gate_kind": gate_kind,
    }
    if gate_kind == "gk":
        # KDA：逐 K 门控，按 value head 给（HV 个），k 仍按 HK 头 ⇒ HK < HV（GVA）合法。
        inputs["gk"] = real_gate((HV, K)).to(elem if gate_dtype != "fp32" else torch.float32).to(calc)
        inputs["g"] = None
    else:
        inputs["g"] = real_gate((HV,)).to(elem if gate_dtype != "fp32" else torch.float32).to(calc)
        inputs["gk"] = None
    return inputs


def _to_bnsd(x: torch.Tensor) -> torch.Tensor:
    """token-major `[T,H,(D)]` -> DUT 要求的 BNSD `[1,H,T,(D)]`。"""
    return x.movedim(1, 0).unsqueeze(0).contiguous()


def _reference_hm(inputs: dict[str, Any], *, high_precision: bool) -> torch.Tensor:
    """在 `inputs` 所在设备上按 `cu_seqlens` **逐段**调用标杆，再 stack 成 `[Nseq, HV, K, V+K]`。

    与算子契约一一对应：算子的 `hm[i]` == 标杆对第 i 段单独调用一次的结果。
    设备无关：同一份实现同时服务 CPU/GPU 的 golden 与 benchmark 角色。
    """
    ref = _load_reference()
    cu = [int(x) for x in inputs["cu_seqlens"]]
    chunk_size = int(inputs["chunk_size"])

    # 契约版（同精度对照）与高精度真值：只有 accum_dtype 与三个舍入开关不同。
    if high_precision:
        accum_dtype = torch.float64
        round_h = round_vnew = round_affine = False
    else:
        accum_dtype = torch.float32
        round_h = round_vnew = round_affine = True

    outs = []
    for i in range(len(cu) - 1):
        outs.append(
            ref(
                inputs["k"], inputs["v"], inputs["w"],
                g=inputs["g"], gk=inputs["gk"], bg=None, u=inputs["u"],
                chunk_size=chunk_size,
                cu_seqlens=[cu[i], cu[i + 1]],
                accum_dtype=accum_dtype,
                round_h_to_input_dtype=round_h,
                round_v_new_to_input_dtype=round_vnew,
                round_affine_chain_to_float32=round_affine,
            )
        )
    return torch.stack(outs, dim=0)


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """CPU 标杆：`high_precision=True` 为 FP64 真值，`False` 为契约版同精度标杆。"""
    inputs = build_inputs(spec, torch.device("cpu"), high_precision)
    return _reference_hm(inputs, high_precision=high_precision)


def run_gpu_truth(spec: dict[str, Any]):
    """GPU 双标杆的**真值**：FP64 小算子拼接，真跑在 CUDA 上。"""
    inputs = build_inputs(spec, _cuda_device(), high_precision=True)
    return _reference_hm(inputs, high_precision=True)


def _triton_hm(callable_obj, inputs: dict[str, Any]) -> torch.Tensor:
    """逐段启动上游 Triton kernel，返回 `[Nseq, HV, K, V+K]`。

    上游 kernel 是 token-major `[B,T,H,*]`、以 `MULTI_SEQS=False` 处理**一个窗口**，
    与本算子的 DUT 契约（`hm[i]` = 第 i 段单独调用）逐段对应。
    `AFFINE_CHAIN_PRECISION="ieee"` 对齐契约里"`M_c @ m` 每 chunk 回落 FP32"的口径。
    """
    import triton  # 与上游 kernel 同环境；缺失时 _load_triton_callable 已经返回 None

    k, v, w, u = inputs["k"], inputs["v"], inputs["w"], inputs["u"]
    g, gk = inputs["g"], inputs["gk"]
    T, HK, K = k.shape
    HV, V = u.shape[1], u.shape[2]
    chunk_size = int(inputs["chunk_size"])
    dev = k.device
    cu = [int(x) for x in inputs["cu_seqlens"]]

    block = 32 if K <= 64 else 64
    grid = (triton.cdiv(V, block) + triton.cdiv(K, block), HV)

    outs = []
    for i in range(len(cu) - 1):
        hm = torch.zeros((HV, K, V + K), dtype=torch.float32, device=dev)
        callable_obj[grid](
            k=k.unsqueeze(0).contiguous(),
            v=v.unsqueeze(0).contiguous(),
            w=w.unsqueeze(0).contiguous(),
            g=(None if g is None else g.unsqueeze(0).contiguous()),
            gk=(None if gk is None else gk.unsqueeze(0).contiguous()),
            bg=None,
            u=u.unsqueeze(0).contiguous(),
            hm=hm,
            cu_seqlens=torch.tensor([cu[i], cu[i + 1]], dtype=torch.int32, device=dev),
            T=T,
            H=HK,
            HV=HV,
            K=K,
            V=V,
            BT=chunk_size,
            BLOCK_SIZE=block,
            BK1=triton.next_power_of_2(K),
            MULTI_SEQS=False,
            AFFINE_CHAIN_PRECISION="ieee",
        )
        outs.append(hm)
    return torch.stack(outs, dim=0)


def run_gpu_control(spec: dict[str, Any]):
    """GPU 双标杆的**同精度标杆**：优先上游 Triton，失败回落 GPU torch 契约精度标杆。"""
    inputs = build_inputs(spec, _cuda_device(), high_precision=False)
    triton_callable = _load_triton_callable()
    if triton_callable is not None:
        try:
            return _triton_hm(triton_callable, inputs)
        except Exception as exc:  # noqa: BLE001
            import warnings

            warnings.warn(f"上游 GPU Triton 标杆不可用，回落 GPU torch 契约精度标杆：{exc!r}")
    return _reference_hm(inputs, high_precision=False)


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """NPU DUT：BNSD 输入，调用仓内 `fla_npu.ops.ascendc.pre_process_fwd_kernel_merged`。"""
    dev = _marker_device(input_data)
    inputs = build_inputs(spec, dev, high_precision=False)
    from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged as npu_ppfm

    return npu_ppfm(
        _to_bnsd(inputs["k"]), _to_bnsd(inputs["w"]), _to_bnsd(inputs["u"]),
        v=None,
        g=(None if inputs["g"] is None else _to_bnsd(inputs["g"])),
        gk=(None if inputs["gk"] is None else _to_bnsd(inputs["gk"])),
        bg=None,
        cu_seqlens=list(inputs["cu_seqlens"]),
        chunk_size=int(inputs["chunk_size"]),
    )


@register("executor_pre_process_fwd_kernel_merged")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.is_benchmark_task = bool(task_result.is_benchmark_task)
        # CPU / 远端 GPU 参考节点都会以两种角色各被调用一次：golden（真值）与 benchmark（同精度）。
        self.high_precision = self.device in {"cpu", "gpu"} and self.is_benchmark_task

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, OP_NAME)
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec, self.high_precision)
        elif self.device == "gpu":
            outputs = run_gpu_truth(spec) if self.high_precision else run_gpu_control(spec)
        else:
            raise RuntimeError(
                f"{OP_NAME} 需要 NPU DUT 和 CPU/GPU 参考节点，"
                f"device={self.device!r}, "
                f"benchmark={self.is_benchmark_task}"
            )
        return _finite_tuple(outputs, golden=(self.device != "npu"))
