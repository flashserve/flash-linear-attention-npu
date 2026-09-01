import os
import pathlib
import warnings
import torch


def _warn_if_embedded_opp_not_preconfigured() -> None:
    """Warn when the wheel-embedded custom OPP was not pre-configured (issue #429).

    The v26.1.0 wheel does not embed an OPP (the run package installs it into
    the CANN vendor directory), so this is a no-op unless a future wheel embeds
    one. CANN discovers custom operator host/tiling/kernel binaries when the
    runtime is initialized; if it initializes before ``import fla_npu``, setting
    ``ASCEND_CUSTOM_OPP_PATH`` in this process may be too late (issue #429).
    """
    vendor_dir = pathlib.Path(__file__).resolve().parent / "opp" / "vendors" / "fla_npu_transformer"
    if not vendor_dir.is_dir():
        return
    parts = [part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if part]
    if str(vendor_dir) in parts:
        return
    warnings.warn(
        "[fla-npu] ASCEND_CUSTOM_OPP_PATH does not contain the custom OPP implements "
        "before import fla_npu, If you need to use fla_npu operators,  run:\n\n"
        f"  export ASCEND_CUSTOM_OPP_PATH=\"{vendor_dir}:{vendor_dir / 'op_api' / 'lib'}:${{ASCEND_CUSTOM_OPP_PATH:-}}\"",
        RuntimeWarning,
        stacklevel=2,
    )


# Load the custom operator library
def _load_opextension_so():
    _warn_if_embedded_opp_not_preconfigured()
    so_dir = pathlib.Path(__file__).parents[0]
    so_files = list(so_dir.glob('custom_aclnn_extension_lib*.so'))

    if not so_files:
        raise FileNotFoundError(f"not find custom_aclnn_extension_lib*.so in {so_dir}")

    atb_so_path = str(so_files[0])
    torch.ops.load_library(atb_so_path)

_load_opextension_so()
