"""Layout only: installation and resolution use the shared wheel layer manager."""
from . import _wheel_overlay as manager

manager.configure_layout(
    package="fla_npu_experimental",
    base_vendor="fla_npu_experimental_transformer",
    overlay_vendor="fla_npu_experimental_overlay_transformer",
    launcher="libfla_npu_experimental_stable.so",
    stamp="_stable_hash.py",
    overlay_prefix="fla_npu_experimental_overlay_",
)
