"""Video delta-network linear attention on Ascend NPUs."""

from .forward import vdn_forward

from .frontend import vdn_forward_from_features

__all__ = ["vdn_forward", "vdn_forward_from_features"]
