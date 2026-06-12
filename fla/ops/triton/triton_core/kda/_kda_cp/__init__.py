# _kda_cp: Context Parallel stub (not supported in NPU version)

class FLACPContext:
    """Stub: Context Parallel is not supported in NPU version."""
    pass


def build_cp_context(*args, **kwargs):
    raise NotImplementedError("Context Parallel is not supported in NPU version")
