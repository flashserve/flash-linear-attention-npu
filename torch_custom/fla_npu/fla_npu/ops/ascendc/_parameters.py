"""Pure argument conversions and public CANN storage-format identifiers.

This module has no runtime, descriptor, dynamic-library, or tensor dependencies.
"""

ACL_FORMAT_NCHW = 0
ACL_FORMAT_ND = 2
ACL_FORMAT_NCDHW = 30
ACL_FORMAT_NCL = 47


def optional_bool(value, default: bool) -> bool:
    return default if value is None else bool(value)


def optional_float(value, default: float) -> float:
    return default if value is None else float(value)
