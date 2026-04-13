import pathlib
import torch


# Load the custom operator library
def _load_opextension_so():
    so_dir = pathlib.Path(__file__).parents[0]
    so_files = list(so_dir.glob('custom_aclnn_extension_lib*.so'))

    if not so_files:
        raise FileNotFoundError(f"not find custom_aclnn_extension_lib*.so in {so_dir}")

    atb_so_path = str(so_files[0])
    torch.ops.load_library(atb_so_path)

_load_opextension_so()
