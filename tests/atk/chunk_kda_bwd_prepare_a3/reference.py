import importlib.util
from pathlib import Path
_path = Path(__file__).resolve().parents[1] / "chunk_kda_bwd_prepare" / "reference.py"
_spec = importlib.util.spec_from_file_location("_canonical_kda_bwd_prepare_reference", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_cpu_golden = _module._cpu_golden
