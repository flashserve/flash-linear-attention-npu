"""Compatibility entry that reuses the canonical ChunkKdaFwd test backend."""

from tests.operators._shared.chunk_kda_backend import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("tests.operators._shared.chunk_kda_backend", run_name="__main__")
