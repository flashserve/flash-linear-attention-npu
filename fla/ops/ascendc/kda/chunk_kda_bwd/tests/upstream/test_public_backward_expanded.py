"""Expanded parameters; upstream test function and assertions are unchanged."""
import importlib.util
from pathlib import Path
import pytest
import torch
from public_backward_plugin import public_chunk
spec=importlib.util.spec_from_file_location("upstream_kda",Path("tests/ops/test_kda.py"))
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
@pytest.mark.parametrize("D",[60,64,100,128,256])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("save",[True,False])
def test_expanded_upstream_contract(D,dtype,save,monkeypatch):
    monkeypatch.setattr(upstream,"chunk_kda",public_chunk)
    upstream.test_chunk_varlen_fused_training_contract(2,D,[0,1,14,78,129],save,dtype)
