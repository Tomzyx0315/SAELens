"""Model adapters. Each adapter exposes the same interface so that collect/geometry/analyze are model-agnostic.

Adapter interface:

- `tok`: tokenizer; `encode(text) -> (1, T) LongTensor` on the model device (handles BOS per model default)
- `structure(loops) -> dict` with keys n_prelude, n_core, n_coda, loops, sink_pos, train_loops
- `run(ids, loops) -> Trace`

See `models/base.py` for the Trace definition.
"""

import torch

from models.base import Trace
from models.huginn import HuginnAdapter
from models.llama import LlamaAdapter
from models.ouro import OuroAdapter

ADAPTERS = {"ouro": OuroAdapter, "huginn": HuginnAdapter, "llama": LlamaAdapter}


def get_adapter(name: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
    return ADAPTERS[name](dtype, device)


__all__ = ["Trace", "get_adapter", "ADAPTERS"]
