"""Feedforward baseline Llama-3.2-1B, exposed through the looped interface as a single loop of 16 layers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import Trace
from paths import WEIGHTS


class LlamaAdapter:
    name = "llama"
    train_loops = 1

    def __init__(self, dtype=torch.bfloat16, device="cuda"):
        path = WEIGHTS["llama"]
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(device).eval()
        self.n_core = self.model.config.num_hidden_layers

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)  # prepends BOS

    def structure(self, loops: int) -> dict:
        assert loops == 1
        return dict(n_prelude=0, n_core=self.n_core, n_coda=0, loops=1, sink_pos=0, train_loops=1)

    @torch.no_grad()
    def run(self, ids: torch.Tensor, loops: int = 1) -> Trace:
        assert loops == 1
        resid, attn, final = [], [], []
        hooks = []
        for layer in self.model.model.layers:
            hooks.append(layer.register_forward_hook(
                lambda mod, i, o: resid.append((o[0] if isinstance(o, tuple) else o)[0])))
            hooks.append(layer.self_attn.register_forward_hook(lambda mod, i, o: attn.append(o[1][0])))
        hooks.append(self.model.model.norm.register_forward_hook(lambda mod, i, o: final.append(o[0])))
        try:
            self.model(ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return Trace(resid=torch.stack(resid), attn=torch.stack(attn), loop_state=torch.stack(final),
                     readout=self.model.lm_head)
