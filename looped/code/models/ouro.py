"""Ouro-1.4B: (0, 24, 0), no input injection, trained with 4 loops. Eager attention returns probabilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import Trace
from paths import WEIGHTS


class OuroAdapter:
    name = "ouro"
    train_loops = 4

    def __init__(self, dtype=torch.bfloat16, device="cuda"):
        path = WEIGHTS["ouro"]
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=dtype, attn_implementation="eager"
        ).to(device).eval()
        self.n_core = self.model.config.num_hidden_layers

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)  # no BOS (tokenizer default)

    def structure(self, loops: int) -> dict:
        return dict(n_prelude=0, n_core=self.n_core, n_coda=0, loops=loops, sink_pos=0, train_loops=self.train_loops)

    @torch.no_grad()
    def run(self, ids: torch.Tensor, loops: int) -> Trace:
        m = self.model
        m.config.total_ut_steps = loops
        m.model.total_ut_steps = loops
        resid, attn, normed = [], [], []
        hooks = []
        for layer in m.model.layers:
            hooks.append(layer.register_forward_hook(lambda mod, i, o: resid.append(o[0])))
            hooks.append(layer.self_attn.register_forward_hook(lambda mod, i, o: attn.append(o[1][0])))
        hooks.append(m.model.norm.register_forward_hook(lambda mod, i, o: normed.append(o[0])))
        try:
            m(ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        assert len(resid) == loops * self.n_core
        return Trace(resid=torch.stack(resid), attn=torch.stack(attn), loop_state=torch.stack(normed), readout=m.lm_head)
