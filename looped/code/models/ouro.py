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

    @torch.no_grad()
    def run_states(self, ids: torch.Tensor, loops: int, seed: int = 0) -> torch.Tensor:
        """Per-loop state = output of the end-of-loop `model.norm` (fed to the next loop and to lm_head), the Ouro
        analogue of Huginn's loop state. Returns (loops, T, D=2048). Ouro is deterministic; `seed` is ignored.
        """
        m = self.model
        m.config.total_ut_steps = loops
        m.model.total_ut_steps = loops
        out = []
        h = m.model.norm.register_forward_hook(lambda mod, i, o: out.append(o[0]))
        try:
            m(ids, use_cache=False)
        finally:
            h.remove()
        assert len(out) == loops, len(out)
        return torch.stack(out)

    @torch.no_grad()
    def run_layer_states(self, ids: torch.Tensor, loop: int, seed: int = 0) -> torch.Tensor:
        """Output (residual) of every decoder layer at the given loop. Returns (n_core=24, T, D). Pre end-of-loop
        norm; the last layer's output + norm is the loop state.
        """
        m = self.model
        m.config.total_ut_steps = loop
        m.model.total_ut_steps = loop
        out = []
        hooks = [layer.register_forward_hook(lambda mod, i, o: out.append(o[0])) for layer in m.model.layers]
        try:
            m(ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return torch.stack(out[-self.n_core:])

    def _readout(self, state: torch.Tensor) -> torch.Tensor:
        """Loop state (already post-norm) -> logits via lm_head, Ouro's per-loop exit readout."""
        return self.model.lm_head(state)
