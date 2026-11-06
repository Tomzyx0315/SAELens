"""Huginn-0125: (2, 4, 2)_I sandwich-norm blocks with linear input injection, trained with ~32 sampled loops.

The upstream attention only uses SDPA / flex attention and never returns probabilities, so we recompute
A = softmax(q k^T / sqrt(d) + causal) in fp32 from the block's own q/k inside a forward pre-hook on the
attention module, using exactly the upstream projection, qk-bias and rotary code path.
`check_attention()` verifies the recomputation by reproducing the module output (A v -> proj).

The recurrent state is initialised with random noise (`initialize_state`), so `run` seeds torch's RNG
per call to make every trace reproducible.
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import Trace
from paths import WEIGHTS


class HuginnAdapter:
    name = "huginn"
    train_loops = 32  # mean of the training recurrence distribution

    def __init__(self, dtype=torch.bfloat16, device="cuda"):
        path = WEIGHTS["huginn"]
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.n_core = cfg.n_layers_in_recurrent_block
        self.n_prelude = cfg.n_layers_in_prelude
        self.n_coda = cfg.n_layers_in_coda
        self.rope = self._import_rope()

    def _import_rope(self):
        import sys

        mod = sys.modules[type(self.model).__module__]
        return mod.apply_rotary_emb_complex_like

    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)  # tokenizer default (prepends BOS)

    def structure(self, loops: int) -> dict:
        return dict(
            n_prelude=self.n_prelude, n_core=self.n_core, n_coda=self.n_coda, loops=loops, sink_pos=0,
            train_loops=self.train_loops,
        )

    def _attn_probs(self, attn_mod, x: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute attention probabilities (H, T, T) fp32 and values (H, T, hd) for batch element 0."""
        B, S, E = x.shape
        q, k, v = attn_mod.Wqkv(x).split(attn_mod.chunks, dim=2)
        q = q.view(B, S, attn_mod.n_head, attn_mod.head_dim)
        k = k.view(B, S, attn_mod.n_kv_heads, attn_mod.head_dim)
        v = v.view(B, S, attn_mod.n_kv_heads, attn_mod.head_dim)
        if attn_mod.config.qk_bias:
            q_bias, k_bias = attn_mod.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = self.rope(q, k, freqs_cis=freqs_cis)
        q, k, v = (t[0].transpose(0, 1).float() for t in (q, k, v))  # (H, S, hd)
        scores = q @ k.transpose(-1, -2) / math.sqrt(attn_mod.head_dim)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        return scores.softmax(-1), v

    @torch.no_grad()
    def run(self, ids: torch.Tensor, loops: int, seed: int = 0, check: bool = False) -> Trace:
        m = self.model
        tr = m.transformer
        resid, pre_norm, attn, errs = [], [], [], []
        hooks = []

        def attn_pre(mod, args):
            x, freqs_cis = args[0], args[1]
            A, v = self._attn_probs(mod, x, freqs_cis)
            attn.append(A)
            if check:
                y = (A @ v).transpose(0, 1).reshape(1, x.shape[1], -1)
                errs.append(y)

        def attn_post(mod, args, out):
            if check:
                y = mod.proj(errs.pop().to(out.dtype))
                errs.append(((y - out).norm() / out.norm()).item())

        for block in tr.core_block:
            hooks.append(block.attn.register_forward_pre_hook(attn_pre))
            hooks.append(block.attn.register_forward_hook(attn_post))
            hooks.append(block.register_forward_hook(lambda mod, i, o: resid.append(o[0])))
            hooks.append(block.norm_4.register_forward_pre_hook(lambda mod, args: pre_norm.append(args[0][0])))
        torch.manual_seed(seed)
        try:
            m(ids, num_steps=loops, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        assert len(resid) == loops * self.n_core, len(resid)
        R = torch.stack(resid)
        state = R.view(loops, self.n_core, *R.shape[1:])[:, -1]  # output of the last core layer per loop
        trace = Trace(resid=R, attn=torch.stack(attn), loop_state=state, readout=self._readout,
                      resid_pre_norm=torch.stack(pre_norm))
        if check:
            trace.attn_check_rel_err = max(errs)  # type: ignore[attr-defined]
        return trace

    @torch.no_grad()
    def run_states(self, ids: torch.Tensor, loops: int, seed: int = 0) -> torch.Tensor:
        """Lightweight path for SAE data: only the per-loop state (last core-layer output), no attention recompute.

        Returns (loops, T, D) in model dtype. Identical values to `run(...).loop_state` for the same seed.
        """
        out = []
        h = self.model.transformer.core_block[-1].register_forward_hook(lambda mod, i, o: out.append(o[0]))
        torch.manual_seed(seed)
        try:
            self.model(ids, num_steps=loops, use_cache=False)
        finally:
            h.remove()
        assert len(out) == loops, len(out)
        return torch.stack(out)

    def _readout(self, state: torch.Tensor) -> torch.Tensor:
        """Latent after a loop (T, D) -> logits (T, V): ln_f -> coda -> ln_f -> lm_head (upstream predict_from_latents)."""
        return self.model.predict_from_latents(state[None]).logits[0]

    @torch.no_grad()
    def final_logits(self, ids: torch.Tensor, loops: int, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        return self.model(ids, num_steps=loops, use_cache=False).logits[0]

    @torch.no_grad()
    def run_with_state_edit(self, ids, loops, edit_fn, edit_loops, seed=0):
        """Run Huginn, replacing the loop state (output of core_block[-1]) at the given loops with edit_fn(state).

        edit_fn maps a (T, D) state to a (T, D) state (e.g. SAE reconstruction); edit_loops is a set of 1-based
        loop indices to edit. The edited state feeds the next loop, so the edit propagates. Returns final logits
        (T, V) and the list of per-loop states actually produced (post-edit), for error-propagation analysis.
        """
        counter = {"i": 0}
        states = []

        def hook(mod, inp, out):
            counter["i"] += 1
            x = out[0] if isinstance(out, tuple) else out
            s = x[0]  # (T, D)
            if counter["i"] in edit_loops:
                s = edit_fn(s).to(s.dtype)
                x = (s.unsqueeze(0), *out[1:]) if isinstance(out, tuple) else s.unsqueeze(0)
            states.append(s.detach())
            return x

        h = self.model.transformer.core_block[-1].register_forward_hook(hook)
        torch.manual_seed(seed)
        try:
            logits = self.model(ids, num_steps=loops, use_cache=False).logits[0]
        finally:
            h.remove()
        return logits, states
