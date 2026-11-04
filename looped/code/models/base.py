from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Trace:
    """Activations from one forward pass, depth-major (depth = loop * n_core + layer_in_block).

    - resid: (loops * n_core, T, D) residual after each recurrent-block layer
    - attn: (loops * n_core, H, T, T) attention probabilities of each recurrent-block layer
    - loop_state: (loops, T, D) state handed to the next loop (or the final hidden state for feedforward)
    - readout: maps one loop_state (T, D) to logits (T, V) with the model's own head; applied lazily, one loop
      at a time, to avoid materialising (loops, T, V)
    - resid_pre_norm: same shape as resid or None; for models whose layer output is normed (Huginn),
      the value before that final per-layer norm
    """

    resid: torch.Tensor
    attn: torch.Tensor
    loop_state: torch.Tensor
    readout: Callable[[torch.Tensor], torch.Tensor]
    resid_pre_norm: torch.Tensor | None = None

    def loop_logits(self, t: int) -> torch.Tensor:
        return self.readout(self.loop_state[t])
