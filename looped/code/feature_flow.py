"""Feature flow within a single sentence: how do SAE features switch on/off across loops along the recurrent
trajectory of each token?

The SAE lives on the loop state (end-of-loop latent), so for a fixed prompt we get, per token, a (loops, d_sae)
binary activation map. We summarize the flow with:

- per (token, feature) onset/offset loops: the first and last loop where the feature is active, and whether it is
  "early-only", "late-only", "persistent", or "transient" (on then off before the end);
- transition rates between consecutive loops: fraction of active features that turn off, stay on, or newly turn on;
- for the final token (the one that predicts the next word), the named features active at loop 1 vs loop 32, using
  the interpretation of those feature ids when available.

Also decodes the logit-lens prediction after each loop (the model's own per-loop readout) so the feature flow can
be related to how the prediction forms.

Outputs a JSON with the per-token flow summary and, for a few chosen token positions, the ordered feature timeline.

Usage: python feature_flow.py --ckpt <final> --device cuda:0 --out <stem> [--prompt "..."]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME

from models import get_adapter

PROMPTS = {
    "math": "Question: What is 17 times 24? Answer: 17 times 24 equals",
    "fact": "The capital of the country whose currency is the yen is",
    "code": "def add(a, b):\n    return a +",
}
ALL_LOOPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", default=None, help="raw text; overrides --preset")
    p.add_argument("--preset", default="math", choices=list(PROMPTS))
    p.add_argument("--max_loops", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interp", default=None, help="interp json (feat->top contexts) to name features")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    text = args.prompt or PROMPTS[args.preset]
    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)
    loops = [l for l in ALL_LOOPS if l <= args.max_loops]

    ids = ad.encode(text)
    toks = [ad.tok.decode([t]) for t in ids[0].tolist()]
    with torch.no_grad():
        st = ad.run_states(ids, args.max_loops, seed=args.seed)  # (max_loops, T, d)
        acts = torch.stack([(sae.encode(scaler(st[l - 1].float())) > 0) for l in loops])  # (n_loops, T, d_sae)
        # per-loop logit-lens next-token prediction for the last position
        lastpred = []
        for l in loops:
            logits = ad._readout(st[l - 1])[-1]
            lastpred.append(ad.tok.decode([int(logits.argmax())]))
    acts = acts.cpu().numpy()  # (L, T, d)
    L, T, d = acts.shape

    # transition rates between consecutive loops (per token, averaged)
    trans = []
    for i in range(1, L):
        a, b = acts[i - 1], acts[i]  # (T, d)
        on = a.sum(1)
        stay = (a & b).sum(1)
        off = (a & ~b).sum(1)
        newon = (~a & b).sum(1)
        trans.append({"from_loop": loops[i - 1], "to_loop": loops[i],
                      "mean_active": float(on.mean()),
                      "frac_stay": float((stay / np.maximum(on, 1)).mean()),
                      "frac_turn_off": float((off / np.maximum(on, 1)).mean()),
                      "mean_newly_on": float(newon.mean())})

    # per (token,feature) onset/offset classification
    classes = {"early_only": 0, "late_only": 0, "persistent": 0, "transient": 0, "intermittent": 0}
    for t in range(T):
        A = acts[:, t, :]  # (L, d)
        active_feats = np.where(A.any(0))[0]
        for f in active_feats:
            col = A[:, f]
            first, last = col.argmax(), L - 1 - col[::-1].argmax()
            n = col.sum()
            span = last - first + 1
            if n == span:  # contiguous
                if first == 0 and last == L - 1:
                    classes["persistent"] += 1
                elif first == 0:
                    classes["early_only"] += 1
                elif last == L - 1:
                    classes["late_only"] += 1
                else:
                    classes["transient"] += 1
            else:
                classes["intermittent"] += 1

    names = {}
    if args.interp and Path(args.interp).exists():
        interp = json.load(open(args.interp))
        feats = interp.get("features", interp.get("early", {}))  # support both json shapes
        for fid, info in feats.items():
            top = info.get("top") or (info.get("top_by_loop", {}).get("8") or next(iter(info.get("top_by_loop", {}).values()), []))
            if top:
                names[int(fid)] = top[0]["text"][:60]

    # last-token timeline: for each feature active at any loop on the last token, its on/off vector
    last = acts[:, -1, :]
    lt = []
    for f in np.where(last.any(0))[0]:
        onvec = last[:, f].astype(int).tolist()
        lt.append({"feat": int(f), "on_by_loop": dict(zip([str(l) for l in loops], onvec)),
                   "name": names.get(int(f), "")})
    lt.sort(key=lambda r: (r["on_by_loop"][str(loops[0])], -sum(r["on_by_loop"].values())), reverse=True)

    res = {"text": text, "loops": loops, "tokens": toks,
           "last_token_pred_by_loop": dict(zip([str(l) for l in loops], lastpred)),
           "transition_rates": trans,
           "token_feature_classes": classes,
           "last_token_timeline": lt[:60]}
    json.dump(res, open(Path("../results/sae") / f"{args.out}.json", "w"), indent=1, ensure_ascii=False)

    print("prompt:", text)
    print("last-token prediction by loop:", res["last_token_pred_by_loop"])
    print("\ntransition rates (mean active / frac stay / frac off / mean newly on):")
    for tr in trans:
        print(f"  {tr['from_loop']:2d}->{tr['to_loop']:2d}: active {tr['mean_active']:6.1f} "
              f"stay {tr['frac_stay']:.2f} off {tr['frac_turn_off']:.2f} newon {tr['mean_newly_on']:5.1f}")
    tot = sum(classes.values())
    print("\n(token,feature) activation classes:", {k: f"{v} ({100*v/tot:.0f}%)" for k, v in classes.items()})
    print("saved", Path("../results/sae") / f"{args.out}.json")


if __name__ == "__main__":
    main()
