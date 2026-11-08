"""C1: causal test of whether SAE features drive the forming prediction, with persistent (every-loop) intervention.

A single-loop edit is overwritten by the recurrence (shown in splice_eval), so we intervene at *every* loop from a
start loop onward: at each such loop we edit the loop state by modifying one SAE feature, then let the recurrence
continue. The feature edit is done in SAE space:

    edit(state) = state - a_j(state) * W_dec[j]        (ablate feature j: remove its current contribution)
    edit(state) = state + alpha * mean_act_j * W_dec[j] (steer feature j up by alpha standard activations)

We track the model's own per-loop logit-lens prediction of the next token at the last position, and the CE on a
target completion, so we can see whether ablating a feature delays / prevents the prediction from forming, and
whether a feature that is active exactly when the prediction flips is causally responsible.

The target feature(s) for a prompt are found automatically: the features most correlated with the prediction flip
(active after the flip loop, inactive before) at the last position, ranked by activation * decoder-norm.

Usage: python causal_steer.py --ckpt <final> --device cuda:0 --out causal_steer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler

from models import get_adapter

ALL_LOOPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
PROMPTS = {
    "yen": ("The capital of the country whose currency is the yen is", " Tokyo"),
    "sqrt": ("The square root of 144 is", " 12"),
    "france": ("The capital of France is", " Paris"),
    "antonym": ("The opposite of the word 'hot' is", " cold"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_loops", type=int, default=32)
    p.add_argument("--n_target", type=int, default=8, help="top features to intervene on (jointly and singly)")
    p.add_argument("--steer_alpha", type=float, default=-1.0, help="unused scalar kept for reference")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)
    W = sae.W_dec.detach().float()  # (d_sae, d_in)
    dnorm = W.norm(dim=-1)

    def feats(state):  # (T, d) model space -> (T, d_sae) SAE activations
        return sae.encode(scaler(state.float()))

    def lastpred(state):
        return ad.tok.decode([int(ad._readout(state)[-1].argmax())])

    out = {"prompts": {}}
    for name, (text, target) in PROMPTS.items():
        ids = ad.encode(text)
        tgt_id = ad.tok(target, add_special_tokens=False).input_ids[0]
        with torch.no_grad():
            st = ad.run_states(ids, args.max_loops, seed=args.seed)  # (L, T, d)
            preds = [lastpred(st[l - 1]) for l in ALL_LOOPS if l <= args.max_loops]
            loops = [l for l in ALL_LOOPS if l <= args.max_loops]
            # flip loop: first loop whose prediction == final prediction and != loop-1 prediction
            final = preds[-1]
            flip_i = next((i for i, pp in enumerate(preds) if pp == final and preds[0] != final), 0)
            flip_loop = loops[flip_i]
            # candidate features: active at last position at the final loop but not at loop 1, ranked by act*dnorm
            f_last_final = feats(st[-1])[-1]
            f_last_1 = feats(st[0])[-1]
            cand = torch.where((f_last_final > 0) & (f_last_1 == 0))[0]
            score = f_last_final[cand] * dnorm[cand]
            order = cand[score.argsort(descending=True)][: args.n_target]
            target_feats = order.tolist()

        # mean activation of each target feature (for steering magnitude); from the final-loop last position
        mean_act = {int(f): float(f_last_final[f]) for f in target_feats}

        def ablate_fn(state, fset):
            f = feats(state)
            delta = torch.zeros_like(state)
            for j in fset:
                delta = delta - f[:, j:j + 1] * W[j]
            return (state + delta).to(state.dtype)

        results = {"text": text, "target": target, "baseline_pred_by_loop": dict(zip(map(str, loops), preds)),
                   "flip_loop": flip_loop, "final_pred": final, "target_feats": target_feats, "interventions": {}}

        # for each start loop, ablate the whole target set at every loop >= start, record predictions + target CE
        for start in [1, max(1, flip_loop - 2), flip_loop]:
            edit_loops = set(l for l in range(start, args.max_loops + 1))
            with torch.no_grad():
                logits, states = ad.run_with_state_edit(
                    ids, args.max_loops, lambda s: ablate_fn(s, target_feats), edit_loops, seed=args.seed)
                # per-loop readout under the running intervention
                pr = [lastpred(states[l - 1]) for l in loops]
                ce = torch.nn.functional.cross_entropy(logits[-1:].float(), torch.tensor([tgt_id], device=args.device)).item()
            results["interventions"][f"ablate_all_from_loop{start}"] = {
                "pred_by_loop": dict(zip(map(str, loops), pr)), "final_pred": pr[-1],
                "target_token_ce": round(ce, 3), "target_token_prob": round(float(torch.softmax(logits[-1].float(), -1)[tgt_id]), 4)}

        # baseline target CE (no edit) and single-feature ablations from loop 1
        with torch.no_grad():
            base_ce = torch.nn.functional.cross_entropy(ad._readout(st[-1])[-1:].float(), torch.tensor([tgt_id], device=args.device)).item()
        results["baseline_target_ce"] = round(base_ce, 3)
        single = {}
        for j in target_feats[:4]:
            with torch.no_grad():
                logits, _ = ad.run_with_state_edit(ids, args.max_loops, lambda s: ablate_fn(s, [j]),
                                                   set(range(1, args.max_loops + 1)), seed=args.seed)
                ce = torch.nn.functional.cross_entropy(logits[-1:].float(), torch.tensor([tgt_id], device=args.device)).item()
            single[str(j)] = {"target_ce": round(ce, 3), "final_pred": ad.tok.decode([int(logits[-1].argmax())])}
        results["single_feature_ablation_from_loop1"] = single
        # control: ablate the whole set at a SINGLE loop only (should be undone by the recurrence)
        single_loop_ctrl = {}
        for one in (1, flip_loop, args.max_loops):
            with torch.no_grad():
                logits, _ = ad.run_with_state_edit(ids, args.max_loops, lambda s: ablate_fn(s, target_feats),
                                                   {one}, seed=args.seed)
                ce = torch.nn.functional.cross_entropy(logits[-1:].float(), torch.tensor([tgt_id], device=args.device)).item()
            single_loop_ctrl[f"loop{one}"] = {"target_ce": round(ce, 3), "final_pred": ad.tok.decode([int(logits[-1].argmax())])}
        results["single_loop_ablation_control"] = single_loop_ctrl
        out["prompts"][name] = results
        print(f"\n=== {name}: '{text}' -> target '{target}'")
        print(f"  baseline: flip at loop {flip_loop}, final '{final}', target CE {base_ce:.2f}")
        for k, v in results["interventions"].items():
            print(f"  {k}: final '{v['final_pred']}' target CE {v['target_token_ce']} (p={v['target_token_prob']})")
        print(f"  single-feature ablations: " + ", ".join(f"{j}->CE {d['target_ce']}('{d['final_pred']}')" for j, d in single.items()))
        print(f"  single-LOOP control (one loop only): " + ", ".join(f"{k}->CE {v['target_ce']}('{v['final_pred']}')" for k, v in single_loop_ctrl.items()))

    path = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
