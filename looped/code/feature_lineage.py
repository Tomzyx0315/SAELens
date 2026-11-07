"""Feature lineage across loops (A3): when a feature switches off between loop l and loop l', which features switch
on in its place on the same token? Distinguishes "same concept re-encoded by another latent" from "concept replaced".

For consecutive sampled loops (l, l') and each token we take OFF = active at l but not l', ON = active at l' but not l.
Over many tokens we accumulate a co-occurrence count C[i, j] = #tokens where i is OFF and j is ON (restricted to
features that turn off often). For each OFF feature i we report its top successors j by
  - transition probability P(j ON | i OFF)
  - lift = P(j ON | i OFF) / P(j ON) (specificity beyond j just being a common newcomer)
  - decoder cosine cos(W_dec[i], W_dec[j]) (are successor directions similar?)

Successors are ranked by P(j ON | i OFF) among features that turn on often enough (marginal ON rate > 1e-3), so
rare features do not dominate via inflated lift.

Baseline: the decoder cosine of the top successors is compared with that of random ON features (null), and
with each OFF feature's nearest decoder neighbour in the whole dictionary. A high successor cosine would mean
"re-encoding of the same direction"; low cosine with high lift means a systematic, but different, replacement.

Token-level re-encoding test: on each token, v_off = sum_{i in OFF} a_i W_dec[i] (loop l) and
v_on = sum_{j in ON} a_j W_dec[j] (loop l'). cos(v_off, v_on) near 1 would mean the newcomers rebuild the direction
of the features that left (pure re-encoding); the null pairs v_off with v_on of another token in the same document.

Usage: python feature_lineage.py --ckpt <final> --device cuda:2 --out feature_lineage
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler

from models import get_adapter
from paths import DATA

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]
PAIRS = [(1, 2), (2, 4), (4, 8), (8, 16)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--docs_per_domain", type=int, default=150)
    p.add_argument("--n_track", type=int, default=2000, help="most frequently switching-off features tracked per pair")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)
    d = sae.cfg.d_sae
    W = sae.W_dec.detach().float()
    Wn = W / W.norm(dim=-1, keepdim=True)

    # pass 1: collect binary activations per pair (store as sparse index lists on CPU to bound memory)
    loops_needed = sorted({l for pr in PAIRS for l in pr})
    acts = {l: [] for l in loops_needed}  # list of (T, d) bool tensors on CPU
    tok_cos = {f"{a}->{b}": {"real": [], "null": [], "off_frac": []} for a, b in PAIRS}
    for dom in DOMAINS:
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(args.docs_per_domain, len(df)), random_state=1)
        for _, row in df.iterrows():
            ids = torch.tensor(list(row["input_ids"]), device=args.device).unsqueeze(0)
            with torch.no_grad():
                st = ad.run_states(ids, max(loops_needed), seed=0)[:, 1:]
                f = {l: sae.encode(scaler(st[l - 1].float())) for l in loops_needed}
                for l in loops_needed:
                    acts[l].append((f[l] > 0).cpu())
                for a, b in PAIRS:
                    off_m = (f[a] > 0) & (f[b] <= 0)
                    on_m = (f[a] <= 0) & (f[b] > 0)
                    v_off = (f[a] * off_m) @ W
                    v_on = (f[b] * on_m) @ W
                    v_all = f[a] @ W
                    ok = (v_off.norm(dim=-1) > 0) & (v_on.norm(dim=-1) > 0)
                    if ok.sum() < 2:
                        continue
                    vo, vn = v_off[ok], v_on[ok]
                    real = torch.nn.functional.cosine_similarity(vo, vn, dim=-1)
                    perm = torch.roll(torch.arange(len(vn), device=vn.device), 1)
                    null = torch.nn.functional.cosine_similarity(vo, vn[perm], dim=-1)
                    key = f"{a}->{b}"
                    tok_cos[key]["real"].append(real.mean().item())
                    tok_cos[key]["null"].append(null.mean().item())
                    tok_cos[key]["off_frac"].append((vo.norm(dim=-1) / v_all[ok].norm(dim=-1)).mean().item())
        print(f"{dom}: collected", flush=True)
    A = {l: torch.cat(acts[l]) for l in loops_needed}  # (N_tok, d) bool
    n_tok = A[loops_needed[0]].shape[0]
    print("tokens", n_tok, flush=True)

    res = {"n_tokens": int(n_tok), "pairs": {}}
    for a, b in PAIRS:
        off = (A[a] & ~A[b])
        on = (A[a].logical_not() & A[b])
        off_count = off.sum(0).float()
        on_rate = on.float().mean(0)  # marginal P(j ON)
        track = torch.argsort(off_count, descending=True)[: args.n_track]
        # co-occurrence C[i in track, j] = sum over tokens off[:, i] * on[:, j], computed in chunks on GPU
        C = torch.zeros(len(track), d, device=args.device)
        offt = off[:, track]
        for s in range(0, n_tok, 4096):
            C += offt[s:s + 4096].to(args.device).float().T @ on[s:s + 4096].to(args.device).float()
        P = C / off_count[track].to(args.device)[:, None].clamp(min=1)  # P(j ON | i OFF)
        onr = on_rate.to(args.device)[None, :]
        lift = P / onr.clamp(min=1e-6)
        P_rank = P.clone()
        P_rank[(onr < 1e-3).expand_as(P_rank)] = -1  # rank successors by P among non-rare newcomers
        Wt = Wn[track.to(args.device)]
        top_succ = P_rank.topk(5, dim=1).indices  # (n_track, 5)
        cos_succ = (Wt[:, None, :] * Wn[top_succ]).sum(-1)  # (n_track, 5)
        p_succ = P.gather(1, top_succ)
        lift_succ = lift.gather(1, top_succ)
        # null: random ON features weighted by their ON rate
        rnd = torch.multinomial(on_rate.to(args.device).clamp(min=0) + 1e-12, len(track) * 5, replacement=True)
        cos_null = (Wt.repeat_interleave(5, 0) * Wn[rnd]).sum(-1)
        # nearest decoder neighbour of each OFF feature in the whole dictionary (upper bound for "same direction")
        sims = Wt @ Wn.T
        sims[torch.arange(len(track)), track.to(args.device)] = -1
        nn_cos = sims.max(1).values
        tc = tok_cos[f"{a}->{b}"]
        res["pairs"][f"{a}->{b}"] = {
            "token_cos_off_vs_on": round(float(np.mean(tc["real"])), 4),
            "token_cos_off_vs_on_null": round(float(np.mean(tc["null"])), 4),
            "off_part_norm_frac_of_state": round(float(np.mean(tc["off_frac"])), 3),
            "mean_off_per_token": round(float(off.sum(1).float().mean()), 2),
            "mean_on_per_token": round(float(on.sum(1).float().mean()), 2),
            "top1_successor_P": [round(float(x), 3) for x in np.percentile(p_succ[:, 0].cpu().numpy(), [25, 50, 75])],
            "top1_successor_lift_median": round(float(lift_succ[:, 0].median()), 2),
            "decoder_cos_top1_successor": [round(float(x), 3) for x in np.percentile(cos_succ[:, 0].cpu().numpy(), [25, 50, 75])],
            "decoder_cos_top5_successors_mean": round(float(cos_succ.mean()), 3),
            "decoder_cos_null_mean": round(float(cos_null.mean()), 3),
            "decoder_cos_nearest_neighbour_median": round(float(nn_cos.median()), 3),
            "frac_top1_successor_cos_gt_0.5": round(float((cos_succ[:, 0] > 0.5).float().mean()), 3),
            "examples": [{"off": int(track[i]), "succ": top_succ[i].tolist(), "P": [round(float(x), 3) for x in p_succ[i]],
                          "lift": [round(float(x), 1) for x in lift_succ[i]], "cos": [round(float(x), 3) for x in cos_succ[i]]}
                         for i in range(10)],
        }
        r = res["pairs"][f"{a}->{b}"]
        print(f"{a}->{b}: token cos(v_off, v_on) {r['token_cos_off_vs_on']} (null {r['token_cos_off_vs_on_null']}, "
              f"off-part norm frac {r['off_part_norm_frac_of_state']})", flush=True)
        print(f"{a}->{b}: off/token {r['mean_off_per_token']} on/token {r['mean_on_per_token']} | top1 P quartiles "
              f"{r['top1_successor_P']} lift med {r['top1_successor_lift_median']} | decoder cos top1 quartiles "
              f"{r['decoder_cos_top1_successor']} (null {r['decoder_cos_null_mean']}, nearest-nbr med "
              f"{r['decoder_cos_nearest_neighbour_median']}) frac cos>0.5 {r['frac_top1_successor_cos_gt_0.5']}", flush=True)

    out = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
