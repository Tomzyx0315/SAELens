"""Compare "early" vs "late-emerging" SAE features: definition from a loop snapshot, semantics from interpret_loops.

Early feature:        freq at loop 1 > thr  AND  freq at the converged loop < freq_loop1 / ratio
Late-emerging:        freq at converged loop > thr  AND  freq at loop 1 < freq_conv / ratio
Persistent:           freq > thr at both

Inputs:
- a snapshot npz (`snapshot_*_freq.npz` from sae_snapshot.py) OR the per-loop freq in interpret_loops' npz, giving
  per-feature firing frequency at each loop (all-domain);
- interpret_loops' json for the max-activating contexts (loop 1 and the converged loop).

Reports: counts of each class; per-domain composition of early vs late features (are early features more web/code/
math?); and example contexts (loop-1 context for early features, converged-loop context for late features).

Usage: python early_vs_late.py --interp results/sae/interp_v2_640M_loops.json \
         --freq_npz /data_fs/.../interp/interp_v2_640M_loops.npz --out results/sae/early_vs_late_640M.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq_npz", required=True, help="interpret_loops npz with freq_dom_<loop> arrays")
    p.add_argument("--interp", required=True, help="interpret_loops json for example contexts")
    p.add_argument("--loop_early", type=int, default=1)
    p.add_argument("--loop_conv", type=int, default=32)
    p.add_argument("--thr", type=float, default=1e-3)
    p.add_argument("--ratio", type=float, default=5.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z = np.load(args.freq_npz, allow_pickle=True)
    fe = z[f"freq_dom_{args.loop_early}"]  # (n_dom, d_sae)
    fc = z[f"freq_dom_{args.loop_conv}"]
    # token-count weights per domain unknown here; approximate all-domain freq by domain mean (domains are similar size)
    e = fe.mean(0)
    c = fc.mean(0)
    d_sae = len(e)

    early = (e > args.thr) & (c < e / args.ratio)
    late = (c > args.thr) & (e < c / args.ratio)
    persistent = (e > args.thr) & (c > args.thr) & ~early & ~late

    def dom_profile(mask, freq_dom):
        # for features in mask, which domains do they fire in? mean over features of per-domain share of firing
        f = freq_dom[:, mask]  # (n_dom, n_feat)
        share = f / np.maximum(f.sum(0, keepdims=True), 1e-12)
        return dict(zip(DOMAINS, share.mean(1).round(3).tolist()))

    res = {
        "n_features": int(d_sae),
        "counts": {"early": int(early.sum()), "late_emerging": int(late.sum()), "persistent": int(persistent.sum())},
        "domain_profile_early(at loop %d)" % args.loop_early: dom_profile(early, fe),
        "domain_profile_late(at loop %d)" % args.loop_conv: dom_profile(late, fc),
        "domain_profile_persistent(at loop %d)" % args.loop_conv: dom_profile(persistent, fc),
    }

    # example contexts from the interp json (only for the sampled features it contains)
    interp = json.load(open(args.interp))["features"]
    res["early_examples"] = []
    res["late_examples"] = []
    for fid, info in interp.items():
        i = int(fid)
        le = str(args.loop_early)
        lc = str(args.loop_conv)
        if early[i] and info["top_by_loop"].get(le):
            res["early_examples"].append({"feat": i, "freq_e": round(float(e[i]), 5), "freq_c": round(float(c[i]), 5),
                                          "ctx_loop%d" % args.loop_early: [t["text"][:120] for t in info["top_by_loop"][le][:3]]})
        if late[i] and info["top_by_loop"].get(lc):
            res["late_examples"].append({"feat": i, "freq_e": round(float(e[i]), 5), "freq_c": round(float(c[i]), 5),
                                         "ctx_loop%d" % args.loop_conv: [t["text"][:120] for t in info["top_by_loop"][lc][:3]]})

    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("examples")}, indent=1))
    print(f"\nearly examples: {len(res['early_examples'])}, late examples: {len(res['late_examples'])} "
          f"(limited to the {len(interp)} sampled features in the interp json)")
    for e_ in res["early_examples"][:6]:
        print(f"  EARLY feat {e_['feat']} (e={e_['freq_e']}, c={e_['freq_c']}):", e_[f"ctx_loop{args.loop_early}"][0])
    for l_ in res["late_examples"][:6]:
        print(f"  LATE  feat {l_['feat']} (e={l_['freq_e']}, c={l_['freq_c']}):", l_[f"ctx_loop{args.loop_conv}"][0])


if __name__ == "__main__":
    main()
