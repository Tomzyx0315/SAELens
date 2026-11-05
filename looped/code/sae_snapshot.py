"""Mid-training snapshot analysis of a resume_* checkpoint, without touching the running job.

Loads the checkpoint's unfolded weights + activation scaler, streams a small held-out set of Huginn loop states
(different shuffle seed; Huginn runs on --gpu), and reports per loop: FVU, L0, number of alive features, and
how feature usage overlaps with the converged loop (32). Also: dead features overall, and the init-seed
consistency of feature usage per loop.

Usage: python sae_snapshot.py --ckpt /data_fs/.../resume_N --gpu 1 --sae_device cuda:1 --rows 64
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME

from sae_data import LOOPS, HuginnActivationProvider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gpu", type=int, default=1, help="GPU for the Huginn producer")
    p.add_argument("--sae_device", default="cuda:1")
    p.add_argument("--rows", type=int, default=64, help="held-out documents")
    p.add_argument("--context_size", type=int, default=512)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.sae_device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    print("scaling factor", scaler.scaling_factor, flush=True)

    prov = HuginnActivationProvider(train_batch_size=4096, context_size=args.context_size, n_seeds=2,
                                    device=args.sae_device, producer_gpus=(args.gpu,), buffer_vectors=10**9,
                                    max_rows=args.rows, shuffle_seed=4242)
    xs, tags = [], []
    for b in prov:
        xs.append(b)
        tags.append(prov.last_tags.copy())
    prov.close()
    X = torch.cat(xs)
    T = np.concatenate(tags)
    print("held-out vectors", X.shape[0], flush=True)

    d = sae.cfg.d_sae
    loops = list(LOOPS)
    res = {}
    fire_cnt = {}
    with torch.no_grad():
        for l in loops + ["all"]:
            for s in (0, 1, "all"):
                m = np.ones(len(T), bool)
                if l != "all":
                    m &= T[:, 0] == l
                if s != "all":
                    m &= T[:, 1] == s
                idx = torch.from_numpy(np.where(m)[0])
                if len(idx) == 0:
                    continue
                cnt = torch.zeros(d, device=args.sae_device)
                sse = tot = l0 = 0.0
                for c in idx.split(4096):
                    x = scaler(X[c])
                    f = sae.encode(x)
                    xh = sae.decode(f)
                    sse += ((xh - x) ** 2).sum().item()
                    tot += ((x - x.mean(0)) ** 2).sum().item()
                    l0 += (f > 0).float().sum().item()
                    cnt += (f > 0).float().sum(0)
                fire_cnt[(l, s)] = (cnt / len(idx)).cpu().numpy()
                if s == "all":
                    res[str(l)] = {"n": int(len(idx)), "fvu": sse / tot, "l0": l0 / len(idx),
                                   "alive_features(freq>1e-4)": int((fire_cnt[(l, s)] > 1e-4).sum())}

    def overlap(a, b):  # usage-distribution overlap in [0, 1]
        pa, pb = a / a.sum(), b / b.sum()
        return float(np.minimum(pa, pb).sum())

    ref = fire_cnt[(32, "all")]
    for l in loops:
        res[str(l)]["usage_overlap_with_loop32"] = overlap(fire_cnt[(l, "all")], ref)
        res[str(l)]["usage_overlap_seed0_vs_seed1"] = overlap(fire_cnt[(l, 0)], fire_cnt[(l, 1)])
        # features used at this loop but (almost) never at loop 32, and vice versa
        f_l = fire_cnt[(l, "all")]
        res[str(l)]["n_feat_only_here(>1e-3, <1e-4 at L32)"] = int(((f_l > 1e-3) & (ref < 1e-4)).sum())
        res[str(l)]["n_feat_only_L32(>1e-3, <1e-4 here)"] = int(((ref > 1e-3) & (f_l < 1e-4)).sum())
    res["dead_overall(freq<1e-6)"] = int((fire_cnt[("all", "all")] < 1e-6).sum())
    res["d_sae"] = d
    out = ckpt.parent / f"snapshot_{ckpt.name}.json"
    json.dump(res, open(out, "w"), indent=1)
    np.savez(ckpt.parent / f"snapshot_{ckpt.name}_freq.npz",
             loops=np.array(loops), freq=np.stack([fire_cnt[(l, "all")] for l in loops]),
             freq_seed0=np.stack([fire_cnt[(l, 0)] for l in loops]),
             freq_seed1=np.stack([fire_cnt[(l, 1)] for l in loops]))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
