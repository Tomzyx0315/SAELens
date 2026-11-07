"""B1: train one SAE per loop (each on that loop's state only), then build the cross-loop transfer matrix.

Producers stream, per document, the state at every target loop tagged by loop (reusing sae_data). Each loop's SAE is
trained only on batches of its own loop, drawn from a per-loop shuffle buffer. This gives SAEs that do NOT assume a
shared basis across loops, so the transfer matrix FVU[l_train, m_eval] = how well the loop-l SAE reconstructs loop-m
states directly answers "is the representation the same across loops?".

After training, every SAE is evaluated on the fixed eval cache (all loops), and we also match features between
loop SAEs by decoder cosine (Hungarian) to see how many features are shared.

Config per SAE: BatchTopK, d_sae 16384, k 64, lr 1e-4 + 20% decay (smaller than the main run for speed; identical
across loops so transfer is comparable). Trains on --device for the SAEs (one GPU each) with Huginn producers on
--producer_gpus.

Usage: python train_perloop_saes.py --loops 1,2,4,8,16,32 --total 120000000 --sae_gpus 0,1,2 --producer_gpus 3,4,5,6,7 --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

from scipy.optimize import linear_sum_assignment

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer

from sae_data import D_IN, HuginnActivationProvider


@torch.no_grad()
def fvu_on(sae, scaler, X, idx, dev, chunk=8192):
    sse = tot = 0.0
    for c in np.array_split(idx, max(1, len(idx) // chunk)):
        x = scaler(X[c].to(dev, torch.float32))
        sse += ((sae.decode(sae.encode(x)) - x) ** 2).sum().item()
        tot += ((x - x.mean(0)) ** 2).sum().item()
    return sse / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--loops", default="1,2,4,8,16,32")
    p.add_argument("--total", type=int, default=120_000_000, help="training vectors per loop-SAE")
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--sae_gpus", default="0,1,2")
    p.add_argument("--producer_gpus", default="3,4,5,6,7")
    p.add_argument("--eval_cache", default="/data_fs/zhangyuxuan/looped/sae/eval_cache.pt")
    p.add_argument("--eval_vectors", type=int, default=40000, help="eval vectors per loop")
    p.add_argument("--norm_batches", type=int, default=100)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loops = [int(x) for x in args.loops.split(",")]
    sae_gpus = [int(g) for g in args.sae_gpus.split(",")]
    prod_gpus = tuple(int(g) for g in args.producer_gpus.split(",") if g != "")
    json.dump({**vars(args), "loops": loops}, open(out / "args.json", "w"), indent=1)

    # stream only the needed loops; tag by loop so each SAE pulls its own
    provider = HuginnActivationProvider(train_batch_size=args.batch, context_size=512, n_seeds=2, device="cpu",
                                        producer_gpus=prod_gpus, buffer_vectors=2_000_000, shuffle_seed=0,
                                        split="train", loops=tuple(loops))
    steps = args.total // args.batch
    trainers, bufs = {}, {l: [] for l in loops}
    for i, l in enumerate(loops):
        dev = f"cuda:{sae_gpus[i % len(sae_gpus)]}"
        sae = BatchTopKTrainingSAE(BatchTopKTrainingSAEConfig(
            d_in=D_IN, d_sae=args.d_sae, k=args.k, device=dev, normalize_activations="expected_average_only_in"))
        cfg = SAETrainerConfig(total_training_samples=args.total, train_batch_size_samples=args.batch, lr=1e-4,
                               lr_end=1e-5, lr_scheduler_name="constant", lr_warm_up_steps=0,
                               lr_decay_steps=int(0.2 * steps), device=dev, logger=LoggingConfig(log_to_wandb=False),
                               n_batches_for_norm_estimate=args.norm_batches)
        trainers[l] = SAETrainer(cfg=cfg, sae=sae, data_provider=iter(()))
        print(f"loop {l}: SAE on {dev}", flush=True)

    # norm estimate per loop from the first batches
    warm = {l: [] for l in loops}
    need = args.norm_batches
    while min(len(v) for v in warm.values()) < need:
        b = next(provider)
        tg = provider.last_tags[:, 0]
        for l in loops:
            m = tg == l
            if m.any() and len(warm[l]) < need:
                warm[l].append(b[m])
    for l in loops:
        trainers[l].activation_scaler.estimate_scaling_factor(d_in=D_IN, data_provider=iter(warm[l]),
                                                              n_batches_for_norm_estimate=len(warm[l]))
    del warm
    print("norm estimates done", flush=True)

    # train: route each batch's loop-l rows to the loop-l SAE buffer, step when a full batch is available
    done = {l: 0 for l in loops}
    import time
    t0 = time.time()
    next_print = args.batch * 500
    while min(done.values()) < args.total:
        b = next(provider)
        tg = provider.last_tags[:, 0]
        for l in loops:
            if done[l] >= args.total:
                continue
            rows = b[tg == l]
            if rows.numel():
                bufs[l].append(rows)
            tot = sum(x.shape[0] for x in bufs[l])
            while tot >= args.batch and done[l] < args.total:
                cat = torch.cat(bufs[l])
                batch, rest = cat[: args.batch], cat[args.batch:]
                bufs[l] = [rest] if rest.numel() else []
                trainers[l].maybe_reset_sparsity()
                trainers[l].step(batch)
                trainers[l].n_training_steps += 1
                done[l] += args.batch
                tot = rest.shape[0]
        if min(done.values()) >= next_print:
            print(f"min done {min(done.values())}/{args.total}  {min(done.values())/(time.time()-t0):.0f} vec/s per SAE",
                  flush=True)
            next_print += args.batch * 500
    provider.close()

    # fold scaling, save
    for l in loops:
        t = trainers[l]
        if t.activation_scaler.scaling_factor is not None:
            t.sae.fold_activation_norm_scaling_factor(t.activation_scaler.scaling_factor)
            t.activation_scaler.scaling_factor = None
        t.sae.eval()
        t.save_checkpoint(checkpoint_name=f"loop{l}", base_path=str(out))

    # transfer matrix on the eval cache: FVU[l_train, m_eval]
    ev = torch.load(args.eval_cache)
    X, tags, ev_loops = ev["X"], ev["tags"].numpy(), list(ev["loops"])
    rng = np.random.default_rng(0)
    eval_idx = {m: rng.choice(np.where(tags[:, 0] == m)[0], args.eval_vectors, replace=False) for m in loops if m in ev_loops}
    from sae_lens.training.activation_scaler import ActivationScaler  # scaling already folded; identity scaler
    ident = ActivationScaler()
    T = np.full((len(loops), len(loops)), np.nan)
    for i, l in enumerate(loops):
        sae = trainers[l].sae
        dev = sae.device
        for j, m in enumerate(loops):
            if m in eval_idx:
                T[i, j] = fvu_on(sae, ident, X, eval_idx[m], dev)
        print(f"loop{l} SAE transfer FVU: " + " ".join(f"{m}:{T[i,j]:.3f}" for j, m in enumerate(loops) if not np.isnan(T[i,j])), flush=True)

    # decoder feature matching between consecutive loop SAEs (Hungarian on cosine)
    match = {}
    for a, b in zip(loops[:-1], loops[1:]):
        Wa = torch.nn.functional.normalize(trainers[a].sae.W_dec.detach().float(), dim=-1).cpu().numpy()
        Wb = torch.nn.functional.normalize(trainers[b].sae.W_dec.detach().float(), dim=-1).cpu().numpy()
        S = Wa @ Wb.T
        ri, ci = linear_sum_assignment(-S)
        cs = S[ri, ci]
        match[f"{a}->{b}"] = {"mean_matched_cos": round(float(cs.mean()), 3),
                              "frac_matched_cos>0.5": round(float((cs > 0.5).mean()), 3),
                              "frac_matched_cos>0.8": round(float((cs > 0.8).mean()), 3)}
        print(f"decoder match {a}->{b}: mean cos {match[f'{a}->{b}']['mean_matched_cos']}, "
              f">0.5 {match[f'{a}->{b}']['frac_matched_cos>0.5']}, >0.8 {match[f'{a}->{b}']['frac_matched_cos>0.8']}", flush=True)

    res = {"loops": loops, "transfer_fvu": T.round(4).tolist(), "decoder_match": match,
           "diag_fvu": {str(l): round(float(T[i, i]), 4) for i, l in enumerate(loops)}}
    json.dump(res, open(Path(__file__).resolve().parent.parent / "results/sae/perloop_transfer.json", "w"), indent=1)
    print("saved results/sae/perloop_transfer.json", flush=True)


if __name__ == "__main__":
    main()
