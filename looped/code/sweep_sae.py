"""Short config sweep for the Huginn cross-loop SAE: several SAEs trained on the SAME activation stream.

Each sweep member is a (k, lr, lr schedule) config with its own SAETrainer on its own GPU; every batch from the
shared HuginnActivationProvider is fed to all members, so differences between configs are not due to data.
Producers (Huginn) run on --producer_gpus. Evaluation uses the fixed held-out cache (cache_eval.py): overall and
per-loop FVU, L0, and dead/alive feature counts, at several points during training -> sweep_results.json.

Batch size is the SAELens default (4096) for every member. d_sae is fixed at 32768: 65536 does not fit next to the
sglang server sharing these GPUs (measured: 32768 peaks at 9.2 GB of the 13.3 GB free, independent of k).
A GPU can host either one SAE (9.2 GB) or one Huginn producer (~8 GB), not both -> 4 SAEs + 4 producers per round.
Note on decay: with the "constant" scheduler SAELens decays linearly from lr to 0 over the last decay steps.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

from sae_lens.config import LoggingConfig, SAETrainerConfig  # noqa: E402
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig  # noqa: E402
from sae_lens.training.sae_trainer import SAETrainer  # noqa: E402

from sae_data import D_IN, HuginnActivationProvider  # noqa: E402

CONFIGS = {
    # name: (k, lr, scheduler, decay_frac)  decay_frac = fraction of steps with linear decay to 0 at the end
    "k32_lr3e-4_const": (32, 3e-4, "constant", 0.0),  # v1 config (baseline, checks the eval reproduces v1)
    "k32_lr3e-4_decay": (32, 3e-4, "constant", 0.2),
    "k32_lr1e-3_decay": (32, 1e-3, "constant", 0.2),
    "k64_lr3e-4_decay": (64, 3e-4, "constant", 0.2),
    "k128_lr3e-4_decay": (128, 3e-4, "constant", 0.2),
    "k64_lr1e-3_decay": (64, 1e-3, "constant", 0.2),
    "k64_lr1e-4_decay": (64, 1e-4, "constant", 0.2),
    # round 2 (lr 1e-3 ruled out in round 1: half the dictionary dies; decay looked strongly helpful)
    "k64_lr3e-4_const": (64, 3e-4, "constant", 0.0),  # decay control for k64_lr3e-4_decay (round 1)
    "k96_lr3e-4_decay": (96, 3e-4, "constant", 0.2),
    "k128_lr1e-4_decay": (128, 1e-4, "constant", 0.2),
}
ROUND2 = "k64_lr3e-4_const,k64_lr1e-4_decay,k96_lr3e-4_decay,k128_lr1e-4_decay"
# B3 seed-stability: same config as the full run (k128, lr 1e-4 + decay); a name suffix "@sN" sets the
# parameter-init seed, so members differ only in initialisation (identical data stream).
SEED_RUN = "k128_lr1e-4_decay@s1,k128_lr1e-4_decay@s2"
ROUND1 = "k32_lr3e-4_const,k32_lr1e-3_decay,k64_lr3e-4_decay,k128_lr3e-4_decay"


@torch.no_grad()
def evaluate(trainer, X, tags, loops, chunk=4096):
    """FVU / L0 overall and per loop on the fixed eval cache, using the trainer's (unfolded) SAE + activation scaler."""
    sae, scaler = trainer.sae, trainer.activation_scaler
    sae.eval()
    dev = sae.device
    d = sae.cfg.d_sae
    stats = {}
    fire = torch.zeros(d, device=dev)
    for l in list(loops) + ["all"]:
        idx = np.arange(len(tags)) if l == "all" else np.where(tags[:, 0] == l)[0]
        sse = tot = l0 = 0.0
        for c in np.array_split(idx, max(1, len(idx) // chunk)):
            x = scaler(X[c].to(dev, torch.float32))
            f = sae.encode(x)
            sse += ((sae.decode(f) - x) ** 2).sum().item()
            tot += ((x - x.mean(0)) ** 2).sum().item()
            l0 += (f > 0).float().sum().item()
            if l == "all":
                fire += (f > 0).float().sum(0)
        stats[str(l)] = {"fvu": sse / tot, "l0": l0 / len(idx)}
    freq = (fire / len(tags)).cpu().numpy()
    stats["alive(freq>1e-5)"] = int((freq > 1e-5).sum())
    stats["dead(freq==0)"] = int((freq == 0).sum())
    stats["train_dead(>2000 steps)"] = int(trainer.dead_neurons.sum().item())
    sae.train()
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=100_000_000, help="training vectors per config")
    p.add_argument("--configs", default=ROUND1, help="comma-separated config names")
    p.add_argument("--sae_gpus", default="0,1,2,3", help="one GPU per config")
    p.add_argument("--producer_gpus", default="4,5,6,7")
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--n_evals", type=int, default=5)
    p.add_argument("--eval_cache", default="/data_fs/zhangyuxuan/looped/sae/eval_cache.pt")
    p.add_argument("--eval_vectors", type=int, default=200_000)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--norm_batches", type=int, default=200)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.configs.split(",") if n]
    sae_gpus = [int(g) for g in args.sae_gpus.split(",")]
    assert len(sae_gpus) >= len(names), "one GPU per SAE (an SAE and a Huginn producer do not fit on one GPU)"
    prod_gpus = tuple(int(g) for g in args.producer_gpus.split(",") if g != "")

    ev = torch.load(args.eval_cache)
    rng = np.random.default_rng(0)
    sel = rng.choice(ev["X"].shape[0], min(args.eval_vectors, ev["X"].shape[0]), replace=False)
    X_eval, tags_eval, loops = ev["X"][sel], ev["tags"][sel].numpy(), ev["loops"]
    del ev

    provider = HuginnActivationProvider(train_batch_size=args.batch, context_size=512, n_seeds=2,
                                        device="cpu", producer_gpus=prod_gpus, buffer_vectors=2_000_000,
                                        max_rows=args.max_rows, shuffle_seed=0, split="train")
    steps = args.total // args.batch
    trainers = {}
    for i, n in enumerate(names):
        base, _, seed_s = n.partition("@s")
        k, lr, sched, decay_frac = CONFIGS[base]
        dev = f"cuda:{sae_gpus[i % len(sae_gpus)]}"
        torch.manual_seed(int(seed_s) if seed_s else 0)
        sae = BatchTopKTrainingSAE(BatchTopKTrainingSAEConfig(
            d_in=D_IN, d_sae=args.d_sae, k=k, device=dev, normalize_activations="expected_average_only_in"))
        cfg = SAETrainerConfig(
            total_training_samples=args.total, train_batch_size_samples=args.batch, lr=lr, lr_end=lr / 10,
            lr_scheduler_name=sched, lr_warm_up_steps=0, lr_decay_steps=int(decay_frac * steps), device=dev,
            logger=LoggingConfig(log_to_wandb=False), n_batches_for_norm_estimate=args.norm_batches)
        trainers[n] = SAETrainer(cfg=cfg, sae=sae, data_provider=iter(()))
        print(f"{n}: k={k} lr={lr} sched={sched} decay_steps={int(decay_frac * steps)} on {dev}", flush=True)

    # shared input-norm estimate (identical data for every member)
    norm_batches = [next(provider) for _ in range(args.norm_batches)]
    for t in trainers.values():
        t.activation_scaler.estimate_scaling_factor(d_in=D_IN, data_provider=iter(norm_batches),
                                                    n_batches_for_norm_estimate=args.norm_batches)
    del norm_batches
    print("scaling factor", next(iter(trainers.values())).activation_scaler.scaling_factor, flush=True)

    eval_at = set(int(steps * f) for f in np.linspace(0, 1, args.n_evals + 1)[1:])
    results = {n: {"config": dict(zip(["k", "lr", "sched", "decay_frac"], CONFIGS[n.partition("@s")[0]])),
                   "evals": []} for n in names}
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(provider)
        except StopIteration:
            print(f"data exhausted at step {step}", flush=True)
            break
        for t in trainers.values():
            t.maybe_reset_sparsity()
            t.step(batch)
            t.n_training_steps += 1
        if step % 500 == 0:
            print(f"step {step}/{steps}  {step * args.batch / (time.time() - t0):.0f} vec/s", flush=True)
        if step in eval_at:
            for n, t in trainers.items():
                r = evaluate(t, X_eval, tags_eval, loops)
                r["step"], r["vectors"] = step, step * args.batch
                results[n]["evals"].append(r)
                print(f"[eval {step}] {n}: FVU {r['all']['fvu']:.4f} L0 {r['all']['l0']:.1f} "
                      f"alive {r['alive(freq>1e-5)']} dead {r['dead(freq==0)']}", flush=True)
            json.dump(results, open(out / "sweep_results.json", "w"), indent=1)
    provider.close()
    for n, t in trainers.items():
        t.save_checkpoint(checkpoint_name=f"{n}_final", base_path=str(out))
    print("done ->", out / "sweep_results.json")


if __name__ == "__main__":
    main()
