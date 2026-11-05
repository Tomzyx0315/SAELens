"""Train a shared cross-loop BatchTopK SAE on Huginn-0125 loop states, streaming activations (no disk cache).

Huginn runs in producer processes (one per GPU in --producer_gpus); the SAE trains on --device. After training,
a fresh held-out stream is passed through the final SAE to record, per latent, its activation frequency split by
loop and by init seed (`feature_loop_tags.npz`), to classify features as early / persistent / late-emerging and to
separate genuine computation from init-noise.

Smoke (in-process, CPU):  python train_sae.py --out <dir> --total 16384 --d_sae 512 --k 8 --batch 1024 \
                            --context_size 128 --device cpu --max_rows 60 --tag_batches 4 --norm_batches 3
Full (8 GPUs):            python train_sae.py --out <dir> --device cuda:0 --producer_gpus 1,2,3,4,5,6,7
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True  # SAE matmuls in TF32 (~2x on 4090); weights/optimizer stay fp32
torch.backends.cudnn.allow_tf32 = True

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer

from sae_data import D_IN, LOOPS, HuginnActivationProvider


@torch.no_grad()
def tag_features(sae, provider: HuginnActivationProvider, loops, n_batches: int):
    """Per-latent firing frequency split by loop and by init seed, over a fresh stream."""
    d = sae.cfg.d_sae
    loop_list = list(loops)
    fire_loop = np.zeros((len(loop_list), d))
    n_loop = np.zeros(len(loop_list), dtype=np.int64)
    fire_seed = np.zeros((provider.n_seeds, d))
    n_seed = np.zeros(provider.n_seeds, dtype=np.int64)
    sae.eval()
    for _ in range(n_batches):
        try:
            batch = next(provider)
        except StopIteration:
            break
        fired = (sae.encode(batch.to(sae.device)) > 0).float().cpu().numpy()
        loops_t, seeds_t = provider.last_tags[:, 0], provider.last_tags[:, 1]
        for i, l in enumerate(loop_list):
            m = loops_t == l
            if m.any():
                fire_loop[i] += fired[m].sum(0)
                n_loop[i] += m.sum()
        for s in range(provider.n_seeds):
            m = seeds_t == s
            if m.any():
                fire_seed[s] += fired[m].sum(0)
                n_seed[s] += m.sum()
    return {
        "loops": np.array(loop_list),
        "freq_by_loop": fire_loop / np.maximum(n_loop[:, None], 1),
        "freq_by_seed": fire_seed / np.maximum(n_seed[:, None], 1),
        "count_by_loop": n_loop,
        "count_by_seed": n_seed,
    }


def _save_resume(trainer, out: Path):
    """Save a resumable checkpoint (unfolded weights + trainer state incl. activation scaler) and keep only the latest."""
    name = f"resume_{trainer.n_training_samples}"
    trainer.save_checkpoint(checkpoint_name=name)
    for old in sorted(out.glob("resume_*"), key=lambda p: int(p.name.split("_")[1]))[:-1]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"checkpoint {name}", flush=True)


def run_training_with_checkpoints(trainer, provider, args, out: Path):
    """Mirror SAETrainer.fit() but save a resumable checkpoint every args.checkpoint_every vectors and fold the
    input-norm scaling into the weights only once, at the very end. Checkpoints store unfolded weights + the
    activation scaler, so resuming reuses the original scaling estimate instead of re-estimating it."""
    from tqdm.auto import tqdm

    trainer.sae.to(trainer.cfg.device)
    if (trainer.sae.cfg.normalize_activations == "expected_average_only_in"
            and trainer.activation_scaler.scaling_factor is None):
        trainer.activation_scaler.estimate_scaling_factor(
            d_in=trainer.sae.cfg.d_in, data_provider=provider,
            n_batches_for_norm_estimate=trainer.cfg.n_batches_for_norm_estimate,
        )
    pbar = tqdm(total=args.total, initial=trainer.n_training_samples, desc="Training SAE")
    next_ckpt = (trainer.n_training_samples // args.checkpoint_every + 1) * args.checkpoint_every
    while trainer.n_training_samples < args.total:
        trainer.maybe_reset_sparsity()
        out_step = trainer.step(next(provider))
        trainer.n_training_steps += 1
        trainer._update_pbar(out_step, pbar)
        if trainer.n_training_samples >= next_ckpt and trainer.n_training_samples < args.total:
            _save_resume(trainer, out)
            next_ckpt += args.checkpoint_every
    pbar.close()
    # fold the scaling into the weights once, record final metadata, save the final checkpoint
    if trainer.activation_scaler.scaling_factor is not None:
        trainer.sae.fold_activation_norm_scaling_factor(trainer.activation_scaler.scaling_factor)
        trainer.activation_scaler.scaling_factor = None
    trainer.set_final_sae_metadata()
    trainer.save_checkpoint(checkpoint_name=f"final_{trainer.n_training_samples}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=1_000_000_000, help="total activation vectors to train on")
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--k", type=float, default=32)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--context_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--producer_gpus", default="", help="comma-separated GPU ids running Huginn; empty = in-process")
    p.add_argument("--buffer", type=int, default=1_000_000, help="shuffle-buffer size in activation vectors")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--tag_batches", type=int, default=500)
    p.add_argument("--norm_batches", type=int, default=200, help="batches used to estimate the input-norm scale")
    p.add_argument("--checkpoint_every", type=int, default=20_000_000,
                   help="save a resumable checkpoint every N activation vectors (for preemption on shared GPUs)")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = tuple(int(g) for g in args.producer_gpus.split(",") if g != "")  # -1 = CPU producer (smoke only)
    json.dump({**vars(args), "d_in": D_IN, "loops": list(LOOPS)}, open(out / "run_args.json", "w"), indent=1)

    def make_provider(shuffle_seed, max_rows):
        return HuginnActivationProvider(
            train_batch_size=args.batch, context_size=args.context_size, n_seeds=args.n_seeds, device=args.device,
            producer_gpus=gpus, buffer_vectors=args.buffer, max_rows=max_rows, shuffle_seed=shuffle_seed,
        )

    # resume: load SAE weights + trainer state from the latest `resume_*` checkpoint if present
    resume_ckpts = sorted(out.glob("resume_*"), key=lambda p: int(p.name.split("_")[1]))
    resume_from = resume_ckpts[-1] if resume_ckpts else None
    if resume_from is not None:
        from sae_lens.saes.sae import TrainingSAE

        sae = TrainingSAE.load_from_disk(str(resume_from), device=args.device)
        print(f"resuming from {resume_from}", flush=True)
    else:
        sae = BatchTopKTrainingSAE(BatchTopKTrainingSAEConfig(
            d_in=D_IN, d_sae=args.d_sae, k=args.k, device=args.device, dtype="float32",
            normalize_activations="expected_average_only_in",  # single-scalar norm; folded into weights at the end
        ))
    provider = make_provider(0, args.max_rows)
    trainer_cfg = SAETrainerConfig(
        total_training_samples=args.total,
        train_batch_size_samples=args.batch,
        lr=args.lr,
        lr_end=args.lr / 10,  # SAETrainerConfig has no post-init default when built directly
        lr_scheduler_name="constant",
        lr_warm_up_steps=max(args.total // args.batch // 20, 1),
        device=args.device,
        n_checkpoints=0,
        checkpoint_path=str(out),
        save_final_checkpoint=True,
        dead_feature_window=2000,
        feature_sampling_window=2000,
        logger=LoggingConfig(log_to_wandb=args.wandb, wandb_project="looped-sae"),
        n_batches_for_norm_estimate=args.norm_batches,
    )
    trainer = SAETrainer(cfg=trainer_cfg, sae=sae, data_provider=provider)
    if resume_from is not None:
        trainer.load_trainer_state(resume_from)
        print(f"resumed at {trainer.n_training_samples} / {args.total} vectors", flush=True)
    run_training_with_checkpoints(trainer, provider, args, out)
    sae = trainer.sae
    n_train_vectors = trainer.n_training_samples
    provider.close()
    # inference export: BatchTopK -> JumpReLU with the learned threshold; load with sae_lens.SAE.load_from_disk
    sae.save_inference_model(str(out / "inference"))

    # held-out tagging stream (different shuffle seed -> different documents first)
    tag_provider = make_provider(999, None if args.max_rows is None else max(args.max_rows // 4, 8))
    tags = tag_features(sae, tag_provider, LOOPS, args.tag_batches)
    np.savez(out / "feature_loop_tags.npz", **tags)

    # held-out sanity: training-mode (BatchTopK) vs exported inference (JumpReLU, learned threshold) L0 / FVU.
    # A large inference-L0 excess means the BatchTopK threshold EMA has not converged.
    from sae_lens import SAE

    inf = SAE.load_from_disk(str(out / "inference"), device=args.device)
    x = next(tag_provider).to(args.device)
    tag_provider.close()
    check = {}
    with torch.no_grad():
        for name, m in (("train_batchtopk", sae), ("inference_jumprelu", inf)):
            f = m.encode(x)
            fvu = ((m.decode(f) - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
            check[name] = {"fvu": fvu.item(), "l0": (f > 0).float().sum(-1).mean().item()}
    print("held-out check:", check)
    json.dump({**vars(args), "d_in": D_IN, "loops": list(LOOPS), "n_train_vectors": n_train_vectors,
               "heldout_check": check}, open(out / "run_args.json", "w"), indent=1)
    freq = tags["freq_by_loop"]
    print("features firing >1% by loop:", {int(l): int((freq[i] > 0.01).sum()) for i, l in enumerate(tags["loops"])})
    print("done ->", out)


if __name__ == "__main__":
    main()
