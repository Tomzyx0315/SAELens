"""D1: per-core-layer SAEs within a loop, and the layer x layer transfer matrix.

Huginn's recurrent block has 4 core layers. The loop-state SAE only sees the last layer's output; this trains one
SAE per core layer (at a fixed loop, default 32 = converged) and builds FVU[train layer, eval layer], the layer
analogue of B1. Answers the "different layers" half of the research question: are the 4 core layers one shared
space or distinct spaces (like feedforward layers)?

Producers stream, per document, every core layer's output at --loop, tagged by layer index. Each layer's SAE trains
only on its own layer. Config: BatchTopK, d_sae 16384, k 64, lr 1e-4 + decay (same as the per-loop SAEs).

Usage: python train_perlayer_saes.py --loop 32 --total 60000000 --sae_gpus 0,1 --producer_gpus 2,3,4,5,6,7 --out <dir>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

torch.backends.cuda.matmul.allow_tf32 = True

from scipy.optimize import linear_sum_assignment

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.sae_trainer import SAETrainer

from sae_data import D_IN, iter_token_rows

N_CORE = 4


def producer(rank, gpu, n_prod, queue, cfg):
    dev = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    if gpu >= 0:
        torch.cuda.set_device(gpu)
    from models import get_adapter
    import time as _t

    ad = get_adapter("huginn", dtype=torch.bfloat16, device=dev)
    rng = np.random.default_rng(1000 + rank)
    for row in iter_token_rows(cfg["context_size"], cfg["max_rows"], cfg["shuffle_seed"], rank, n_prod, split=cfg["split"]):
        ids = torch.tensor(row, device=dev).unsqueeze(0)
        for _ in range(240):
            try:
                with torch.no_grad():
                    ls = ad.run_layer_states(ids, cfg["loop"], seed=int(rng.integers(1 << 30)))[:, 1:]  # (n_core, T', d)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                _t.sleep(15)
        else:
            continue
        queue.put(ls.transpose(0, 1).contiguous().cpu())  # (T', n_core, d)
    queue.put(None)


class Stream:
    def __init__(self, gpus, loop, batch, buffer_tokens=300_000, context_size=512, max_rows=None, split="train"):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=64)
        cfg = dict(context_size=context_size, max_rows=None if max_rows is None else -(-max_rows // len(gpus)),
                   shuffle_seed=0, split=split, loop=loop)
        self.procs = [ctx.Process(target=producer, args=(r, g, len(gpus), self.q, cfg), daemon=True)
                      for r, g in enumerate(gpus)]
        for p_ in self.procs:
            p_.start()
        self.alive = len(self.procs)
        self.batch, self.buffer_tokens = batch, buffer_tokens
        self.buf = None
        self.pos = 0

    def _fill(self):
        chunks, n = [], 0
        while n < self.buffer_tokens and self.alive > 0:
            item = self.q.get()
            if item is None:
                self.alive -= 1
                continue
            chunks.append(item)
            n += item.shape[0]
        if not chunks:
            return False
        X = torch.cat(chunks)
        self.buf = X[torch.randperm(X.shape[0])]
        self.pos = 0
        return True

    def __next__(self):
        if self.buf is None or self.pos + self.batch > self.buf.shape[0]:
            if not self._fill() or self.buf.shape[0] < self.batch:
                raise StopIteration
        b = self.buf[self.pos:self.pos + self.batch]
        self.pos += self.batch
        return b  # (batch, n_core, d)

    def close(self):
        for p_ in self.procs:
            p_.terminate()


@torch.no_grad()
def fvu_on(sae, X_layer, dev, chunk=8192):
    sse = tot = 0.0
    mean = X_layer.mean(0, keepdim=True)
    for s in range(0, X_layer.shape[0], chunk):
        x = X_layer[s:s + chunk].to(dev, torch.float32)
        sse += ((sae.decode(sae.encode(x)) - x) ** 2).sum().item()
        tot += ((x - mean.to(dev)) ** 2).sum().item()
    return sse / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--loop", type=int, default=32)
    p.add_argument("--total", type=int, default=60_000_000)
    p.add_argument("--d_sae", type=int, default=16384)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--sae_gpus", default="0,1")
    p.add_argument("--producer_gpus", default="2,3,4,5,6,7")
    p.add_argument("--eval_tokens", type=int, default=120000)
    p.add_argument("--norm_batches", type=int, default=100)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sae_gpus = [int(g) for g in args.sae_gpus.split(",")]
    prod_gpus = [int(g) for g in args.producer_gpus.split(",")]
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    stream = Stream(prod_gpus, args.loop, args.batch)
    steps = args.total // args.batch
    trainers = {}
    for l in range(N_CORE):
        dev = f"cuda:{sae_gpus[l % len(sae_gpus)]}"
        sae = BatchTopKTrainingSAE(BatchTopKTrainingSAEConfig(
            d_in=D_IN, d_sae=args.d_sae, k=args.k, device=dev, normalize_activations="expected_average_only_in"))
        cfg = SAETrainerConfig(total_training_samples=args.total, train_batch_size_samples=args.batch, lr=1e-4,
                               lr_end=1e-5, lr_scheduler_name="constant", lr_warm_up_steps=0,
                               lr_decay_steps=int(0.2 * steps), device=dev, logger=LoggingConfig(log_to_wandb=False),
                               n_batches_for_norm_estimate=args.norm_batches)
        trainers[l] = SAETrainer(cfg=cfg, sae=sae, data_provider=iter(()))
        print(f"core layer {l}: SAE on {dev}", flush=True)

    warm = [next(stream) for _ in range(args.norm_batches)]
    for l in range(N_CORE):
        trainers[l].activation_scaler.estimate_scaling_factor(
            d_in=D_IN, data_provider=iter([b[:, l] for b in warm]), n_batches_for_norm_estimate=len(warm))
    del warm
    print("norm done", flush=True)

    t0 = time.time()
    done = 0
    while done < args.total:
        try:
            b = next(stream)
        except StopIteration:
            print("data exhausted", flush=True)
            break
        for l in range(N_CORE):
            trainers[l].maybe_reset_sparsity()
            trainers[l].step(b[:, l])
            trainers[l].n_training_steps += 1
        done += args.batch
        if done % (args.batch * 500) < args.batch:
            print(f"done {done}/{args.total}  {done / (time.time() - t0):.0f} vec/s per layer", flush=True)
    stream.close()

    for l in range(N_CORE):
        t = trainers[l]
        if t.activation_scaler.scaling_factor is not None:
            t.sae.fold_activation_norm_scaling_factor(t.activation_scaler.scaling_factor)
            t.activation_scaler.scaling_factor = None
        t.sae.eval()
        t.save_checkpoint(checkpoint_name=f"layer{l}", base_path=str(out))

    # eval cache: collect layer states on held-out
    ev = Stream([prod_gpus[0]], args.loop, args.batch, buffer_tokens=args.eval_tokens, split="heldout", max_rows=300)
    X = torch.cat([next(ev) for _ in range(max(1, args.eval_tokens // args.batch))])  # (N, n_core, d)
    ev.close()
    ident = ActivationScaler()
    T = np.zeros((N_CORE, N_CORE))
    for i in range(N_CORE):
        sae = trainers[i].sae
        for j in range(N_CORE):
            T[i, j] = fvu_on(sae, X[:, j], sae.device)
        print(f"layer{i} SAE transfer FVU: " + " ".join(f"{j}:{T[i,j]:.3f}" for j in range(N_CORE)), flush=True)
    match = {}
    for a in range(N_CORE - 1):
        Wa = torch.nn.functional.normalize(trainers[a].sae.W_dec.detach().float(), dim=-1).cpu().numpy()
        Wb = torch.nn.functional.normalize(trainers[a + 1].sae.W_dec.detach().float(), dim=-1).cpu().numpy()
        ri, ci = linear_sum_assignment(-(Wa @ Wb.T))
        cs = (Wa @ Wb.T)[ri, ci]
        match[f"{a}->{a+1}"] = {"mean_cos": round(float(cs.mean()), 3), "frac>0.8": round(float((cs > 0.8).mean()), 3)}
    res = {"loop": args.loop, "n_core": N_CORE, "transfer_fvu": T.round(4).tolist(),
           "diag_fvu": [round(float(T[i, i]), 4) for i in range(N_CORE)], "decoder_match": match}
    json.dump(res, open(Path(__file__).resolve().parent.parent / "results/sae/perlayer_transfer.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
