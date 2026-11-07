"""B2: cross-loop crosscoder on Huginn loop states.

One set of latents shared across loops L = (1, 2, 4, 32); each loop l has its own encoder W_enc[l] (d_in x m) and
decoder W_dec[l] (m x d_in). For a token, the inputs are its states at all loops (same token, same init seed):

    pre      = sum_l x_l W_enc[l] + b_enc                       (m)
    f        = BatchTopK(ReLU(pre))                             (m), on average k active per token
    x_hat_l  = f W_dec[l] + b_dec[l]

Loss = sum_l ||x_l - x_hat_l||^2 / sum_l ||x_l - mean||^2 + aux (dead-latent revival, as in TopK SAEs).
With the BatchTopK sparsity there is no L1 term, so the per-loop decoder norms ||W_dec[l][j]|| are free and
informative: a latent with norm mass only at loop 1 is an early feature, only at loop 32 a late feature, at all
loops a shared feature. This is the Anthropic "sparse crosscoder" setup applied across loops instead of layers.
Each loop's input is normalised to unit mean squared norm (scale estimated once), so loops contribute equally.

Data: same producers as the SAE (sae_data), restricted to the crosscoder loops; tokens are grouped so that one
training row = one token's states at all loops (same document, position and seed).

Outputs (out_dir): weights, per-loop decoder norms, relative decoder norm per latent, per-loop held-out FVU.
Usage: python crosscoder.py --out <dir> --device cuda:0 --producer_gpus 1,2,3,4,5,6,7
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

torch.backends.cuda.matmul.allow_tf32 = True

from sae_data import D_IN, iter_token_rows  # noqa: E402

CC_LOOPS = (1, 2, 4, 32)


def cc_producer(rank, gpu, n_prod, queue, cfg):
    """Push (n_tok, n_loops, d) bf16 tensors: one row per token with its states at the crosscoder loops."""
    import time as _t

    dev = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    if gpu >= 0:
        torch.cuda.set_device(gpu)
    from models import get_adapter

    ad = get_adapter("huginn", dtype=torch.bfloat16, device=dev)
    rng = np.random.default_rng(1000 + rank)
    for row in iter_token_rows(cfg["context_size"], cfg["max_rows"], cfg["shuffle_seed"], rank, n_prod, split=cfg["split"]):
        ids = torch.tensor(row, device=dev).unsqueeze(0)
        for _ in range(240):
            try:
                with torch.no_grad():
                    st = ad.run_states(ids, max(cfg["loops"]), seed=int(rng.integers(1 << 30)))[:, 1:]
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                _t.sleep(15)
        else:
            continue
        x = st[[l - 1 for l in cfg["loops"]]].transpose(0, 1).contiguous()  # (T, n_loops, d)
        queue.put(x.cpu())
    queue.put(None)


class Stream:
    def __init__(self, gpus, loops, batch, buffer_tokens=400_000, context_size=512, max_rows=None, split="train",
                 shuffle_seed=0):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=64)
        cfg = dict(context_size=context_size, max_rows=None if max_rows is None else -(-max_rows // len(gpus)),
                   shuffle_seed=shuffle_seed, split=split, loops=loops)
        self.procs = [ctx.Process(target=cc_producer, args=(r, g, len(gpus), self.q, cfg), daemon=True)
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
        return b

    def close(self):
        for p_ in self.procs:
            p_.terminate()


class Crosscoder(torch.nn.Module):
    def __init__(self, n_loops, d_in, m, k):
        super().__init__()
        self.k = k
        self.W_enc = torch.nn.Parameter(torch.empty(n_loops, d_in, m))
        self.W_dec = torch.nn.Parameter(torch.empty(n_loops, m, d_in))
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(torch.zeros(n_loops, d_in))
        torch.nn.init.normal_(self.W_dec, std=1.0)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=-1, keepdim=True) * np.sqrt(n_loops) / 0.1  # small total decoder norm
            self.W_enc.copy_(self.W_dec.transpose(1, 2))
        self.register_buffer("threshold", torch.tensor(0.0))

    def encode_pre(self, x):  # x (B, n_loops, d)
        return torch.einsum("bld,ldm->bm", x - self.b_dec, self.W_enc) + self.b_enc

    def decode(self, f):
        return torch.einsum("bm,lmd->bld", f, self.W_dec) + self.b_dec

    def dec_norm(self):
        return self.W_dec.norm(dim=-1)  # (n_loops, m)

    def forward(self, x, train=True):
        pre = torch.relu(self.encode_pre(x))
        # rank latents by contribution = activation * total decoder norm (standard for crosscoders)
        score = pre * self.dec_norm().sum(0)
        if train:
            n = int(self.k * x.shape[0])
            thr = score.flatten().topk(n).values[-1]
            mask = score >= thr
            with torch.no_grad():
                pos = score[mask]
                if pos.numel():
                    self.threshold.mul_(0.99).add_(0.01 * pos.min())
        else:
            mask = score > self.threshold
        f = pre * mask
        return f, pre


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--producer_gpus", default="1,2,3,4,5,6,7")
    p.add_argument("--m", type=int, default=16384)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--total_tokens", type=int, default=25_000_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--buffer_tokens", type=int, default=400_000)
    p.add_argument("--eval_tokens", type=int, default=40_000)
    p.add_argument("--warm_batches", type=int, default=20, help="batches used to estimate the per-loop scale")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(args), "loops": CC_LOOPS}, open(out / "args.json", "w"), indent=1)
    gpus = [int(g) for g in args.producer_gpus.split(",")]
    dev = args.device
    L = len(CC_LOOPS)

    stream = Stream(gpus, CC_LOOPS, args.batch, args.buffer_tokens, max_rows=args.max_rows)
    # per-loop scale: unit mean squared norm
    warm = torch.cat([next(stream) for _ in range(args.warm_batches)]).float()
    scale = (warm.pow(2).sum(-1).mean(0).sqrt() / np.sqrt(D_IN)).to(dev)  # (L,)
    del warm
    cc = Crosscoder(L, D_IN, args.m, args.k).to(dev)
    opt = torch.optim.Adam(cc.parameters(), lr=args.lr, betas=(0.9, 0.999))
    steps = args.total_tokens // args.batch
    decay_from = int(0.8 * steps)
    since_fired = torch.zeros(args.m, device=dev)
    t0 = time.time()
    log = []
    for step in range(steps):
        try:
            x = next(stream).to(dev, torch.float32) / scale[None, :, None]
        except StopIteration:
            print("data exhausted", flush=True)
            break
        lr = args.lr * (1.0 if step < decay_from else max(0.0, (steps - step) / (steps - decay_from)))
        for g in opt.param_groups:
            g["lr"] = lr
        f, pre = cc(x, train=True)
        xh = cc.decode(f)
        var = (x - x.mean(0, keepdim=True)).pow(2).sum()
        mse = (xh - x).pow(2).sum() / var
        # aux loss: let dead latents (no firing for 1000 steps) reconstruct the residual, top-k_aux of them
        fired = (f > 0).any(0)
        since_fired = torch.where(fired, torch.zeros_like(since_fired), since_fired + 1)
        dead = since_fired > 1000
        aux = torch.tensor(0.0, device=dev)
        if dead.sum() > 0:
            k_aux = min(int(dead.sum()), D_IN // 2)
            pre_dead = torch.where(dead[None, :], pre, torch.zeros_like(pre))
            vals, idx = pre_dead.topk(k_aux, dim=-1)
            f_aux = torch.zeros_like(pre).scatter(-1, idx, vals)
            resid = (x - xh).detach()
            aux = (torch.einsum("bm,lmd->bld", f_aux, cc.W_dec) - resid).pow(2).sum() / var
        loss = mse + 1 / 32 * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cc.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            n_dead = int(dead.sum())
            print(f"step {step}/{steps} fvu {mse.item():.4f} aux {aux.item():.4f} dead {n_dead} "
                  f"{(step + 1) * args.batch / (time.time() - t0):.0f} tok/s", flush=True)
            log.append({"step": step, "fvu": mse.item(), "dead": n_dead})
    stream.close()

    torch.save({"state_dict": cc.state_dict(), "scale": scale.cpu(), "loops": CC_LOOPS}, out / "crosscoder.pt")

    # held-out eval: per-loop FVU, using the inference threshold
    ev = Stream([gpus[0]], CC_LOOPS, args.batch, buffer_tokens=args.eval_tokens, split="heldout", shuffle_seed=77,
                max_rows=200)
    X = torch.cat([next(ev) for _ in range(max(1, args.eval_tokens // args.batch))]).to(dev, torch.float32)
    ev.close()
    X = X / scale[None, :, None]
    with torch.no_grad():
        l0 = 0.0
        num = torch.zeros(L, device=dev)
        den = torch.zeros(L, device=dev)
        fire = torch.zeros(args.m, device=dev)
        for c in X.split(4096):
            f, _ = cc(c, train=False)
            xh = cc.decode(f)
            num += (xh - c).pow(2).sum((0, 2))
            den += (c - X.mean(0, keepdim=True)).pow(2).sum((0, 2))
            l0 += (f > 0).float().sum().item()
            fire += (f > 0).float().sum(0)
        fvu_loop = (num / den).cpu().numpy()
        freq = (fire / X.shape[0]).cpu().numpy()
        dn = cc.dec_norm().detach()  # (L, m)
        rel = (dn / dn.sum(0, keepdim=True).clamp(min=1e-8)).cpu().numpy()
    np.save(out / "dec_norm.npy", dn.cpu().numpy())
    np.save(out / "heldout_freq.npy", freq)
    res = {"loops": list(CC_LOOPS), "heldout_fvu_by_loop": dict(zip(map(str, CC_LOOPS), fvu_loop.round(4).tolist())),
           "heldout_l0": l0 / X.shape[0], "heldout_tokens": int(X.shape[0]), "train_log": log}
    # latent classes from relative decoder norm (share of decoder norm at each loop), among live latents
    alive = freq > 1e-4
    first, last = rel[0], rel[-1]
    res["latents_alive"] = int(alive.sum())
    res["share_mostly_loop1(>0.7)"] = int(((first > 0.7) & alive).sum())
    res["share_mostly_loop32(>0.7)"] = int(((last > 0.7) & alive).sum())
    res["shared(all loops 0.15-0.35)"] = int((((rel > 0.15) & (rel < 0.35)).all(0) & alive).sum())
    res["rel_norm_hist_loop1"] = np.histogram(first[alive], bins=10, range=(0, 1))[0].tolist()
    res["rel_norm_hist_loop32"] = np.histogram(last[alive], bins=10, range=(0, 1))[0].tolist()
    json.dump(res, open(out / "results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "train_log"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
