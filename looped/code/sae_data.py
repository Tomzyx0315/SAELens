"""Online activation provider for training a shared cross-loop SAE on Huginn-0125 (method 2: no activations on disk).

Producers (one per GPU, separate processes) stream Huginn pretraining-validation documents, run the model once per
document for the configured number of loops and init seeds, and push activation chunks tagged with (loop, seed)
into a shared queue. The consumer (SAE trainer) pops chunks into a shuffle buffer and yields batches.

Design choices (also recorded in looped/DATA.md):

- Site: `loop_state` = output of the last core-block layer per loop (the latent carried to the next loop), d_in 5280.
- Loops sampled log-uniformly from LOOPS so early loops (where the active set turns over) are not drowned out by
  the near-identical converged loops.
- Each document is run with `n_seeds` random initial states; every (loop, seed) copy is emitted and tagged so that
  later analysis can tell init-noise features from genuine computation.
- Producers shard documents disjointly (row index mod n_producers), so no document is seen twice.
- The shuffle buffer mixes activations across documents, loops and seeds before batching.
"""

import glob
import random
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp

HUGINN_VAL = "/data_fs/zhangyuxuan/looped/data/huginn_val/validation"
LOOPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)  # log-spaced within train depth (mean 32)
D_IN = 5280
# One shard is reserved for SAE evaluation and never read during SAE training.
HELDOUT_SHARD = "shard_1000_of_4096.parquet"


def iter_token_rows(context_size: int, max_rows: int | None, seed: int, worker: int = 0,
                    n_workers: int = 1, split: str = "train") -> Iterator[list[int]]:
    """Yield Huginn-tokenized rows (length 4097, BOS-prefixed) truncated to context_size.

    split="train": all shards except HELDOUT_SHARD; split="heldout": only HELDOUT_SHARD.
    Shards are visited in a seeded shuffled order; rows within a shard in a seeded shuffled order. Rows are
    split disjointly across workers by a global counter.
    """
    all_shards = sorted(glob.glob(f"{HUGINN_VAL}/*.parquet"))
    if split == "heldout":
        shards = [s for s in all_shards if s.endswith(HELDOUT_SHARD)]
    else:
        shards = [s for s in all_shards if not s.endswith(HELDOUT_SHARD)]
    assert shards, f"no shards for split={split}"
    rng = random.Random(seed)
    rng.shuffle(shards)
    n_seen, n_yield = 0, 0
    for shard in shards:
        col = pq.read_table(shard, columns=["input_ids"]).column("input_ids")
        order = list(range(len(col)))
        rng.shuffle(order)
        for i in order:
            if n_seen % n_workers == worker:
                yield col[i].as_py()[:context_size]
                n_yield += 1
                if max_rows is not None and n_yield >= max_rows:
                    return
            n_seen += 1


@torch.no_grad()
def doc_activations(ad, ids, loops, n_seeds, rng, drop_bos=True):
    """(n_vec, D_IN) bf16 activations and (n_vec, 2) int16 [loop, seed] tags for one document."""
    t0 = 1 if drop_bos else 0
    max_loops = max(loops)
    states, tags = [], []
    for s in range(n_seeds):
        st = ad.run_states(ids, max_loops, seed=int(rng.integers(1 << 30)))[:, t0:]  # (max_loops, T', d) bf16
        sel = st[[l - 1 for l in loops]].reshape(-1, st.shape[-1])  # loop-major
        states.append(sel)
        lt = np.repeat(np.array(loops, dtype=np.int16), st.shape[1])
        tags.append(np.stack([lt, np.full_like(lt, s)], 1))
        del st
    return torch.cat(states, 0), np.concatenate(tags, 0)


def producer(rank, gpu, n_producers, queue, cfg):
    """Run Huginn on physical GPU `gpu` (or CPU if gpu < 0), push (cpu bf16 activations, tags) chunks into `queue`."""
    import time

    dev = f"cuda:{gpu}" if gpu >= 0 else "cpu"  # explicit index; CUDA_VISIBLE_DEVICES is unreliable in spawn children
    if gpu >= 0:
        torch.cuda.set_device(gpu)
    else:
        torch.set_num_threads(8)
    from models import get_adapter

    ad = get_adapter("huginn", dtype=torch.bfloat16, device=dev)
    rng = np.random.default_rng(cfg["shuffle_seed"] * 1000 + rank)
    rows = iter_token_rows(cfg["context_size"], cfg["max_rows_per_producer"], cfg["shuffle_seed"], rank, n_producers,
                           split=cfg["split"])
    for row in rows:
        ids = torch.tensor(row, device=dev).unsqueeze(0)
        for attempt in range(240):  # GPUs are shared with other jobs; retry transient OOM (e.g. sglang KV spikes)
            try:
                X, tg = doc_activations(ad, ids, cfg["loops"], cfg["n_seeds"], rng)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                time.sleep(15)
        else:
            continue  # give up on this document, move on
        queue.put((X.cpu(), tg))
    queue.put(None)


class HuginnActivationProvider:
    """Iterator over (train_batch_size, D_IN) fp32 activation batches on `device`.

    `producer_gpus`: GPUs running Huginn (one process each). If empty, runs Huginn in-process on `device`
    (single-GPU / CPU smoke mode). The tags of the most recent batch are in `.last_tags` ((n, 2): loop, seed).
    """

    def __init__(
        self,
        train_batch_size: int = 4096,
        context_size: int = 512,
        loops: tuple[int, ...] = LOOPS,
        n_seeds: int = 2,
        device: str = "cuda",
        producer_gpus: tuple[int, ...] = (),
        buffer_vectors: int = 1_000_000,
        max_rows: int | None = None,
        shuffle_seed: int = 0,
        split: str = "train",
    ):
        self.bs = train_batch_size
        self.device = device
        self.loops = loops
        self.n_seeds = n_seeds
        self.buffer_vectors = buffer_vectors
        self.rng = np.random.default_rng(shuffle_seed)
        self.n_emitted = 0
        self.last_tags = np.empty((0, 2), dtype=np.int16)
        self._chunks: list[tuple[torch.Tensor, np.ndarray]] = []
        self._n_buffered = 0
        self._batch_x: torch.Tensor | None = None
        self._batch_t: np.ndarray | None = None
        self._pos = 0
        self._exhausted = False
        n_prod = max(len(producer_gpus), 1)
        cfg = dict(context_size=context_size, loops=loops, n_seeds=n_seeds, shuffle_seed=shuffle_seed, split=split,
                   max_rows_per_producer=None if max_rows is None else -(-max_rows // n_prod))
        if producer_gpus:
            ctx = mp.get_context("spawn")
            self.queue = ctx.Queue(maxsize=64)
            self.procs = [ctx.Process(target=producer, args=(r, g, len(producer_gpus), self.queue, cfg), daemon=True)
                          for r, g in enumerate(producer_gpus)]
            for pr in self.procs:
                pr.start()
            self._alive = len(self.procs)
            self._local = None
        else:
            from models import get_adapter

            self.procs, self.queue = [], None
            self._local = (get_adapter("huginn", dtype=torch.bfloat16, device=device),
                           iter_token_rows(context_size, cfg["max_rows_per_producer"], shuffle_seed, split=split),
                           np.random.default_rng(shuffle_seed * 1000))

    def _get_chunk(self):
        if self._local is not None:
            ad, rows, rng = self._local
            try:
                row = next(rows)
            except StopIteration:
                return None
            ids = torch.tensor(row, device=self.device).unsqueeze(0)
            X, tg = doc_activations(ad, ids, self.loops, self.n_seeds, rng)
            return X.cpu(), tg
        while self._alive > 0:
            item = self.queue.get()
            if item is None:
                self._alive -= 1
                continue
            return item
        return None

    def _fill(self):
        """Fill the shuffle buffer to `buffer_vectors`, then shuffle it into the current batch source."""
        while self._n_buffered < self.buffer_vectors and not self._exhausted:
            item = self._get_chunk()
            if item is None:
                self._exhausted = True
                break
            self._chunks.append(item)
            self._n_buffered += item[0].shape[0]
        if not self._chunks:
            return False
        X = torch.cat([c[0] for c in self._chunks], 0)
        T = np.concatenate([c[1] for c in self._chunks], 0)
        perm = torch.from_numpy(self.rng.permutation(X.shape[0]))
        # keep half of the shuffled buffer for the next round (mixes across refills), emit the other half
        n_emit = X.shape[0] if self._exhausted else X.shape[0] // 2
        emit, keep = perm[:n_emit], perm[n_emit:]
        self._batch_x, self._batch_t = X[emit], T[emit.numpy()]
        self._chunks = [(X[keep], T[keep.numpy()])] if len(keep) else []
        self._n_buffered = len(keep)
        self._pos = 0
        return True

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self._batch_x is None or self._pos + self.bs > self._batch_x.shape[0]:
            if not self._fill():
                raise StopIteration
            if self._batch_x.shape[0] < self.bs:
                raise StopIteration
        s = slice(self._pos, self._pos + self.bs)
        self._pos += self.bs
        self.last_tags = self._batch_t[s]
        self.n_emitted += self.bs
        return self._batch_x[s].to(self.device, dtype=torch.float32, non_blocking=True)

    def close(self):
        for pr in self.procs:
            pr.terminate()
