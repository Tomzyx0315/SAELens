"""Path layout: large artifacts (weights, datasets, raw per-example activations) on /data_fs,
everything small (code, figures, summaries, logs, docs) inside the repo under looped/."""

from pathlib import Path

# large storage
BIG = Path("/data_fs/zhangyuxuan/looped")
WEIGHTS = {
    "ouro": BIG / "Ouro-1.4B",
    "huginn": BIG / "huginn-0125",
    "llama": BIG / "Llama-3.2-1B",
}
DATA = BIG / "data"
RAW_ROOT = BIG / "results"  # per-example npz files: RAW_ROOT/<model>/<run>/ex*.npz

# small outputs (in repo)
HERE = Path(__file__).resolve().parent.parent  # .../SAELens/looped
RESULTS_ROOT = HERE / "results"  # RESULTS_ROOT/<model>/{figs,summary.json,...}, RESULTS_ROOT/compare
LOGS = RESULTS_ROOT / "logs"


def raw_dir(model: str) -> Path:
    return RAW_ROOT / model


def results_dir(model: str) -> Path:
    return RESULTS_ROOT / model
