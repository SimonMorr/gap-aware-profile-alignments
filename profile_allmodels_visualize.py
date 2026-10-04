from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

OUTPUT_DIR = Path("output")
PDF_OUT_DIR = Path("/tmp/gap_aware_profile_eval")

# CSV model key -> (display label, colour)
MODELS = {
    "classic": ("Linear", "#1f77b4"),
    "affine": ("Affine", "#ff7f0e"),
    "log": ("Logarithmic", "#2ca02c"),
    "power": ("Power", "#d62728"),
}
MODEL_ORDER = ["classic", "affine", "log", "power"]
THRESHOLD_ORDER = ["_pn30", "_pn50", "_pn70", "single"]
THRESHOLD_LABELS = {"_pn30": "pn30", "_pn50": "pn50", "_pn70": "pn70", "single": "model"}


def find_summary() -> Path | None:
    candidates = sorted(OUTPUT_DIR.glob("*/profile_allmodels_summary.csv"),
                        key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def load(path: Path) -> dict:
    """{log: {threshold: {model: row}}}."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            data[row["log"]][row["threshold"]][row["cost_model"]] = row
    return data


def _grouped(ax, thresholds, per_thr, value_key, ylabel, ylim=None):
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(thresholds))
    for i, m in enumerate(MODEL_ORDER):
        label, color = MODELS[m]
        vals = [float(per_thr[t][m][value_key]) if m in per_thr[t] else 0.0 for t in thresholds]
        ax.bar(x + (i - (n_models - 1) / 2) * width, vals, width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([THRESHOLD_LABELS.get(t, t) for t in thresholds])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)


def make_figure(pdf, log, per_thr):
    thresholds = [t for t in THRESHOLD_ORDER if t in per_thr]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    _grouped(ax1, thresholds, per_thr, "penalized_share",
             "Penalized share (freq)", ylim=(0, 1))
    ax1.set_title("Share of cases receiving a penalty")
    ax1.legend(fontsize=8, ncol=4, loc="upper center")

    _grouped(ax2, thresholds, per_thr, "mean_cost_weighted", "Mean cost")
    # mark the unweighted mean as a short black tick on each bar
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(thresholds))
    for i, m in enumerate(MODEL_ORDER):
        for j, t in enumerate(thresholds):
            if m in per_thr[t]:
                u = float(per_thr[t][m]["mean_cost_unweighted"])
                xpos = x[j] + (i - (n_models - 1) / 2) * width
                ax2.plot([xpos - width / 2, xpos + width / 2], [u, u], color="black", linewidth=1.2)
    ax2.set_title("Mean profile-weighted cost (black tick = unweighted mean)")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    summary = find_summary()
    if summary is None:
        print("No profile_allmodels_summary.csv found under output/. Run profile_allmodels.py first.")
        return 1
    print(f"Using {summary}")
    data = load(summary)
    PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for log, per_thr in data.items():
        out = PDF_OUT_DIR / f"{log}_allmodels_comparison.pdf"
        with PdfPages(out) as pdf:
            make_figure(pdf, log, per_thr)
        print(f"  ✓  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
