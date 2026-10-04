from __future__ import annotations

import ast
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

OUTPUT_DIR = Path("output")
PDF_OUT_DIR = Path("/tmp/gap_aware_profile_eval")   # local, avoids iCloud write stalls
THRESHOLD_TAGS = {0.3: "pn30", 0.5: "pn50", 0.7: "pn70", None: ""}
AFFINE = 1  # index of the affine cost in the (classic, affine, log, power) tuples

C_UNW, C_WT = "#1f77b4", "#d62728"   # unweighted / weighted


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

FILE_TYPES = {
    "eval": "_profile_eval.csv",
    "clusters": "_profile_clusters.csv",
    "sens_theta": "_profile_sensitivity_theta.csv",
    "sens_rho": "_profile_sensitivity_rho.csv",
    "sens_clusters": "_profile_sensitivity_clusters.csv",
}


def discover() -> dict[str, dict[str, dict[str, Path]]]:
    """{log: {tag: {file_type: latest_path}}} over all timestamped runs."""
    found: dict = defaultdict(lambda: defaultdict(dict))
    latest_ts: dict = defaultdict(lambda: defaultdict(dict))
    for ts_dir in OUTPUT_DIR.glob("*"):
        if not (ts_dir.is_dir() and ts_dir.name.isdigit()):
            continue
        for log_dir in ts_dir.iterdir():
            if not log_dir.is_dir():
                continue
            for tag in ("pn30", "pn50", "pn70", ""):
                prefix = f"result_{tag}" if tag else "result"
                for ftype, suffix in FILE_TYPES.items():
                    path = log_dir / f"{prefix}{suffix}"
                    if path.exists() and ts_dir.name > latest_ts[log_dir.name][tag].get(ftype, ""):
                        found[log_dir.name][tag][ftype] = path
                        latest_ts[log_dir.name][tag][ftype] = ts_dir.name
    return found


# --------------------------------------------------------------------------- #
# CSV loaders (stdlib only)
# --------------------------------------------------------------------------- #

def load_eval(path: Path):
    """Return dict with weighted-expandable arrays for the affine cost."""
    unw, wt, freq, penal, changed = [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            freq.append(int(row["frequency"]))
            unw.append(ast.literal_eval(row["cost_unweighted"])[AFFINE])
            wt.append(ast.literal_eval(row["cost_weighted"])[AFFINE])
            penal.append(int(row["penalized"]))
            changed.append(int(row["alignment_changed"]))
    return {"unw": np.array(unw, float), "wt": np.array(wt, float),
            "freq": np.array(freq, int), "penal": np.array(penal, int),
            "changed": np.array(changed, int)}


def load_clusters(path: Path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append({"n_variants": int(row["n_variants"]), "frequency": int(row["frequency"]),
                         "reference_length": int(row["reference_length"]),
                         "stable_fraction": float(row["stable_fraction"])})
    return rows


def load_sensitivity(path: Path, axis: str):
    xs, cost, pen = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            xs.append(float(row[axis]))
            cost.append(float(row["mean_cost_weighted"]))
            pen.append(float(row["penalized_share"]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [cost[i] for i in order], [pen[i] for i in order]


def weighted(values, freq):
    return np.repeat(values, freq)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def page_cost_distributions(pdf, log, per_tag):
    tags = [t for t in ("pn30", "pn50", "pn70", "") if "eval" in per_tag.get(t, {})]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 5), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        d = load_eval(per_tag[tag]["eval"])
        series = [(d["unw"], d["freq"]), (d["wt"], d["freq"])]
        box = [weighted(v, f) for v, f in series]
        # Box / whisker / median / mean stay frequency-weighted; the outliers are
        # drawn by us, deduplicated (one marker per distinct value, intensity ~
        # combined frequency) so the plot does not stack thousands of coincident
        # markers like np.repeat would.
        bp = ax.boxplot(box, tick_labels=["unweighted", "weighted"], showmeans=True, meanline=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=1.6),          # median: black, thicker
                        meanprops=dict(color="black", linestyle="--", linewidth=1.0))  # mean: black dashed
        for pos, (vals, freqs) in enumerate(series, start=1):
            lo = bp["whiskers"][2 * (pos - 1)].get_ydata().min()
            hi = bp["whiskers"][2 * (pos - 1) + 1].get_ydata().max()
            agg: dict[float, float] = {}
            for v, fr in zip(vals, freqs):
                if v < lo or v > hi:                         # outliers only
                    agg[round(float(v), 6)] = agg.get(round(float(v), 6), 0.0) + float(fr)
            if not agg:
                continue
            ys = np.array(list(agg))
            fs = np.array([agg[y] for y in agg])
            lf = np.log1p(fs)
            span = float(lf.max() - lf.min())
            norm = (lf - lf.min()) / span if span > 0 else np.ones_like(lf)
            for y, a in zip(ys, 0.50 + 0.45 * norm):
                ax.plot(pos, y, marker="o", markersize=3.5, linestyle="none",
                        color="black", markeredgecolor="none", alpha=float(a), zorder=1)
        share = (d["penal"] * d["freq"]).sum() / d["freq"].sum()
        ax.set_title(f"{tag or 'model'}\npenalized share (freq): {share:.2f}")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Alignment cost")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_scatter(pdf, log, per_tag):
    tags = [t for t in ("pn30", "pn50", "pn70", "") if "eval" in per_tag.get(t, {})]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 4.6), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        d = load_eval(per_tag[tag]["eval"])
        # Collapse variants that share the same (unweighted, weighted) cost into a
        # single marker whose size reflects their combined frequency. Hundreds of
        # coincident points become one -> same appearance, far fewer artists.
        agg: dict[tuple[float, float], float] = {}
        for x, y, fr in zip(d["unw"], d["wt"], d["freq"]):
            key = (round(float(x), 6), round(float(y), 6))
            agg[key] = agg.get(key, 0.0) + float(fr)
        xs = np.array([k[0] for k in agg])
        ys = np.array([k[1] for k in agg])
        fs = np.array([agg[k] for k in agg])
        # A log-normalised frequency in [0, 1] drives both the marker size and the
        # red intensity (rare -> frequent), restoring -- deliberately now -- the
        # density shading that overplotting produced before the dedup. Ranges are
        # chosen so even rare locations stay clearly visible.
        lf = np.log1p(fs)
        span = float(lf.max() - lf.min())
        norm = (lf - lf.min()) / span if span > 0 else np.ones_like(lf)
        sizes = 20 + 55 * norm
        base = mcolors.to_rgb(C_WT)
        colors = np.array([(base[0], base[1], base[2], 0.55 + 0.45 * float(a)) for a in norm])
        ax.scatter(xs, ys, s=sizes, c=colors, edgecolor="white", linewidth=0.4)
        hi = max(1.0, float(max(d["unw"].max(), d["wt"].max())))
        ax.plot([0, hi], [0, hi], "--", color="gray", linewidth=1)
        rho = _spearman(d["unw"], d["wt"])
        moved = (d["changed"] * d["freq"]).sum() / d["freq"].sum()
        ax.set_title(f"{tag or 'model'}\nSpearman={rho:.3f}, alignment changed={moved:.2f}")
        ax.set_xlabel("unweighted cost")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("profile-weighted cost")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_sensitivity(pdf, log, per_tag):
    specs = [("sens_theta", "theta", "Stability threshold θ"),
             ("sens_rho", "rho", "Penalty factor ρ"),
             ("sens_clusters", "n_clusters", "Number of clusters")]
    # one row per threshold that has any sensitivity file
    tags = [t for t in ("pn30", "pn50", "pn70", "")
            if any(k in per_tag.get(t, {}) for k, _, _ in specs)]
    if not tags:
        return
    fig, axes = plt.subplots(len(tags), 3, figsize=(15, 4.2 * len(tags)), squeeze=False)
    for r, tag in enumerate(tags):
        for c, (ftype, axis, xlabel) in enumerate(specs):
            ax = axes[r][c]
            if ftype not in per_tag.get(tag, {}):
                ax.axis("off")
                continue
            xs, cost, pen = load_sensitivity(per_tag[tag][ftype], axis)
            ax.plot(xs, cost, "o-", color=C_WT, label="mean cost")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("mean weighted cost", color=C_WT)
            ax.tick_params(axis="y", labelcolor=C_WT)
            ax2 = ax.twinx()
            ax2.plot(xs, pen, "s--", color=C_UNW, label="penalized share")
            ax2.set_ylabel("penalized share", color=C_UNW)
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", labelcolor=C_UNW)
            ax.set_title(f"{tag or 'model'}")
            ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_clusters(pdf, log, per_tag):
    tags = [t for t in ("pn30", "pn50", "pn70", "") if "clusters" in per_tag.get(t, {})]
    if not tags:
        return
    fig, axes = plt.subplots(2, len(tags), figsize=(4.6 * len(tags), 8), squeeze=False)
    for c, tag in enumerate(tags):
        rows = load_clusters(per_tag[tag]["clusters"])
        # Sort clusters by size (largest first) and apply the SAME order to both
        # rows, so each column refers to the same cluster and the dominant cluster
        # sits on the left. Top row: cluster size; bottom row: stable-position
        # share (so the reader first sees how large a cluster is, then how stable).
        order = sorted(range(len(rows)), key=lambda i: rows[i]["n_variants"], reverse=True)
        sizes = [rows[i]["n_variants"] for i in order]
        stable = [rows[i]["stable_fraction"] for i in order]
        axes[0][c].bar(range(len(sizes)), sizes, color=C_WT)
        axes[0][c].set_title(f"{tag or 'model'}: variants per cluster")
        axes[0][c].set_xlabel("cluster (by size)")
        axes[1][c].bar(range(len(stable)), stable, color=C_UNW)
        axes[1][c].set_title("stable-position share per cluster")
        axes[1][c].set_ylim(0, 1)
        axes[1][c].set_xlabel("cluster (by size)")
    axes[0][0].set_ylabel("#variants")
    axes[1][0].set_ylabel("stable share")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _spearman(x, y):
    n = len(x)
    if n < 2:
        return 1.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 1.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    logs = discover()
    if not logs:
        print("No profile evaluation CSVs found under output/. Run profile_evaluation.py first.")
        return 1
    PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for log, per_tag in logs.items():
        out = PDF_OUT_DIR / f"{log}_profile_evaluation.pdf"
        with PdfPages(out) as pdf:
            page_cost_distributions(pdf, log, per_tag)
            page_scatter(pdf, log, per_tag)
            page_sensitivity(pdf, log, per_tag)
            page_clusters(pdf, log, per_tag)
        print(f"  ✓  {out}")
    print(f"\nDone. PDFs in {PDF_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
