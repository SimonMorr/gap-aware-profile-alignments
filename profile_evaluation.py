from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

from gap_aware_alignments import (
    GapParams,
    build_profile_model,
    get_trace_variants,
    import_petri_net,
    read_event_log,
)
from gap_aware_alignments.gap_aware_alignment import align
from gap_aware_alignments.profile_alignment import profile_align
from gap_aware_alignments.reachability_graph import ReachabilityGraph

PN_TAGS = ["_pn30", "_pn50", "_pn70"]

# --- configuration (kept in sync with profile_main.py) --------------------- #
COST_MODEL = "affine"          # reported cost model (index into the 4-tuples)
STABILITY_THRESHOLD = 0.8      # theta
PENALTY_FACTOR = 2.0           # rho
GAP_OPENING_COST = 1.0
GAP_EXTENSION_COST = 0.5
GAP_POWER = 0.7
N_CLUSTERS: int | None = None      # None -> data-driven number of clusters via CLUSTER_THRESHOLD
CLUSTER_THRESHOLD: float | None = 4.0   # tau: cut the dendrogram where linkage exceeds this (Def. 5.2)
MAX_VARIANTS = 300

COST_LABELS = ["classic", "affine", "log", "power"]


def gap_cost(cost_model: str, n: float,
             c_open: float = GAP_OPENING_COST,
             c_ext: float = GAP_EXTENSION_COST,
             d: float = GAP_POWER) -> float:
    """Gap cost ``g(n)`` for a single gap of length ``n`` (matches the models in
    :mod:`gap_aware_alignments.gap_aware_alignment`)."""
    if cost_model == "classic":
        return n
    if cost_model == "affine":
        return c_open + c_ext * n
    if cost_model == "log":
        return c_open + math.log(n + 1) * c_ext
    if cost_model == "power":
        return c_open + c_ext * (n ** d)
    raise ValueError(f"Unknown cost model: {cost_model}")


def profile_fitness(weighted_cost: float, trace_length: int, best_worst_cost: int,
                    has_stable: bool, cost_model: str, penalty_factor: float) -> float:
    """Profile-weighted fitness ``1 - C^P_g / g_P(|sigma| + |p_min|)``.

    The reference (worst-case) alignment is a single gap spanning the whole
    reference, so its footprint hits every stable position and its cost is
    ``rho * g(|sigma| + |p_min|)`` whenever the assigned cluster has a stable
    position (``rho`` otherwise drops out). Clamped to ``[0, 1]``.
    """
    ref_len = trace_length + best_worst_cost
    g_ref = gap_cost(cost_model, ref_len)
    if g_ref <= 0:
        return 1.0
    norm = (penalty_factor if has_stable else 1.0) * g_ref
    return min(1.0, max(0.0, 1.0 - weighted_cost / norm))


def _params() -> GapParams:
    return GapParams(
        cost_model=COST_MODEL,
        gap_opening_cost=GAP_OPENING_COST,
        gap_extension_cost=GAP_EXTENSION_COST,
        gap_power=GAP_POWER,
        stability_threshold=STABILITY_THRESHOLD,
        penalty_factor=PENALTY_FACTOR,
    )


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties (no SciPy)."""
    n = len(x)
    if n < 2:
        return 1.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 1.0


def evaluate_benchmark(
    variants: list[tuple[str, ...]],
    freqs: list[int],
    model_graph: ReachabilityGraph,
    result_path: Path,
    file_tag: str,
) -> None:
    params = _params()
    ci = params.cost_index
    print("  Learning profile model ...")
    model = build_profile_model(
        variants, freqs, model_graph, params,
        threshold=CLUSTER_THRESHOLD, n_clusters=N_CLUSTERS,
    )
    print(f"  -> {len(model.clusters)} clusters")

    # ---- cluster diagnostics (Part 4) ----
    with open(result_path / f"result{file_tag}_profile_clusters.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cluster", "n_variants", "frequency", "reference_length",
                    "n_stable", "stable_fraction", "stable_positions", "reference"])
        for idx, c in enumerate(model.clusters):
            freq = sum(freqs[i] for i in c.indices)
            h = len(c.reference)
            w.writerow([idx, len(c.indices), freq, h, len(c.stable),
                        round(len(c.stable) / h, 3) if h else 0.0,
                        sorted(c.stable), c.reference])

    # ---- per-variant weighted vs. unweighted (Parts 1 + 2) ----
    bw = model_graph.best_worst_cost    # |p_min|: shortest accepting model run
    unw_cost, wt_cost, wt_fitness = [], [], []   # affine cost / fitness per variant
    penalized_freq = changed_freq = total_freq = 0
    rows = []
    for variant, freq in zip(variants, freqs):
        assigned = model.assign(variant)
        cluster = model.clusters[assigned]
        u_costs, u_aligns = align(variant, model_graph, GAP_OPENING_COST, GAP_EXTENSION_COST, GAP_POWER)
        p_costs, p_aligns = profile_align(variant, model_graph, cluster, params)
        penalized = p_costs[ci] > u_costs[ci] + 1e-9
        changed = u_aligns[ci]["alignment"] != p_aligns[ci]["alignment"]
        fit_p = profile_fitness(p_costs[ci], len(variant), bw, bool(cluster.stable),
                                COST_MODEL, PENALTY_FACTOR)
        unw_cost.append(u_costs[ci])
        wt_cost.append(p_costs[ci])
        wt_fitness.append(fit_p)
        total_freq += freq
        penalized_freq += freq if penalized else 0
        changed_freq += freq if changed else 0
        rows.append([assigned, freq, len(variant), variant,
                     tuple(round(c, 4) for c in u_costs), tuple(round(c, 4) for c in p_costs),
                     int(penalized), int(changed), round(fit_p, 4)])

    with open(result_path / f"result{file_tag}_profile_eval.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cluster", "frequency", "length", "variant",
                    "cost_unweighted", "cost_weighted", "penalized", "alignment_changed",
                    "fitness_weighted"])
        w.writerows(rows)

    # ---- aggregate summary ----
    def wmean(values):
        return sum(v * fr for v, fr in zip(values, freqs)) / total_freq if total_freq else 0.0

    summary = {
        "cost_model": COST_MODEL,
        "n_variants": len(variants),
        "weighted_traces": total_freq,
        "n_clusters": len(model.clusters),
        "mean_reference_length": round(sum(len(c.reference) for c in model.clusters) / len(model.clusters), 2),
        "mean_stable_fraction": round(sum(
            len(c.stable) / len(c.reference) for c in model.clusters if c.reference) / len(model.clusters), 3),
        "penalized_share": round(penalized_freq / total_freq, 4) if total_freq else 0.0,
        "alignment_changed_share": round(changed_freq / total_freq, 4) if total_freq else 0.0,
        "mean_cost_unweighted": round(wmean(unw_cost), 4),
        "mean_cost_weighted": round(wmean(wt_cost), 4),
        "mean_fitness_weighted": round(wmean(wt_fitness), 4),
        "spearman_unw_vs_wt": round(_spearman(unw_cost, wt_cost), 4),
    }
    with open(result_path / f"result{file_tag}_profile_summary.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(list(summary.keys()))
        w.writerow(list(summary.values()))

    print(f"  penalized share (freq): {summary['penalized_share']:.3f} | "
          f"alignment changed: {summary['alignment_changed_share']:.3f} | "
          f"mean cost {summary['mean_cost_unweighted']} -> {summary['mean_cost_weighted']} | "
          f"fitness^P {summary['mean_fitness_weighted']} | "
          f"Spearman {summary['spearman_unw_vs_wt']}")


def main() -> int:
    data_path = Path("data").resolve()
    result_path = Path("output") / datetime.now().strftime("%Y%m%d%H%M%S")
    result_path.mkdir(parents=True)

    for xes_file in sorted(data_path.glob("*.xes")):
        if not xes_file.is_file():
            continue
        cur_path = result_path / xes_file.stem
        cur_path.mkdir()
        print(f"{xes_file.stem}")
        variants_freqs = get_trace_variants(read_event_log(xes_file))[:MAX_VARIANTS]
        variants = [v for v, _ in variants_freqs]
        freqs = [fr for _, fr in variants_freqs]
        print(f" -> {len(variants)} trace variants")

        benchmarks: list[tuple[str, Path]] = []
        if (pnml_file := data_path / f"{xes_file.stem}.pnml").is_file():
            benchmarks.append(("", pnml_file))
        else:
            for tag in PN_TAGS:
                pnml_file = data_path / f"{xes_file.stem}{tag}.pnml"
                if not pnml_file.is_file():
                    raise FileNotFoundError(f"Expected Petri net not found: {pnml_file}")
                benchmarks.append((tag, pnml_file))

        for file_tag, pnml_file in benchmarks:
            print(f" -> {pnml_file.stem}")
            model_graph = ReachabilityGraph(import_petri_net(pnml_file))
            evaluate_benchmark(variants, freqs, model_graph, cur_path, file_tag)

    print(f"\nDone. Results in {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
