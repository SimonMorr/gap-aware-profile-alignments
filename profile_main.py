from __future__ import annotations

import csv
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
from gap_aware_alignments.profile_alignment import profile_align
from gap_aware_alignments.reachability_graph import ReachabilityGraph

PN_TAGS = ["_pn30", "_pn50", "_pn70"]

# --- profile configuration (tune these) ------------------------------------ #
COST_MODEL = "affine"          # gap cost g used for delta_g / clustering / profiles
STABILITY_THRESHOLD = 0.8      # theta: reference position is stable if P_l(p) >= theta
PENALTY_FACTOR = 2.0           # rho: multiplicative penalty for gaps hitting stable positions
GAP_OPENING_COST = 1.0         # C_open
GAP_EXTENSION_COST = 0.5       # C_extend
GAP_POWER = 0.7                # d
N_CLUSTERS: int | None = None  # None -> data-driven number of clusters via CLUSTER_THRESHOLD
CLUSTER_THRESHOLD: float | None = 4.0  # tau: stop merging when min linkage exceeds this (Def. 5.2)
# Clustering is O(#variants^2) trace-to-trace alignments; cap for large logs.
MAX_VARIANTS = 300


def _params() -> GapParams:
    return GapParams(
        cost_model=COST_MODEL,
        gap_opening_cost=GAP_OPENING_COST,
        gap_extension_cost=GAP_EXTENSION_COST,
        gap_power=GAP_POWER,
        stability_threshold=STABILITY_THRESHOLD,
        penalty_factor=PENALTY_FACTOR,
    )


def run_benchmark(
    variants: list[tuple[str, ...]],
    freqs: list[int],
    model_graph: ReachabilityGraph,
    result_path: Path,
    file_tag: str,
) -> None:
    params = _params()
    print("  Learning profile model (clustering + references + stable positions) ...")
    model = build_profile_model(
        variants, freqs, model_graph, params,
        threshold=CLUSTER_THRESHOLD, n_clusters=N_CLUSTERS,
    )
    print(f"  -> {len(model.clusters)} clusters")

    # Cluster summary
    with open(result_path / f"result{file_tag}_profile_clusters.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cluster", "n_variants", "frequency", "reference_length",
                    "stable_positions", "reference", "profile"])
        for idx, c in enumerate(model.clusters):
            freq = sum(freqs[i] for i in c.indices)
            w.writerow([idx, len(c.indices), freq, len(c.reference),
                        sorted(c.stable), c.reference,
                        [round(x, 3) for x in c.profile[1:]]])

    # Per-variant profile-weighted alignment
    out = result_path / f"result{file_tag}_profile_align.csv"
    total = len(variants)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for done, (variant, freq) in enumerate(zip(variants, freqs), start=1):
            cluster_idx = model.assign(variant)
            costs, aligns = profile_align(variant, model_graph, model.clusters[cluster_idx], params)
            w.writerow([costs, cluster_idx, freq, len(variant), variant,
                        aligns[0]["alignment"], aligns[1]["alignment"],
                        aligns[2]["alignment"], aligns[3]["alignment"]])
            f.flush()
            if done == total or done % 25 == 0:
                print(f"    [{done}/{total}] variants aligned", flush=True)


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
        print(" -> importing event log with Rust4PM ...")
        variants_freqs = get_trace_variants(read_event_log(xes_file))
        if len(variants_freqs) > MAX_VARIANTS:
            variants_freqs = variants_freqs[:MAX_VARIANTS]
        variants = [v for v, _ in variants_freqs]
        freqs = [f for _, f in variants_freqs]
        print(f" -> {len(variants)} trace variants (capped at {MAX_VARIANTS})")

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
            print(f" -> {pnml_file.stem} (Rust4PM PNML import)")
            model_graph = ReachabilityGraph(import_petri_net(pnml_file))
            run_benchmark(variants, freqs, model_graph, cur_path, file_tag)

    print(f"\nDone. Results in {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
