from __future__ import annotations

import copy
import csv
import sys
from datetime import datetime
from pathlib import Path

from gap_aware_alignments import (
    GapParams,
    get_trace_variants,
    import_petri_net,
    read_event_log,
)
from gap_aware_alignments.gap_aware_alignment import align
from gap_aware_alignments.profile import (
    Cluster,
    ProfileModel,
    cluster_medoid,
    cluster_profile,
    cluster_reference,
    cluster_variants,
    pairwise_dissimilarity,
    stable_positions,
)
from gap_aware_alignments.profile_alignment import profile_align
from gap_aware_alignments.reachability_graph import ReachabilityGraph

PN_TAGS = ["_pn30", "_pn50", "_pn70"]

COST_MODEL = "affine"
GAP_OPENING_COST = 1.0
GAP_EXTENSION_COST = 0.5
GAP_POWER = 0.7
MAX_VARIANTS = 120                       # sensitivity re-aligns many times -> keep modest

# reference operating point (held fixed while another axis is swept)
REF_THETA = 0.8
REF_RHO = 2.0
REF_THRESHOLD = 4.0   # tau for the theta/rho sweeps (data-driven cluster count, Def. 5.2)

THETA_GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.01]   # 0 and >1 are the anchor checks
RHO_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]
CLUSTER_GRID = [2, 4, 8, 16, 32]


def _params(theta=REF_THETA, rho=REF_RHO) -> GapParams:
    return GapParams(
        cost_model=COST_MODEL, gap_opening_cost=GAP_OPENING_COST,
        gap_extension_cost=GAP_EXTENSION_COST, gap_power=GAP_POWER,
        stability_threshold=theta, penalty_factor=rho,
    )


def _build_model(variants, freqs, model_graph, params, D, n_clusters=None, threshold=None) -> ProfileModel:
    """Build a ProfileModel from a precomputed dissimilarity matrix ``D``."""
    clustering = cluster_variants(freqs, D, n_clusters=n_clusters, threshold=threshold)
    clusters = []
    for cluster in clustering:
        medoid = cluster_medoid(cluster, freqs, D)
        reference = cluster_reference(variants[medoid], model_graph, params)
        profile = cluster_profile(cluster, variants, freqs, reference, params)
        clusters.append(Cluster(cluster, reference, profile,
                                stable_positions(profile, params.stability_threshold)))
    return ProfileModel(clusters=clusters, params=params, variants=variants)


def _aggregate(variants, freqs, model_graph, model, params):
    """Frequency-weighted mean weighted (affine) cost and penalized share."""
    ci = params.cost_index
    total = sum(freqs)
    cost_sum = pen_sum = 0.0
    for variant, freq in zip(variants, freqs):
        cluster = model.clusters[model.assign(variant)]
        u, _ = align(variant, model_graph, GAP_OPENING_COST, GAP_EXTENSION_COST, GAP_POWER)
        p, _ = profile_align(variant, model_graph, cluster, params)
        cost_sum += p[ci] * freq
        pen_sum += freq if p[ci] > u[ci] + 1e-9 else 0
    return cost_sum / total, pen_sum / total


def sensitivity_benchmark(variants, freqs, model_graph, result_path, file_tag):
    base = _params()
    print("  Computing dissimilarity matrix + base profiles ...")
    D = pairwise_dissimilarity(variants, base)

    # unweighted baseline (theta/rho independent)
    ci = base.cost_index
    total = sum(freqs)
    unw_mean = sum(
        align(v, model_graph, GAP_OPENING_COST, GAP_EXTENSION_COST, GAP_POWER)[0][ci] * fr
        for v, fr in zip(variants, freqs)
    ) / total
    print(f"  unweighted mean affine cost = {unw_mean:.4f}")

    # --- theta sweep (fixed rho, fixed clustering) ---
    model = _build_model(variants, freqs, model_graph, base, D, threshold=REF_THRESHOLD)
    n_base = len(model.clusters)
    with open(result_path / f"result{file_tag}_profile_sensitivity_theta.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["theta", "rho", "n_clusters", "mean_cost_weighted", "penalized_share", "note"])
        for theta in THETA_GRID:
            p = _params(theta=theta, rho=REF_RHO)
            m = copy.deepcopy(model)
            for c in m.clusters:
                c.stable = stable_positions(c.profile, theta)
            m.params = p
            mean_cost, pen = _aggregate(variants, freqs, model_graph, m, p)
            note = ""
            if theta == 0.0:
                note = f"anchor: expect rho*unweighted={REF_RHO * unw_mean:.4f}"
            elif theta > 1.0:
                note = f"anchor: expect unweighted={unw_mean:.4f}"
            w.writerow([theta, REF_RHO, n_base, round(mean_cost, 4), round(pen, 4), note])
            print(f"    theta={theta}: mean_cost={mean_cost:.4f} penalized={pen:.3f} {note}")

    # --- rho sweep (fixed theta, fixed clustering) ---
    with open(result_path / f"result{file_tag}_profile_sensitivity_rho.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["theta", "rho", "n_clusters", "mean_cost_weighted", "penalized_share"])
        for rho in RHO_GRID:
            p = _params(theta=REF_THETA, rho=rho)
            m = copy.deepcopy(model)
            for c in m.clusters:
                c.stable = stable_positions(c.profile, REF_THETA)
            m.params = p
            mean_cost, pen = _aggregate(variants, freqs, model_graph, m, p)
            w.writerow([REF_THETA, rho, n_base, round(mean_cost, 4), round(pen, 4)])
            print(f"    rho={rho}: mean_cost={mean_cost:.4f} penalized={pen:.3f}")

    # --- clustering granularity sweep (fixed theta, rho) ---
    with open(result_path / f"result{file_tag}_profile_sensitivity_clusters.csv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["theta", "rho", "n_clusters", "mean_cost_weighted", "penalized_share"])
        for k in CLUSTER_GRID:
            if k > len(variants):
                continue
            p = _params(theta=REF_THETA, rho=REF_RHO)
            m = _build_model(variants, freqs, model_graph, p, D, k)
            mean_cost, pen = _aggregate(variants, freqs, model_graph, m, p)
            w.writerow([REF_THETA, REF_RHO, k, round(mean_cost, 4), round(pen, 4)])
            print(f"    n_clusters={k}: mean_cost={mean_cost:.4f} penalized={pen:.3f}")


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
                if pnml_file.is_file():
                    benchmarks.append((tag, pnml_file))

        for file_tag, pnml_file in benchmarks:
            print(f" -> {pnml_file.stem}")
            model_graph = ReachabilityGraph(import_petri_net(pnml_file))
            sensitivity_benchmark(variants, freqs, model_graph, cur_path, file_tag)

    print(f"\nDone. Results in {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
