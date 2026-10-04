from __future__ import annotations

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
from gap_aware_alignments.trace_reference import COST_INDEX

from profile_evaluation import _spearman, profile_fitness  # reuse Spearman + fitness

PN_TAGS = ["_pn30", "_pn50", "_pn70"]
COST_MODELS = ["classic", "affine", "log", "power"]

# operating point (same as the affine run, only g varies)
STABILITY_THRESHOLD = 0.8
PENALTY_FACTOR = 2.0
GAP_OPENING_COST = 1.0
GAP_EXTENSION_COST = 0.5
GAP_POWER = 0.7
CLUSTER_THRESHOLD = 4.0     # tau (data-driven cluster count, Def. 5.2)
MAX_VARIANTS = 300


def _params(cost_model: str) -> GapParams:
    return GapParams(
        cost_model=cost_model, gap_opening_cost=GAP_OPENING_COST,
        gap_extension_cost=GAP_EXTENSION_COST, gap_power=GAP_POWER,
        stability_threshold=STABILITY_THRESHOLD, penalty_factor=PENALTY_FACTOR,
    )


def _build_clusters(clustering, variants, freqs, model_graph, params, D) -> ProfileModel:
    clusters = []
    for cluster in clustering:
        medoid = cluster_medoid(cluster, freqs, D)
        reference = cluster_reference(variants[medoid], model_graph, params)
        profile = cluster_profile(cluster, variants, freqs, reference, params)
        clusters.append(Cluster(cluster, reference, profile,
                                stable_positions(profile, params.stability_threshold)))
    return ProfileModel(clusters=clusters, params=params, variants=variants)


def main() -> int:
    data_path = Path("data").resolve()
    result_path = Path("output") / datetime.now().strftime("%Y%m%d%H%M%S")
    result_path.mkdir(parents=True)
    rows = []

    for xes_file in sorted(data_path.glob("*.xes")):
        if not xes_file.is_file():
            continue
        log = xes_file.stem
        print(f"{log}")
        vf = get_trace_variants(read_event_log(xes_file))[:MAX_VARIANTS]
        variants = [v for v, _ in vf]
        freqs = [f for _, f in vf]

        # thresholds for this log
        benchmarks: list[tuple[str, Path]] = []
        if (single := data_path / f"{log}.pnml").is_file():
            benchmarks.append(("", single))
        else:
            for tag in PN_TAGS:
                pnml = data_path / f"{log}{tag}.pnml"
                if pnml.is_file():
                    benchmarks.append((tag, pnml))
        graphs = {tag: ReachabilityGraph(import_petri_net(pnml)) for tag, pnml in benchmarks}

        # Derive the cluster count k from the affine tau-clustering (the main run's
        # granularity). A fixed tau cannot be shared across cost models because
        # delta_g has a different scale per model (log/power are sublinear ->
        # everything would merge into one cluster). Fixing k instead compares the
        # models at matched granularity; the affine D is reused below.
        ref_params = _params("affine")
        D_affine = pairwise_dissimilarity(variants, ref_params)
        k = len(cluster_variants(freqs, D_affine, threshold=CLUSTER_THRESHOLD))
        print(f"  reference clustering (affine, tau={CLUSTER_THRESHOLD}) -> k={k} clusters", flush=True)

        for g in COST_MODELS:
            params = _params(g)
            ci = COST_INDEX[g]
            print(f"  cost model = {g}: dissimilarity matrix ...", flush=True)
            D = D_affine if g == "affine" else pairwise_dissimilarity(variants, params)
            clustering = cluster_variants(freqs, D, n_clusters=k)
            n_clusters = len(clustering)

            for tag, _ in benchmarks:
                rg = graphs[tag]
                bw = rg.best_worst_cost      # |p_min|: shortest accepting model run
                model = _build_clusters(clustering, variants, freqs, rg, params, D)

                unw, wt, fit = [], [], []
                pen_freq = chg_freq = tot = 0
                for v, f in zip(variants, freqs):
                    cl = model.clusters[model.assign(v)]
                    u, ua = align(v, rg, GAP_OPENING_COST, GAP_EXTENSION_COST, GAP_POWER)
                    w, wa = profile_align(v, rg, cl, params)
                    unw.append(u[ci]); wt.append(w[ci]); tot += f
                    fit.append(profile_fitness(w[ci], len(v), bw, bool(cl.stable), g, PENALTY_FACTOR))
                    pen_freq += f if w[ci] > u[ci] + 1e-9 else 0
                    chg_freq += f if ua[ci]["alignment"] != wa[ci]["alignment"] else 0

                mean_u = sum(x * f for x, f in zip(unw, freqs)) / tot
                mean_w = sum(x * f for x, f in zip(wt, freqs)) / tot
                mean_fit = sum(x * f for x, f in zip(fit, freqs)) / tot
                row = {
                    "log": log, "cost_model": g, "threshold": tag or "single",
                    "n_clusters": n_clusters,
                    "penalized_share": round(pen_freq / tot, 4),
                    "alignment_changed": round(chg_freq / tot, 4),
                    "mean_cost_unweighted": round(mean_u, 4),
                    "mean_cost_weighted": round(mean_w, 4),
                    "mean_fitness_weighted": round(mean_fit, 4),
                    "spearman": round(_spearman(unw, wt), 4),
                }
                rows.append(row)
                print(f"    {tag or 'single':6} {g:8} clusters={n_clusters:3} "
                      f"pen={row['penalized_share']:.3f} "
                      f"cost {mean_u:.3f}->{mean_w:.3f} fit^P={mean_fit:.3f} "
                      f"spearman={row['spearman']:.3f}",
                      flush=True)

    out = result_path / "profile_allmodels_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. Summary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
