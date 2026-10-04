from __future__ import annotations

from dataclasses import dataclass, field

from gap_aware_alignments.gap_aware_alignment import align
from gap_aware_alignments.reachability_graph import ReachabilityGraph
from gap_aware_alignments.trace_reference import (
    COST_INDEX,
    Trace,
    align_to_reference,
    model_projection,
    reference_reachability_graph,
)


@dataclass
class GapParams:
    """Gap cost parameters shared across the profile pipeline."""

    cost_model: str = "affine"          # which g is used for delta_g / clustering
    gap_opening_cost: float = 1.0       # C_open
    gap_extension_cost: float = 0.5     # C_extend
    gap_power: float = 0.7              # d (power model)
    stability_threshold: float = 0.8    # theta (Def. 5.7)
    penalty_factor: float = 2.0         # rho (Def. 5.10), > 1

    @property
    def cost_index(self) -> int:
        return COST_INDEX[self.cost_model]


# --------------------------------------------------------------------------- #
# Pairwise dissimilarity
# --------------------------------------------------------------------------- #

def pairwise_dissimilarity(variants: list[Trace], params: GapParams) -> list[list[float]]:
    """Symmetric matrix ``D[i][j] = delta_g(variants[i], variants[j])`` (Def. 5.1)."""
    n = len(variants)
    D = [[0.0] * n for _ in range(n)]
    for j in range(n):
        ref_graph = reference_reachability_graph(variants[j])
        for i in range(n):
            if i == j:
                continue
            if D[i][j]:  # already filled from the symmetric entry
                continue
            costs, _ = align_to_reference(
                variants[i], ref_graph,
                params.gap_opening_cost, params.gap_extension_cost, params.gap_power,
            )
            d = costs[params.cost_index]
            D[i][j] = d
            D[j][i] = d
    return D


# --------------------------------------------------------------------------- #
# Agglomerative clustering (frequency-weighted average linkage, Def. 5.2)
# --------------------------------------------------------------------------- #

def cluster_variants(
    freqs: list[int],
    D: list[list[float]],
    threshold: float | None = None,
    n_clusters: int | None = None,
) -> list[list[int]]:
    """Agglomerative clustering with frequency-weighted average linkage.

    Clusters (lists of variant indices) are merged by minimal linkage
    ``Lambda_g(A, B) = sum_{a in A, b in B} f(a) f(b) D[a][b] / (F(A) F(B))``.
    Merging stops when the minimum linkage exceeds ``threshold`` (Def. 5.2) or
    when ``n_clusters`` clusters remain, whichever triggers first.
    """
    def key(x, y):
        return (x, y) if x < y else (y, x)

    clusters: list[list[int]] = [[i] for i in range(len(freqs))]
    F = {id(c): freqs[c[0]] for c in clusters}
    # cross-cluster weighted dissimilarity sums S(A, B) = sum f(a) f(b) D[a][b]
    S: dict[tuple[int, int], float] = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            S[key(id(clusters[a]), id(clusters[b]))] = freqs[a] * freqs[b] * D[a][b]

    while len(clusters) > 1:
        best = None
        best_link = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ca, cb = clusters[a], clusters[b]
                link = S[key(id(ca), id(cb))] / (F[id(ca)] * F[id(cb)])
                if best_link is None or link < best_link:
                    best_link, best = link, (a, b)
        if n_clusters is not None and len(clusters) <= n_clusters:
            break
        if threshold is not None and best_link > threshold:
            break
        a, b = best
        ca, cb = clusters[a], clusters[b]
        merged = ca + cb
        merged_id = id(merged)
        F[merged_id] = F[id(ca)] + F[id(cb)]
        for cc in clusters:
            if cc is ca or cc is cb:
                continue
            S[key(merged_id, id(cc))] = S[key(id(ca), id(cc))] + S[key(id(cb), id(cc))]
        clusters = [c for c in clusters if c is not ca and c is not cb] + [merged]
    return clusters


# --------------------------------------------------------------------------- #
# Medoid, reference, profile, stable positions
# --------------------------------------------------------------------------- #

def cluster_medoid(cluster: list[int], freqs: list[int], D: list[list[float]]) -> int:
    """Frequency-weighted medoid index (Def. 5.3); deterministic tie-break by index."""
    best, best_cost = None, None
    for i in cluster:
        cost = sum(freqs[j] * D[i][j] for j in cluster)
        if best_cost is None or cost < best_cost - 1e-12:
            best, best_cost = i, cost
    return best


def cluster_reference(medoid: Trace, model_graph: ReachabilityGraph, params: GapParams) -> Trace:
    """Model-conforming reference trace = pi_2 of the medoid's model alignment (Def. 5.4)."""
    _costs, aligns = align(
        medoid, model_graph,
        params.gap_opening_cost, params.gap_extension_cost, params.gap_power,
    )
    return model_projection(aligns[params.cost_index]["alignment"])


def cluster_profile(
    cluster: list[int],
    variants: list[Trace],
    freqs: list[int],
    reference: Trace,
    params: GapParams,
) -> list[float]:
    """Position profile ``P_l(p)`` for p in 1..len(reference) (Def. 5.6)."""
    h = len(reference)
    ref_graph = reference_reachability_graph(reference)
    matched = [0] * (h + 1)   # matched[p] = summed frequency of cases syncing position p
    total = 0
    for i in cluster:
        _costs, aligns = align_to_reference(
            variants[i], ref_graph,
            params.gap_opening_cost, params.gap_extension_cost, params.gap_power,
        )
        moves = aligns[params.cost_index]["alignment"]
        rp = 0
        seen: set[int] = set()
        for log_label, model_label in moves:
            if model_label != ">>":
                rp += 1
                if log_label != ">>" and rp not in seen:   # synchronous match at position rp
                    matched[rp] += freqs[i]
                    seen.add(rp)
        total += freqs[i]
    return [0.0] + [matched[p] / total if total else 0.0 for p in range(1, h + 1)]


def stable_positions(profile: list[float], theta: float) -> set[int]:
    """Stable reference positions ``S_l = {p : P_l(p) >= theta}`` (Def. 5.7)."""
    return {p for p in range(1, len(profile)) if profile[p] >= theta}


# --------------------------------------------------------------------------- #
# Fitted model
# --------------------------------------------------------------------------- #

@dataclass
class Cluster:
    indices: list[int]
    reference: Trace
    profile: list[float]
    stable: set[int]


@dataclass
class ProfileModel:
    clusters: list[Cluster]
    params: GapParams
    variants: list[Trace] = field(default_factory=list)

    def assign(self, trace: Trace) -> int:
        """Cluster assignment c(sigma): index of the nearest reference (Def. 5.8)."""
        best, best_d = None, None
        for idx, cluster in enumerate(self.clusters):
            ref_graph = reference_reachability_graph(cluster.reference)
            costs, _ = align_to_reference(
                trace, ref_graph,
                self.params.gap_opening_cost, self.params.gap_extension_cost, self.params.gap_power,
            )
            d = costs[self.params.cost_index]
            if best_d is None or d < best_d - 1e-12:
                best, best_d = idx, d
        return best


def build_profile_model(
    variants: list[Trace],
    freqs: list[int],
    model_graph: ReachabilityGraph,
    params: GapParams,
    threshold: float | None = None,
    n_clusters: int | None = None,
) -> ProfileModel:
    """Run steps 1-2 of Chapter 5 and return the fitted :class:`ProfileModel`."""
    D = pairwise_dissimilarity(variants, params)
    clustering = cluster_variants(freqs, D, threshold=threshold, n_clusters=n_clusters)

    clusters: list[Cluster] = []
    for cluster in clustering:
        medoid_idx = cluster_medoid(cluster, freqs, D)
        reference = cluster_reference(variants[medoid_idx], model_graph, params)
        profile = cluster_profile(cluster, variants, freqs, reference, params)
        stable = stable_positions(profile, params.stability_threshold)
        clusters.append(Cluster(indices=cluster, reference=reference, profile=profile, stable=stable))
    return ProfileModel(clusters=clusters, params=params, variants=variants)
