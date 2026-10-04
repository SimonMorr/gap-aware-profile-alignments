from __future__ import annotations

from collections import defaultdict

from gap_aware_alignments.gap_aware_alignment import (
    AffineGapCostModel,
    AlignmentStateGraph,
    LogGapCostModel,
    PowerGapCostModel,
)
from gap_aware_alignments.profile import Cluster, GapParams, ProfileModel
from gap_aware_alignments.reachability_graph import ReachabilityGraph
from gap_aware_alignments.trace_reference import Trace, align_to_reference, reference_reachability_graph


def compute_footprints(
    trace: Trace, reference: Trace, params: GapParams
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Event footprints ``E[i]`` (i in 1..|trace|) and boundary footprints
    ``B[q]`` (q in 0..|trace|) of ``trace`` w.r.t. ``reference`` (Def. 5.9)."""
    ref_graph = reference_reachability_graph(reference)
    _costs, aligns = align_to_reference(
        trace, ref_graph, params.gap_opening_cost, params.gap_extension_cost, params.gap_power
    )
    moves = aligns[params.cost_index]["alignment"]
    m, h = len(trace), len(reference)

    # x_t / y_t: trace / reference events consumed after the first t columns.
    xs, ys = [0], [0]
    for u, v in moves:
        xs.append(xs[-1] + (0 if u == ">>" else 1))
        ys.append(ys[-1] + (0 if v == ">>" else 1))
    s = len(moves)

    # Event footprint of every trace position i (the column t with x_t = i).
    event: dict[int, set[int]] = {i: set() for i in range(1, m + 1)}
    for t in range(1, s + 1):
        u, v = moves[t - 1]
        if u == ">>":
            continue
        i = xs[t]  # this column consumes trace event i
        if v != ">>":                                   # synchronous match
            event[i] = {ys[t]}
        else:                                           # log move between references
            event[i] = {p for p in (ys[t], ys[t] + 1) if 1 <= p <= h}

    # Boundary footprint of every trace boundary q (all y_t at columns with x_t = q).
    y_at: dict[int, set[int]] = defaultdict(set)
    for t in range(0, s + 1):
        y_at[xs[t]].add(ys[t])
    boundary: dict[int, set[int]] = {}
    for q in range(0, m + 1):
        y_vals = y_at.get(q)
        if not y_vals:
            boundary[q] = set()
        else:
            lo, hi = min(y_vals), max(y_vals) + 1
            boundary[q] = {p for p in range(lo, hi + 1) if 1 <= p <= h}
    return event, boundary


def _gap_footprint(
    i0: int, idx: int, model_cost: int,
    event: dict[int, set[int]], boundary: dict[int, set[int]],
) -> set[int]:
    """Reference footprint of a gap covering trace events i0+1..idx (Def. 5.9)."""
    fp: set[int] = set()
    for i in range(i0 + 1, idx + 1):
        fp |= event.get(i, set())
    if model_cost > 0:
        fp |= boundary.get(idx, set())
    return fp


def profile_align(
    trace: Trace,
    model_graph: ReachabilityGraph,
    cluster: Cluster,
    params: GapParams,
):
    """Optimal profile-weighted trace-to-model alignment of ``trace`` (Def. 5.11).

    Returns ``(costs, alignments)`` like :func:`align`, but every gap cost is
    multiplied by ``rho`` when the gap's reference footprint hits a stable
    position of ``cluster``.
    """
    event, boundary = compute_footprints(trace, cluster.reference, params)
    stable = cluster.stable

    graph = AlignmentStateGraph(
        model_graph,
        trace,
        AffineGapCostModel(params.gap_opening_cost, params.gap_extension_cost),
        LogGapCostModel(params.gap_opening_cost, params.gap_extension_cost),
        PowerGapCostModel(params.gap_opening_cost, params.gap_extension_cost, params.gap_power),
    )

    rho = params.penalty_factor
    for u, v, data in graph.edges(data=True):
        if data["type"] != "gap":
            continue
        i0, idx = u[0], v[0]
        model_cost = sum(1 for label in data["label"] if label is None)
        footprint = _gap_footprint(i0, idx, model_cost, event, boundary)
        if footprint & stable:
            data["classic_cost"] *= rho
            data["affine_gap_cost"] *= rho
            data["log_gap_cost"] *= rho
            data["power_gap_cost"] *= rho

    return graph.get_optimal_alignment_costs(), graph.get_optimal_alignments()


def profile_align_with_model(
    trace: Trace, model_graph: ReachabilityGraph, model: ProfileModel
):
    """Assign ``trace`` to its nearest cluster (Def. 5.8) and align it profile-weighted."""
    cluster = model.clusters[model.assign(trace)]
    return profile_align(trace, model_graph, cluster, model.params)
