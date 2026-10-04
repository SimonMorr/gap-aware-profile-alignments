from __future__ import annotations

from functools import lru_cache

from gap_aware_alignments.gap_aware_alignment import align
from gap_aware_alignments.petri_net import PetriNet
from gap_aware_alignments.reachability_graph import ReachabilityGraph

Trace = tuple[str, ...]

# Position of each cost model in the tuples returned by ``align``.
COST_INDEX = {"classic": 0, "affine": 1, "log": 2, "power": 3}


def linear_petri_net(reference: Trace) -> PetriNet:
    """Build the linear (chain) Petri net whose only accepted trace is ``reference``."""
    h = len(reference)
    places = {f"p{i}": {"id": f"p{i}"} for i in range(h + 1)}
    transitions = {f"t{j}": {"id": f"t{j}", "label": reference[j - 1]} for j in range(1, h + 1)}
    arcs = []
    for j in range(1, h + 1):
        arcs.append({"from_to": {"type": "PlaceTransition", "nodes": [f"p{j - 1}", f"t{j}"]}, "weight": 1})
        arcs.append({"from_to": {"type": "TransitionPlace", "nodes": [f"t{j}", f"p{j}"]}, "weight": 1})
    net = {
        "places": places,
        "transitions": transitions,
        "arcs": arcs,
        "initial_marking": {"p0": 1},
        "final_markings": [{f"p{h}": 1}],
    }
    return PetriNet(net)


@lru_cache(maxsize=4096)
def reference_reachability_graph(reference: Trace) -> ReachabilityGraph:
    """Reachability graph of the linear net for ``reference`` (cached per reference)."""
    return ReachabilityGraph(linear_petri_net(reference))


def align_to_reference(
    trace: Trace,
    reference_graph: ReachabilityGraph,
    gap_opening_cost: float = 1.0,
    gap_extension_cost: float = 0.5,
    gap_power: float = 0.7,
):
    """Trace-to-trace gap-aware alignment of ``trace`` against a prebuilt reference graph.

    Returns ``(costs, alignments)`` exactly like :func:`align`: ``costs`` is the
    tuple ``(classic, affine, log, power)`` and ``alignments`` the four canonical
    optimal alignments (moves ``(log_label, model_label)`` with ``">>"`` skips).
    """
    return align(trace, reference_graph, gap_opening_cost, gap_extension_cost, gap_power)


def trace_dissimilarity(
    a: Trace,
    b: Trace,
    cost_model: str = "affine",
    gap_opening_cost: float = 1.0,
    gap_extension_cost: float = 0.5,
    gap_power: float = 0.7,
) -> float:
    """Gap-aware trace dissimilarity ``delta_g(a, b)`` (Def. 5.1) under ``cost_model``."""
    costs, _ = align_to_reference(
        a, reference_reachability_graph(b), gap_opening_cost, gap_extension_cost, gap_power
    )
    return costs[COST_INDEX[cost_model]]


def model_projection(alignment_moves: list[tuple[str, str]]) -> Trace:
    """pi_2(gamma)|_Sigma: the model-side activity sequence of an alignment."""
    return tuple(model for _log, model in alignment_moves if model != ">>")
