from __future__ import annotations

import math
import random

import pytest

from gap_aware_alignments import (
    GapParams,
    build_profile_model,
    import_petri_net,  # noqa: F401  (kept for convenience / real-data experiments)
)
from gap_aware_alignments.gap_aware_alignment import align
from gap_aware_alignments.petri_net import PetriNet
from gap_aware_alignments.profile import (
    cluster_profile,
    cluster_variants,
    pairwise_dissimilarity,
    stable_positions,
)
from gap_aware_alignments.profile_alignment import compute_footprints, profile_align
from gap_aware_alignments.reachability_graph import ReachabilityGraph
from gap_aware_alignments.trace_reference import reference_reachability_graph, trace_dissimilarity

C_OPEN, C_EXTEND, D_POWER = 1.0, 0.5, 0.7
COST_MODELS = ["classic", "affine", "log", "power"]
COST_INDEX = {"classic": 0, "affine": 1, "log": 2, "power": 3}


# --------------------------------------------------------------------------- #
# Independent oracle: brute-force trace-to-trace gap-aware dissimilarity
# --------------------------------------------------------------------------- #

def _gap_cost(k: int, model: str) -> float:
    if k <= 0:
        return 0.0
    if model == "classic":
        return float(k)
    if model == "affine":
        return C_OPEN + C_EXTEND * k
    if model == "log":
        return C_OPEN + C_EXTEND * math.log(k + 1)
    if model == "power":
        return C_OPEN + C_EXTEND * (k ** D_POWER)
    raise ValueError(model)


def _alignment_gap_cost(moves: list[str], model: str) -> float:
    """Sum of gap costs over maximal runs of non-'sync' moves."""
    total, run = 0.0, 0
    for mv in moves:
        if mv == "sync":
            if run:
                total += _gap_cost(run, model)
                run = 0
        else:
            run += 1
    if run:
        total += _gap_cost(run, model)
    return total


def brute_delta(a: tuple[str, ...], b: tuple[str, ...], model: str) -> float:
    """delta_g(a, b) by enumerating every trace-to-trace alignment (Def. 5.1)."""
    m, n = len(a), len(b)
    best = math.inf

    def rec(i: int, j: int, moves: list[str]) -> None:
        nonlocal best
        if i == m and j == n:
            best = min(best, _alignment_gap_cost(moves, model))
            return
        # pruning: current committed gap cost cannot exceed the best so far
        if _alignment_gap_cost(moves, model) > best:
            return
        if i < m and j < n and a[i] == b[j]:
            rec(i + 1, j + 1, moves + ["sync"])
        if j < n:
            rec(i, j + 1, moves + ["model"])
        if i < m:
            rec(i + 1, j, moves + ["log"])

    rec(0, 0, [])
    return best


# --------------------------------------------------------------------------- #
# Fixtures: small models
# --------------------------------------------------------------------------- #

def _arc(kind, frm, to):
    return {"from_to": {"type": kind, "nodes": [frm, to]}, "weight": 1}


@pytest.fixture(scope="module")
def branching_model() -> ReachabilityGraph:
    """Model with a choice: L(M) = {(a,b,c,f), (a,d,e,f)}."""
    net = {
        "places": {p: {"id": p} for p in ["p0", "p1", "pb", "pd", "p3", "p4"]},
        "transitions": {
            "ta": {"id": "ta", "label": "a"}, "tb": {"id": "tb", "label": "b"},
            "tc": {"id": "tc", "label": "c"}, "td": {"id": "td", "label": "d"},
            "te": {"id": "te", "label": "e"}, "tf": {"id": "tf", "label": "f"},
        },
        "arcs": [
            _arc("PlaceTransition", "p0", "ta"), _arc("TransitionPlace", "ta", "p1"),
            _arc("PlaceTransition", "p1", "tb"), _arc("TransitionPlace", "tb", "pb"),
            _arc("PlaceTransition", "pb", "tc"), _arc("TransitionPlace", "tc", "p3"),
            _arc("PlaceTransition", "p1", "td"), _arc("TransitionPlace", "td", "pd"),
            _arc("PlaceTransition", "pd", "te"), _arc("TransitionPlace", "te", "p3"),
            _arc("PlaceTransition", "p3", "tf"), _arc("TransitionPlace", "tf", "p4"),
        ],
        "initial_marking": {"p0": 1},
        "final_markings": [{"p4": 1}],
    }
    return ReachabilityGraph(PetriNet(net))


def _in_language(trace, model_graph) -> bool:
    """A trace is accepted iff its classic (linear) alignment cost is zero."""
    costs, _ = align(trace, model_graph, C_OPEN, C_EXTEND, D_POWER)
    return costs[0] == 0


# --------------------------------------------------------------------------- #
# 1. delta_g against the independent brute-force oracle
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model", COST_MODELS)
def test_dissimilarity_matches_bruteforce(model):
    rng = random.Random(7)
    alphabet = "abc"
    for _ in range(60):
        a = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
        b = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
        got = trace_dissimilarity(a, b, model)
        expected = brute_delta(a, b, model)
        assert got == pytest.approx(expected, abs=1e-9), (a, b, model, got, expected)


@pytest.mark.parametrize("model", COST_MODELS)
def test_dissimilarity_metric_basics(model):
    rng = random.Random(11)
    for _ in range(40):
        a = tuple(rng.choice("abc") for _ in range(rng.randint(1, 4)))
        b = tuple(rng.choice("abc") for _ in range(rng.randint(1, 4)))
        assert trace_dissimilarity(a, a, model) == 0                      # identity
        assert trace_dissimilarity(a, b, model) >= 0                      # non-negative
        assert trace_dissimilarity(a, b, model) == pytest.approx(         # symmetric
            trace_dissimilarity(b, a, model), abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Footprints on a hand-computed example (Def. 5.9)
# --------------------------------------------------------------------------- #

def test_footprints_hand_example():
    p = GapParams(cost_model="affine")
    event, boundary = compute_footprints(("a", "b", "c", "d"), ("a", "c", "d"), p)
    assert {i: sorted(s) for i, s in event.items()} == {1: [1], 2: [1, 2], 3: [2], 4: [3]}
    assert {q: sorted(s) for q, s in boundary.items()} == {
        0: [1], 1: [1, 2], 2: [1, 2], 3: [2, 3], 4: [3]}


# --------------------------------------------------------------------------- #
# 3. Structural properties (references, profiles, stability)
# --------------------------------------------------------------------------- #

def test_references_are_model_conforming(branching_model):
    variants = [("a", "b", "c", "f"), ("a", "b", "f"), ("a", "d", "e", "f"), ("a", "d", "f")]
    freqs = [20, 5, 15, 4]
    p = GapParams(cost_model="affine")
    model = build_profile_model(variants, freqs, branching_model, p, n_clusters=2)
    for cluster in model.clusters:
        assert _in_language(cluster.reference, branching_model), cluster.reference


def test_profile_values_in_unit_interval(branching_model):
    variants = [("a", "b", "c", "f"), ("a", "b", "f"), ("a", "d", "e", "f"), ("a", "d", "f")]
    freqs = [20, 5, 15, 4]
    p = GapParams(cost_model="affine")
    model = build_profile_model(variants, freqs, branching_model, p, n_clusters=2)
    for cluster in model.clusters:
        for value in cluster.profile[1:]:
            assert 0.0 <= value <= 1.0


def test_stability_monotone_in_theta(branching_model):
    variants = [("a", "b", "c", "f"), ("a", "b", "f"), ("a", "d", "e", "f"), ("a", "d", "f")]
    freqs = [20, 5, 15, 4]
    p = GapParams(cost_model="affine")
    D = pairwise_dissimilarity(variants, p)
    clustering = cluster_variants(freqs, D, n_clusters=2)
    for cluster in clustering:
        profile = cluster_profile(cluster, variants, freqs,
                                  tuple(variants[cluster[0]]), p)  # any reference; test the set logic
        low = stable_positions(profile, 0.3)
        high = stable_positions(profile, 0.9)
        assert high <= low  # fewer stable positions at a higher threshold


# --------------------------------------------------------------------------- #
# 4. Reduction invariants tying back to the validated Chapter-4 align
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def small_case(branching_model):
    variants = [("a", "b", "c", "f"), ("a", "b", "f"), ("a", "d", "e", "f"), ("a", "d", "f")]
    freqs = [20, 5, 15, 4]
    return branching_model, variants, freqs


def test_reduction_rho_one_equals_align(small_case):
    rg, variants, freqs = small_case
    p = GapParams(cost_model="affine", stability_threshold=0.8, penalty_factor=1.0)
    model = build_profile_model(variants, freqs, rg, p, n_clusters=2)
    for v in variants + [("a", "b", "f"), ("a", "d", "f"), ("a", "f")]:
        u, _ = align(v, rg, C_OPEN, C_EXTEND, D_POWER)
        pw, _ = profile_align(v, rg, model.clusters[model.assign(v)], p)
        assert u == pytest.approx(pw, abs=1e-9), v


def test_reduction_theta_zero_scales_by_rho(small_case):
    rg, variants, freqs = small_case
    rho = 2.0
    p = GapParams(cost_model="affine", stability_threshold=0.0, penalty_factor=rho)
    model = build_profile_model(variants, freqs, rg, p, n_clusters=2)
    for v in variants + [("a", "b", "f"), ("a", "d", "f")]:
        u, _ = align(v, rg, C_OPEN, C_EXTEND, D_POWER)
        pw, _ = profile_align(v, rg, model.clusters[model.assign(v)], p)
        assert [rho * x for x in u] == pytest.approx(list(pw), abs=1e-9), v


def test_profile_never_cheaper_than_unweighted(small_case):
    rg, variants, freqs = small_case
    p = GapParams(cost_model="affine", stability_threshold=0.8, penalty_factor=2.0)
    model = build_profile_model(variants, freqs, rg, p, n_clusters=2)
    for v in variants + [("a", "b", "f"), ("a", "d", "f")]:
        u, _ = align(v, rg, C_OPEN, C_EXTEND, D_POWER)
        pw, _ = profile_align(v, rg, model.clusters[model.assign(v)], p)
        assert all(b >= a - 1e-9 for a, b in zip(u, pw)), v


# --------------------------------------------------------------------------- #
# 5. Fully hand-computed branching example
# --------------------------------------------------------------------------- #

def test_hand_example_end_to_end(small_case):
    rg, variants, freqs = small_case
    p = GapParams(cost_model="affine", stability_threshold=0.8, penalty_factor=2.0)
    model = build_profile_model(variants, freqs, rg, p, n_clusters=2)

    by_ref = {c.reference: c for c in model.clusters}
    assert ("a", "b", "c", "f") in by_ref
    assert ("a", "d", "e", "f") in by_ref

    c_bc = by_ref[("a", "b", "c", "f")]
    c_de = by_ref[("a", "d", "e", "f")]
    # P(c) = 20/25 = 0.8 -> stable at theta=0.8 ; P(e) = 15/19 ≈ 0.79 -> not stable
    assert c_bc.profile[3] == pytest.approx(0.8, abs=1e-9)
    assert 3 in c_bc.stable
    assert c_de.profile[3] == pytest.approx(15 / 19, abs=1e-9)
    assert 3 not in c_de.stable

    # (a,b,f) skips c inside a stable region -> affine 1.5 penalized to 3.0
    u, _ = align(("a", "b", "f"), rg, C_OPEN, C_EXTEND, D_POWER)
    pw, _ = profile_align(("a", "b", "f"), rg, model.clusters[model.assign(("a", "b", "f"))], p)
    assert u[1] == pytest.approx(1.5, abs=1e-9)
    assert pw[1] == pytest.approx(3.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 6. Determinism
# --------------------------------------------------------------------------- #

def test_determinism(small_case):
    rg, variants, freqs = small_case
    p = GapParams(cost_model="affine")
    m1 = build_profile_model(variants, freqs, rg, p, n_clusters=2)
    m2 = build_profile_model(variants, freqs, rg, p, n_clusters=2)
    refs1 = sorted(c.reference for c in m1.clusters)
    refs2 = sorted(c.reference for c in m2.clusters)
    assert refs1 == refs2
    stable1 = sorted(tuple(sorted(c.stable)) for c in m1.clusters)
    stable2 = sorted(tuple(sorted(c.stable)) for c in m2.clusters)
    assert stable1 == stable2
