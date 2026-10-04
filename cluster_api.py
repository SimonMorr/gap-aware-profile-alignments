from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gap_aware_alignments import GapParams, get_trace_variants, import_petri_net, read_event_log
from gap_aware_alignments.profile import cluster_profile, cluster_reference, pairwise_dissimilarity
from gap_aware_alignments.reachability_graph import ReachabilityGraph

DATA = Path("data")
CACHE = Path(".cache")
CACHE.mkdir(exist_ok=True)
MODELS = ["classic", "affine", "log", "power"]
PN_TAGS = ["_pn30", "_pn50", "_pn70"]
DEFAULT_MAX = 150

# in-memory cache of (log, max_variants) -> (variants, freqs), so the profile
# endpoint does not re-read the (large) XES on every cluster click.
_variants: dict[tuple[str, int], tuple[list, list]] = {}


def load_variants(log: str, max_variants: int):
    key = (log, max_variants)
    if key not in _variants:
        xes = DATA / f"{log}.xes"
        if not xes.is_file():
            raise HTTPException(404, f"Event log not found: {log}")
        vf = get_trace_variants(read_event_log(str(xes)))[:max_variants]
        _variants[key] = ([v for v, _ in vf], [f for _, f in vf])
    return _variants[key]

app = FastAPI(title="Trace Variant Clustering API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local dev; tighten if ever exposed
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Frequency-weighted average-linkage AHC, recording the full merge tree
# --------------------------------------------------------------------------- #

def _linkage(freqs: list[int], D: list[list[float]]) -> list[tuple[int, int, float]]:
    """Return the merge sequence ``(node_a, node_b, linkage_height)`` (leaves 0..n-1)."""
    n = len(freqs)
    nodes = {i: {"m": [i], "F": freqs[i]} for i in range(n)}
    active = list(range(n))

    def key(x, y):
        return (x, y) if x < y else (y, x)

    S = {}
    for ai in range(n):
        for bi in range(ai + 1, n):
            a, b = active[ai], active[bi]
            S[key(a, b)] = freqs[a] * freqs[b] * D[a][b]

    nid = n
    merges: list[tuple[int, int, float]] = []
    while len(active) > 1:
        best, best_link = None, None
        for ai in range(len(active)):
            a = active[ai]
            for bi in range(ai + 1, len(active)):
                b = active[bi]
                link = S[key(a, b)] / (nodes[a]["F"] * nodes[b]["F"])
                if best_link is None or link < best_link:
                    best_link, best = link, (a, b)
        a, b = best
        nodes[nid] = {"m": nodes[a]["m"] + nodes[b]["m"], "F": nodes[a]["F"] + nodes[b]["F"]}
        merges.append((a, b, best_link))
        for c in active:
            if c in (a, b):
                continue
            S[key(nid, c)] = S[key(a, c)] + S[key(b, c)]
        active = [c for c in active if c not in (a, b)] + [nid]
        nid += 1
    return merges


def _build_tree(merges, n, leaves) -> dict:
    node = {i: {"id": i, "height": 0.0, "leaf": True, **leaves[i]} for i in range(n)}
    nid = n
    for a, b, h in merges:
        node[nid] = {"id": nid, "height": round(h, 4), "children": [node[a], node[b]]}
        nid += 1
    return node[nid - 1]


def compute_tree(log: str, model: str, max_variants: int) -> dict:
    xes = DATA / f"{log}.xes"
    if not xes.is_file():
        raise HTTPException(404, f"Event log not found: {log}")
    if model not in MODELS:
        raise HTTPException(400, f"Unknown cost model: {model}")

    variants, freqs = load_variants(log, max_variants)
    acts = sorted({a for v in variants for a in v})
    idx = {a: i for i, a in enumerate(acts)}
    leaves = [{"freq": freqs[i], "seq": [idx[a] for a in variants[i]]} for i in range(len(variants))]

    D = pairwise_dissimilarity(variants, GapParams(cost_model=model))
    merges = _linkage(freqs, D)
    return {"activities": acts, "total_freq": sum(freqs), "tree": _build_tree(merges, len(variants), leaves)}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/logs")
def logs():
    out = []
    for p in sorted(DATA.glob("*.xes")):
        name = p.stem
        tags = [t for t in PN_TAGS if (DATA / f"{name}{t}.pnml").is_file()]
        if not tags and (DATA / f"{name}.pnml").is_file():
            tags = [""]
        out.append({"name": name, "thresholds": tags})
    return {"logs": out, "models": MODELS}


class ProfileReq(BaseModel):
    log: str
    model: str
    threshold: str = ""          # "_pn30" / "_pn50" / "_pn70" / ""
    max_variants: int = DEFAULT_MAX
    members: list[int]           # leaf ids = indices into the top-N variant list


@app.post("/api/profile")
def profile(req: ProfileReq):
    """Reference trace and position profile P_l(p) for one cluster (Def. 5.2-5.7).

    The stability threshold theta is applied on the client (P is theta-independent),
    so dragging theta needs no round-trip.
    """
    if req.model not in MODELS:
        raise HTTPException(400, f"Unknown cost model: {req.model}")
    pnml = DATA / f"{req.log}{req.threshold}.pnml"
    if not pnml.is_file():
        raise HTTPException(404, f"Petri net not found: {pnml.name}")
    variants, freqs = load_variants(req.log, req.max_variants)
    members = [variants[i] for i in req.members]
    mfreqs = [freqs[i] for i in req.members]
    params = GapParams(cost_model=req.model)

    # frequency-weighted medoid within the cluster (Def. 5.3)
    Dm = pairwise_dissimilarity(members, params)
    rng = range(len(members))
    medoid = min(rng, key=lambda i: sum(mfreqs[j] * Dm[i][j] for j in rng))

    rg = ReachabilityGraph(import_petri_net(str(pnml)))
    reference = cluster_reference(members[medoid], rg, params)     # Def. 5.4
    prof = cluster_profile(list(rng), members, mfreqs, reference, params)  # Def. 5.6
    return {
        "reference": list(reference),
        "profile": [round(x, 4) for x in prof[1:]],   # P_l(p), p = 1..h
        "medoid": list(members[medoid]),
        "n_cases": sum(mfreqs),
        "n_variants": len(members),
    }


@app.get("/api/tree")
def tree(log: str, model: str, max_variants: int = DEFAULT_MAX):
    cache = CACHE / f"tree_{log}_{model}_{max_variants}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    result = compute_tree(log, model, max_variants)
    cache.write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    # `python cluster_api.py --warm` precomputes+caches every (log, model) at DEFAULT_MAX.
    import sys
    if "--warm" in sys.argv:
        for p in sorted(DATA.glob("*.xes")):
            for m in MODELS:
                print(f"warming {p.stem} / {m} ...", flush=True)
                tree(p.stem, m)
        print("done")
    else:
        import uvicorn
        uvicorn.run("cluster_api:app", host="127.0.0.1", port=8000, reload=True)
