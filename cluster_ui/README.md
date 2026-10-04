# Trace Variant Clustering — local UI

Interactive dendrogram of trace variants under the gap-aware dissimilarity
`δ_g`, with a draggable `τ` cut and cost-model switching. A Python backend
(FastAPI) computes the clustering tree on demand; a Vue 3 frontend renders it.

```
cluster_api.py          FastAPI backend  (reuses the gap_aware_alignments package)
cluster_ui/             Vue 3 + Vite frontend
```

## 1. Backend (interface)

From the project root (`gap_aware_alignments_code/`), in the Python env that has
the project deps (`r4pm`, `polars`, `networkx`) plus FastAPI:

```bash
pip install fastapi uvicorn
uvicorn cluster_api:app --reload --port 8000
```

Endpoints:
- `GET /api/logs` → available event logs + cost models
- `GET /api/tree?log=<name>&model=<classic|affine|log|power>&max_variants=150`
  → the merge tree (with linkage heights), leaf frequencies/sequences, activity dictionary

The pairwise dissimilarity is `O(#variants²)`, so the **first** request per
(log, model, max_variants) is slow (seconds–minutes); results are cached under
`.cache/` and instant afterwards. Optionally pre-warm all combinations:

```bash
python cluster_api.py --warm
```

## 2. Frontend

```bash
cd cluster_ui
npm install
npm run dev          # http://localhost:5173
```

The Vite dev server proxies `/api` to the backend on `:8000`, so both must run.
Choose a log, switch the cost model, drag the `τ` slider (left = 1 cluster,
right = all singletons) or type a cluster count `k`; the right panel lists the
clusters and their member variants. Switching the cost model keeps the cluster
count fixed (δ_g is on a different scale per model).

To build a static bundle: `npm run build` → `cluster_ui/dist/`.
