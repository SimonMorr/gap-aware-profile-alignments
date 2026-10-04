# Gap-Aware and Profile-Based Alignments

An implementation of gap-aware and profile-based trace-to-model alignments for
process deviation diagnostics. The reference model is given as a Petri net
(PNML) and the observed behaviour as an event log (XES), both imported via
[Rust4PM](https://rust4pm.aarkue.eu/) (`r4pm`).

## Installation

Requires Python 3.10–3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[viz,tool,dev]"   # core + figures + exploration tool + tests
```

The core alone needs only the base dependencies:

```bash
pip install -e .                   # r4pm, polars, networkx
```

Optional dependency groups: `viz` (matplotlib / pandas / numpy), `tool`
(fastapi / uvicorn), `dev` (pytest).

## Input

The event logs are not distributed with this repository. Place each log in
`data/` as `data/<log>.xes`, together with its reference Petri net(s): either a
single `data/<log>.pnml` or, per noise threshold, `data/<log>_pn30.pnml`,
`data/<log>_pn50.pnml`, `data/<log>_pn70.pnml`.

## Usage

Run the scripts from the repository root with the virtual environment active.
Results are written to a fresh `output/<timestamp>/` directory; the figure
scripts read the most recent run and write PDFs to `/tmp/`.

Alignments:

```bash
python main.py            # gap-aware alignment of every log against its Petri net(s)
python fitness.py         # fitness for the latest run
python profile_main.py    # cluster variants, learn profiles, profile-weighted alignments
```

Evaluation:

```bash
python profile_evaluation.py    # penalized share, cost shift, alignment changes, Spearman, fitness
python profile_allmodels.py     # the same across all four cost models
python profile_sensitivity.py   # sensitivity to the parameters
```

Figures:

```bash
python visualize_results.py            # gap-aware figures
python profile_visualize.py            # profile figures
python profile_allmodels_visualize.py  # cost-model comparison
```

Tests:

```bash
pytest
```

Exploration tool (clustering dendrogram and cluster profiles):

```bash
uvicorn cluster_api:app --reload --port 8000   # backend (needs the `tool` extra)
cd cluster_ui && npm install && npm run dev     # frontend -> http://localhost:5173
```

## License

MIT — see [LICENSE](LICENSE).
