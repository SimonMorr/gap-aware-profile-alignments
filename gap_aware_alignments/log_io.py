from __future__ import annotations

from pathlib import Path

import polars as pl
import r4pm

CASE_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
TIMESTAMP_COLUMN = "time:timestamp"


def read_event_log(xes_path: str | Path) -> pl.DataFrame:
    """Import an XES event log into a Polars DataFrame using Rust4PM."""
    result = r4pm.df.import_xes(str(xes_path))
    df = result[0] if isinstance(result, tuple) else result
    return df


def get_trace_variants(event_log: pl.DataFrame) -> list[tuple[tuple[str, ...], int]]:
    """Return ``(variant, frequency)`` pairs sorted by descending frequency.

    Events are ordered per case by timestamp (stably preserving the original row
    order on ties), aggregated into an activity-name sequence, and identical
    sequences are counted as one variant whose frequency is the number of cases.
    """
    sort_columns = [c for c in (CASE_COLUMN, TIMESTAMP_COLUMN) if c in event_log.columns]

    per_case = (
        event_log.with_row_index("__row")
        .sort(sort_columns + ["__row"] if sort_columns else ["__row"], maintain_order=True)
        .group_by(CASE_COLUMN, maintain_order=True)
        .agg(pl.col(ACTIVITY_COLUMN).alias("__variant"))
    )

    # ``group_by`` is hash-based and therefore orders equal-frequency variants
    # non-deterministically across runs; sorting by the variant sequence itself as
    # a secondary key gives a total, reproducible order (otherwise the whole
    # downstream clustering / reference pipeline is run-dependent).
    variant_counts = (
        per_case.group_by("__variant")
        .agg(pl.len().alias("__frequency"))
        .with_columns(pl.col("__variant").list.join("").alias("__variant_key"))
        .sort(["__frequency", "__variant_key"], descending=[True, False])
    )

    return [
        (tuple(variant), int(frequency))
        for variant, frequency in zip(
            variant_counts["__variant"].to_list(),
            variant_counts["__frequency"].to_list(),
        )
    ]
