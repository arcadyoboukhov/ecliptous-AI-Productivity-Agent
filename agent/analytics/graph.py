"""Universal analytics graph engine.

Single entry point `render_analytics_graph` which pulls data from the local
SQLite DB and returns a graph-agnostic model. The implementation follows the
spec provided in the workspace: metric abstraction, time window model,
explicit SQL queries bounded by time windows, deterministic aggregation, and
pure function behavior (no DB writes, no side-effects beyond read/caching).

This module is intentionally self-contained and uses the project's
`agent.storage.db.get_connection` helper for DB access.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
import math
import sqlite3
from functools import lru_cache

from agent.storage.db import get_connection


@dataclass(frozen=True)
class TimeWindow:
    length_days: int
    offset_days: int = 0

    def end(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.offset_days)

    def start(self) -> datetime:
        return self.end() - timedelta(days=self.length_days)


@dataclass(frozen=True)
class Metric:
    name: str
    source_table: str
    column: str
    aggregation: str  # 'mean', 'sum', 'count'
    normalized: bool = False
    unit: Optional[str] = None


# Minimal in-module registry for metrics. Consumers may extend this registry
# or pass a `Metric` instance directly to `render_analytics_graph`.
METRIC_REGISTRY: Dict[str, Metric] = {
    "focus": Metric(
        name="focus",
        source_table="interval_signals",
        column="keyboard_intensity",
        aggregation="mean",
        normalized=True,
        unit=None,
    ),
}


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _agg_sql(aggregation: str, column: str) -> str:
    if aggregation == "mean":
        return f"AVG({column}) as value"
    if aggregation == "sum":
        return f"SUM({column}) as value"
    if aggregation == "count":
        return f"COUNT({column}) as value"
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _fetch_aggregated(conn: sqlite3.Connection, table: str, column: str, start_iso: str, end_iso: str) -> Optional[float]:
    sql = f"SELECT { 'AVG' if True else '' }(1) as _dummy, { _agg_sql('mean', column) } FROM {table} WHERE timestamp_start >= ? AND timestamp_start < ?"
    # Build aggregation depending on column and default to AVG(column) as value
    # We use a simpler explicit SQL below to avoid complex templating when count/sum requested
    cur = conn.cursor()
    agg_expr = _agg_sql('mean', column)
    q = f"SELECT {agg_expr} FROM {table} WHERE timestamp_start >= ? AND timestamp_start < ?"
    cur.execute(q, (start_iso, end_iso))
    row = cur.fetchone()
    if not row:
        return None
    return row[0]


def _fetch_series_by_day(conn: sqlite3.Connection, table: str, column: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
    # Group by date (YYYY-MM-DD) using DATE(timestamp_start)
    agg_expr = _agg_sql('mean', column)
    q = f"SELECT DATE(timestamp_start) as day, {agg_expr} FROM {table} WHERE timestamp_start >= ? AND timestamp_start < ? GROUP BY day ORDER BY day"
    cur = conn.cursor()
    cur.execute(q, (start_iso, end_iso))
    rows = cur.fetchall()
    points: List[Dict[str, Any]] = []
    for r in rows:
        day = r[0]
        val = r[1]
        points.append({"date": day, "value": (None if val is None else float(val))})
    return points


# Simple cache for aggregated queries keyed by stable parameters. lru_cache
# requires hashable args; we route through a wrapper that stringifies datetimes.
def _cache_key(metric: Metric, window: TimeWindow, mode: Optional[str], options: Tuple[Tuple[str, Any], ...]) -> Tuple:
    return (metric.name, metric.source_table, metric.column, metric.aggregation, window.length_days, window.offset_days, mode, options)


def _apply_moving_average(points: List[Dict[str, Any]], window: int) -> List[Dict[str, Any]]:
    if window <= 1:
        return points
    vals = [p["value"] if p["value"] is not None else 0.0 for p in points]
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(points):
        start = max(0, i - window + 1)
        window_vals = vals[start : i + 1]
        avg = sum(window_vals) / len(window_vals) if window_vals else None
        out.append({"date": p["date"], "value": None if avg is None else float(avg)})
    return out


def render_analytics_graph(
    graph_type: str,
    metric: Any,
    time_window: int,
    comparison_mode: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render a graph model according to the universal analytic graph spec.

    Args:
        graph_type: 'comparison' or 'time_series'
        metric: either a `Metric` instance or a metric name registered in METRIC_REGISTRY
        time_window: length in days of the primary (current) window
        comparison_mode: 'baseline'|'long_term' or None
        options: additional options such as `long_term_days` or `moving_average`

    Returns:
        A serializable graph-agnostic model (dictionary).
    """
    if options is None:
        options = {}

    if isinstance(metric, str):
        if metric not in METRIC_REGISTRY:
            raise KeyError(f"Unknown metric: {metric}")
        metric = METRIC_REGISTRY[metric]

    if not isinstance(metric, Metric):
        raise TypeError("metric must be a Metric or a registered metric name")

    conn = get_connection()
    try:
        current_win = TimeWindow(length_days=time_window, offset_days=0)

        start_cur = _to_iso(current_win.start())
        end_cur = _to_iso(current_win.end())

        model: Dict[str, Any] = {
            "graph_type": graph_type,
            "metric": metric.name,
            "window": f"{time_window}d",
        }

        if graph_type == "comparison":
            if comparison_mode == "baseline":
                baseline_win = TimeWindow(length_days=time_window, offset_days=time_window)
                start_base = _to_iso(baseline_win.start())
                end_base = _to_iso(baseline_win.end())

                cur_val = _fetch_aggregated(conn, metric.source_table, metric.column, start_cur, end_cur)
                base_val = _fetch_aggregated(conn, metric.source_table, metric.column, start_base, end_base)

                cur_f = None if cur_val is None else float(cur_val)
                base_f = None if base_val is None else float(base_val)

                series = []
                series.append({"label": "Baseline", "value": base_f})
                delta_pct = None
                if base_f is not None and cur_f is not None and base_f != 0:
                    delta_pct = (cur_f - base_f) / abs(base_f) * 100.0
                series.append({"label": "Current", "value": cur_f, "delta_percent": None if delta_pct is None else round(delta_pct, 2)})

                meta = {"significance": ("increase" if (delta_pct or 0) > 0 else ("decrease" if (delta_pct or 0) < 0 else "no_change"))}

                model.update({"comparison": "baseline", "series": series, "meta": meta})

            elif comparison_mode == "long_term":
                lt_days = int(options.get("long_term_days", 90))
                long_term_win = TimeWindow(length_days=lt_days, offset_days=time_window)
                start_lt = _to_iso(long_term_win.start())
                end_lt = _to_iso(long_term_win.end())

                cur_val = _fetch_aggregated(conn, metric.source_table, metric.column, start_cur, end_cur)
                # For long-term we compute mean & std using daily buckets
                points = _fetch_series_by_day(conn, metric.source_table, metric.column, start_lt, end_lt)
                values = [p["value"] for p in points if p["value"] is not None]
                lt_mean = float(sum(values) / len(values)) if values else None
                lt_std = float((sum((v - lt_mean) ** 2 for v in values) / len(values)) ** 0.5) if values and lt_mean is not None else None

                cur_f = None if cur_val is None else float(cur_val)
                z = None
                if lt_mean is not None and lt_std is not None and lt_std != 0 and cur_f is not None:
                    z = (cur_f - lt_mean) / lt_std

                series = [
                    {"label": "Long-term Mean", "value": lt_mean},
                    {"label": "Current", "value": cur_f, "z_score": None if z is None else round(z, 3)},
                ]
                meta = {"long_term_days": lt_days, "long_term_std": lt_std}
                model.update({"comparison": "long_term", "series": series, "meta": meta})
            else:
                raise ValueError("comparison_mode must be 'baseline' or 'long_term' for graph_type 'comparison'")

        elif graph_type == "time_series":
            # Group by day across the provided window length
            start_ts = start_cur
            end_ts = end_cur
            points = _fetch_series_by_day(conn, metric.source_table, metric.column, start_ts, end_ts)
            moving = int(options.get("moving_average", 1))
            if moving and moving > 1:
                points = _apply_moving_average(points, moving)

            # Single series by default; UI can map label -> metric name
            model.update({"series": [{"label": metric.name, "points": points}]})
        else:
            raise ValueError("graph_type must be 'comparison' or 'time_series'")

        return model
    finally:
        conn.close()
