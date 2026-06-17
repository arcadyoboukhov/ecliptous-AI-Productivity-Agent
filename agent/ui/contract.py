from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta, date
from typing import List, Optional, Dict, Any
import json

from agent.analytics.persistence import load_sessions
from agent.storage.db import get_connection, get_latest_analytics_snapshot


@dataclass(frozen=True)
class TimelineSegment:
    start: datetime
    end: datetime
    state: str
    confidence: Optional[float]
    severity: Optional[str]
    task_id: Optional[str]
    signal_snapshot: Dict[str, Any]
    explanation: List[str]


def _load_state_events_range(start_ts: datetime, end_ts: datetime) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT timestamp, payload FROM events WHERE event_type = 'STATE_CHANGE' AND timestamp <= ? ORDER BY timestamp ASC",
            (end_ts.isoformat(),)
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    events = []
    for ts_str, payload_str in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        try:
            payload = json.loads(payload_str) if payload_str else {}
        except Exception:
            payload = {"raw": payload_str}
        events.append({"ts": ts, "payload": payload})
    return events


def build_timeline_for_date(target_date: date) -> Dict[str, Any]:
    # produce a UI contract for the given date (UTC-aware)
    sessions = load_sessions()

    # date range (inclusive start, exclusive end)
    day_start = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    # load all state events up to end of day; we'll use latest-before lookups
    state_events = _load_state_events_range(day_start, day_end)

    timeline_segments: List[TimelineSegment] = []
    total_seconds_by_state: Dict[str, int] = {}

    # helper: find latest state event with ts <= t
    def latest_state_before(t: datetime) -> Optional[Dict[str, Any]]:
        out = None
        for ev in state_events:
            if ev["ts"] <= t:
                out = ev
            else:
                break
        return out

    for s in sessions:
        try:
            s_start = s.start
            s_end = s.end
        except Exception:
            continue
        if s_start is None or s_end is None:
            continue
        # normalize timezone
        if s_start.tzinfo is None:
            s_start = s_start.replace(tzinfo=timezone.utc)
        if s_end.tzinfo is None:
            s_end = s_end.replace(tzinfo=timezone.utc)

        # skip sessions outside our day
        if s_end <= day_start or s_start >= day_end:
            continue

        seg_start = max(s_start, day_start)
        seg_end = min(s_end, day_end)

        # if session has a per-minute timeline, iterate its buckets
        timeline = getattr(s, "timeline", {}) or {}
        if timeline:
            # produce ordered list of bucket times (assume keys are datetimes or ISO strings)
            buckets = []
            for k in sorted(timeline.keys()):
                try:
                    b = k if isinstance(k, datetime) else datetime.fromisoformat(k)
                except Exception:
                    continue
                if b.tzinfo is None:
                    b = b.replace(tzinfo=timezone.utc)
                if b < seg_start or b >= seg_end:
                    continue
                buckets.append((b, timeline[k]))

            if not buckets:
                # fallback: create one segment spanning the session range
                buckets = [(seg_start, {})]

            # map each bucket to a state (from latest committed STATE_CHANGE at or before bucket)
            bucket_states = []
            for b, vals in buckets:
                ev = latest_state_before(b)
                state = ev["payload"].get("to") if ev else "unknown"
                payload = ev["payload"] if ev else {}
                bucket_states.append((b, vals, state, payload))

            # group contiguous buckets with same state and same task
            cur_b = None
            cur_vals = []
            cur_state = None
            cur_payloads = []
            for b, vals, state, payload in bucket_states:
                task = payload.get("active_task")
                if cur_state is None:
                    cur_state = state
                    cur_b = b
                    cur_vals = [vals]
                    cur_payloads = [payload]
                    cur_task = task
                    prev_b = b
                    continue
                # if same state and same task, extend
                if state == cur_state and task == cur_task:
                    cur_vals.append(vals)
                    cur_payloads.append(payload)
                    prev_b = b
                    continue
                # flush current
                seg = _make_segment_from_buckets(cur_b, prev_b + timedelta(minutes=1), cur_state, cur_payloads, cur_vals)
                timeline_segments.append(seg)
                total_seconds_by_state[seg.state] = total_seconds_by_state.get(seg.state, 0) + int((seg.end - seg.start).total_seconds())
                # start new
                cur_state = state
                cur_b = b
                cur_vals = [vals]
                cur_payloads = [payload]
                cur_task = task
                prev_b = b

            # flush final
            if cur_state is not None:
                seg = _make_segment_from_buckets(cur_b, prev_b + timedelta(minutes=1), cur_state, cur_payloads, cur_vals)
                timeline_segments.append(seg)
                total_seconds_by_state[seg.state] = total_seconds_by_state.get(seg.state, 0) + int((seg.end - seg.start).total_seconds())

        else:
            # no per-minute timeline; use session-level state at session start
            ev = latest_state_before(seg_start)
            state = ev["payload"].get("to") if ev else "unknown"
            payload = ev["payload"] if ev else {}
            seg = TimelineSegment(
                start=seg_start,
                end=seg_end,
                state=state,
                confidence=payload.get("confidence"),
                severity=payload.get("severity"),
                task_id=payload.get("active_task"),
                signal_snapshot={
                    "active_app": payload.get("active_app"),
                    "input_events": getattr(s, "input_events", {}),
                    "apps": getattr(s, "apps", []),
                },
                explanation=[payload.get("reason")] if payload.get("reason") else [],
            )
            timeline_segments.append(seg)
            total_seconds_by_state[seg.state] = total_seconds_by_state.get(seg.state, 0) + int((seg.end - seg.start).total_seconds())

    # build summary ratios
    total_secs = sum(total_seconds_by_state.values()) or 1
    summary = {
        "aligned_ratio": round(total_seconds_by_state.get("ACTIVE_ALIGNED", 0) / total_secs, 2),
        "drift_ratio": round(total_seconds_by_state.get("ACTIVE_UNALIGNED", 0) / total_secs, 2),
        "idle_ratio": round(total_seconds_by_state.get("IDLE", 0) / total_secs, 2),
    }

    # load latest analytics snapshot for the day
    insights_snapshot = None
    try:
        insights_snapshot = get_latest_analytics_snapshot(day_start, day_end)
    except Exception:
        insights_snapshot = None

    # serialize segments to dicts
    timeline_out = []
    for seg in timeline_segments:
        timeline_out.append({
            "start": seg.start.astimezone(timezone.utc).isoformat(),
            "end": seg.end.astimezone(timezone.utc).isoformat(),
            "state": seg.state,
            "confidence": seg.confidence,
            "severity": seg.severity,
            "task": seg.task_id,
            "signals": seg.signal_snapshot,
            "explanation": seg.explanation,
        })

    return {
        "timeline": timeline_out,
        "summary": summary,
        "insights": insights_snapshot,
    }


def _make_segment_from_buckets(start: datetime, end: datetime, state: str, payloads: List[Dict[str, Any]], vals_list: List[Dict[str, Any]]) -> TimelineSegment:
    # Build a signal snapshot: aggregate basic metrics from buckets and payloads
    total_keys = 0
    total_clicks = 0
    apps = []
    reasons = []
    confidences = []
    task = None
    for p, v in zip(payloads, vals_list):
        if p.get("active_task"):
            task = p.get("active_task")
        if p.get("active_app"):
            apps.append(p.get("active_app"))
        if p.get("reason"):
            reasons.append(p.get("reason"))
        if p.get("confidence") is not None:
            try:
                confidences.append(float(p.get("confidence")))
            except Exception:
                pass
        # vals may contain keys/clicks
        total_keys += int(v.get("keys", 0))
        total_clicks += int(v.get("clicks", 0))

    minutes = max(1, len(vals_list))
    snapshot = {
        "active_app": max(set(apps), key=apps.count) if apps else None,
        "app_switches_last_min": 0,
        "input_events_per_min": int((total_keys + total_clicks) / minutes),
        "idle_seconds": 0,
    }

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else None

    return TimelineSegment(
        start=start,
        end=end,
        state=state,
        confidence=avg_conf,
        severity=None,
        task_id=task,
        signal_snapshot=snapshot,
        explanation=list(dict.fromkeys([r for r in reasons if r])),
    )


def dump_ui_contract_for_date(target_date: date) -> Dict[str, Any]:
    """Return the UI contract for `target_date` as a dict (not JSON string).

    The caller (CLI / UI) is responsible for serializing this and adding
    version/timestamp metadata.
    """
    data = build_timeline_for_date(target_date)
    return data
