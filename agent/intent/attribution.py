"""Attribution resolver: map session time to intents using intent timeline.

Provides pure function `resolve_session_attribution(session, intents_file)` that
reads the intents persistence file and computes time attributed to each intent
within the session interval. Returns a dict intent_id|None -> seconds.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import json
import os


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _build_active_intervals_from_events(events: List[dict]) -> List[Tuple[datetime, Optional[datetime]]]:
    """Given ordered events for a task, return list of (start, end) active intervals.
    end may be None for ongoing active period.
    """
    intervals: List[Tuple[datetime, Optional[datetime]]] = []
    active_start: Optional[datetime] = None
    for ev in sorted(events, key=lambda e: e.get("ts") or ""):
        state = ev.get("state")
        ts = _parse_iso(ev.get("ts"))
        if state == "ACTIVE":
            active_start = ts
        elif state in ("PAUSED", "COMPLETED", "ABANDONED"):
            if active_start:
                intervals.append((active_start, ts))
                active_start = None
    if active_start:
        intervals.append((active_start, None))
    return intervals


def _intersect_interval(a_start: datetime, a_end: datetime, b_start: datetime, b_end: Optional[datetime]) -> float:
    # return seconds of intersection between [a_start,a_end) and [b_start,b_end)
    if b_end is None:
        b_end = datetime.now(timezone.utc)
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    delta = (earliest_end - latest_start).total_seconds()
    return max(0.0, delta)


def resolve_session_attribution(session, intents_file: str) -> Dict[Optional[str], int]:
    """Compute attribution mapping for a single session.

    session: object with `start` and `end` datetime attributes.
    intents_file: path to intents.json written by IntentManager.
    Returns dict {intent_id_or_None: seconds}
    """
    # If the session already contains intent_segments (persisted Day 4), use them directly
    if hasattr(session, "intent_segments") and getattr(session, "intent_segments"):
        # To avoid double-counting when segments overlap, merge intervals per intent
        def _merge_intervals(intervals: List[Tuple[datetime, Optional[datetime]]]) -> List[Tuple[datetime, Optional[datetime]]]:
            if not intervals:
                return []
            # normalize None ends to now for sorting/merging purposes
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            norm = [(s, e or now) for s, e in intervals if s]
            norm.sort(key=lambda x: x[0])
            merged: List[Tuple[datetime, Optional[datetime]]] = []
            cur_s, cur_e = norm[0]
            for s, e in norm[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            return merged

        totals: Dict[Optional[str], float] = {}
        session_start = session.start
        session_end = session.end
        # collect per-intent segments
        per_intent: Dict[Optional[str], List[Tuple[datetime, Optional[datetime]]]] = {}
        for seg in session.intent_segments:
            intent_id, s, e = seg
            if s is None:
                continue
            per_intent.setdefault(intent_id, []).append((s, e))

        for intent_id, segs in per_intent.items():
            merged = _merge_intervals(segs)
            attributed = 0.0
            for s, e in merged:
                # e may be naive end; keep as-is (merge normalized None to now)
                attributed += _intersect_interval(session_start, session_end, s, e)
            if attributed > 0:
                totals[intent_id] = totals.get(intent_id, 0.0) + attributed
        # anything not covered -> None
        covered = sum(totals.values())
        session_duration = max(0.0, (session_end - session_start).total_seconds())
        remaining = session_duration - covered
        if remaining > 0:
            totals[None] = totals.get(None, 0.0) + remaining
        return {k: int(v) for k, v in totals.items()}

    if not os.path.exists(intents_file):
        return {None: int((session.end - session.start).total_seconds())}

    with open(intents_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", {})

    # collect intervals per task
    task_intervals: Dict[str, List[Tuple[datetime, Optional[datetime]]]] = {}
    for tid, td in tasks.items():
        events = td.get("events", [])
        intervals = _build_active_intervals_from_events(events)
        if intervals:
            task_intervals[tid] = intervals

    # attribution totals
    totals: Dict[Optional[str], float] = {}
    session_start = session.start
    session_end = session.end

    # initialize leftover as full session duration (will be reduced by attributed parts)
    remaining = (session_end - session_start).total_seconds()

    for tid, intervals in task_intervals.items():
        attributed = 0.0
        for s, e in intervals:
            attributed += _intersect_interval(session_start, session_end, s, e)
        if attributed > 0:
            totals[tid] = totals.get(tid, 0.0) + attributed
            remaining -= attributed

    if remaining > 0:
        totals[None] = totals.get(None, 0.0) + remaining

    # convert to ints (seconds)
    return {k: int(v) for k, v in totals.items()}


def compute_task_session_overlap(task_id: str, sessions: List, intents_file: Optional[str] = None) -> dict:
    """Compute overlap of a single task across a list of sessions.

    Returns a dict with:
      - total_seconds: int total attributed seconds for the task across sessions
      - overlaps: list of dicts {session: session_obj, overlap_seconds: int, percent: float}

    This is pure, side-effect free, and deterministic.
    """
    if intents_file is None:
        # default to manager constant if available
        try:
            from agent.intent.manager import INTENTS_FILE as _IF
            intents_file = _IF
        except Exception:
            intents_file = None

    total = 0
    overlaps = []

    for s in sessions:
        mapping = resolve_session_attribution(s, intents_file) if callable(resolve_session_attribution) else {}
        seconds = int(mapping.get(task_id, 0)) if mapping else 0
        if seconds > 0:
            session_seconds = int((s.end - s.start).total_seconds())
            percent = (seconds / session_seconds) * 100 if session_seconds > 0 else 0.0
            overlaps.append({
                "session": s,
                "overlap_seconds": seconds,
                "percent": percent,
            })
            total += seconds

    # sort overlaps by session start time
    overlaps.sort(key=lambda x: x["session"].start)
    return {"total_seconds": total, "overlaps": overlaps}


def compute_total_active_seconds(task_id: str, intents_file: Optional[str] = None) -> int:
    """Compute the total active (unpaused) seconds for a task using its event timeline.

    This sums all ACTIVE intervals (excluding PAUSED/COMPLETED/ABANDONED) and
    treats an open ACTIVE interval as ending at now.
    """
    if intents_file is None:
        try:
            from agent.intent.manager import INTENTS_FILE as _IF
            intents_file = _IF
        except Exception:
            intents_file = None

    if not intents_file or not os.path.exists(intents_file):
        return 0

    with open(intents_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", {})
    td = tasks.get(task_id)
    if not td:
        return 0

    events = td.get("events", [])
    intervals = _build_active_intervals_from_events(events)
    total = 0.0
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for s, e in intervals:
        if e is None:
            e = now
        # only count positive intervals
        try:
            delta = (e - s).total_seconds()
            if delta > 0:
                total += delta
        except Exception:
            continue
    return int(total)


def compute_total_active_seconds_bulk(intents_file: Optional[str] = None) -> Dict[str, int]:
    """Compute total active seconds for all tasks by reading intents file once.

    Returns dict mapping task_id -> total active seconds.
    """
    if intents_file is None:
        try:
            from agent.intent.manager import INTENTS_FILE as _IF
            intents_file = _IF
        except Exception:
            intents_file = None

    res: Dict[str, int] = {}
    if not intents_file or not os.path.exists(intents_file):
        return res

    with open(intents_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", {})
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    for tid, td in tasks.items():
        events = td.get("events", [])
        intervals = _build_active_intervals_from_events(events)
        total = 0.0
        for s, e in intervals:
            if e is None:
                e = now
            try:
                delta = (e - s).total_seconds()
                if delta > 0:
                    total += delta
            except Exception:
                continue
        res[tid] = int(total)

    return res
