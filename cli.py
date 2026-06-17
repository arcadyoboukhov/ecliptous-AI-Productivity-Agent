#!/usr/bin/env python3
"""Command-line interface for task and session management (Week2 Day7).

Usage examples:
  python cli.py task list
  python cli.py task start "Write Docs"
  python cli.py task pause "Write Docs"
  python cli.py task stop "Write Docs"
  python cli.py task summary
  python cli.py session list
  python cli.py session stats
  python cli.py report daily
  python cli.py report trends
  python cli.py agent start
  python cli.py agent stop
  python cli.py agent status
"""
import argparse
from datetime import datetime, timezone, date, timedelta
import json
import os
from typing import Optional
from datetime import datetime, timezone
from agent.intent.manager import IntentManager
from agent.analytics.persistence import load_sessions, save_sessions, append_session
from agent.analytics.daily import aggregate_daily_summary, format_daily_report
from agent.analytics.trends import compute_trend_metrics, format_trend_report
from agent.storage.db import get_connection
from agent.intent.attribution import resolve_session_attribution
from agent.session.sessionizer import Session, SessionManager
from agent.process_manager import start as pm_start, stop as pm_stop, status as pm_status, is_running
from agent.ui.contract import dump_ui_contract_for_date
import shutil
from pathlib import Path


EVENTS_TABLE = "events"


def log_event_to_db(event_type: str, payload: Optional[dict] = None, ts: Optional[datetime] = None, session_id: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if ts is None:
        ts = datetime.now(timezone.utc)
    payload_str = None
    try:
        if payload is not None:
            payload_str = json.dumps(payload)
    except Exception:
        payload_str = str(payload)

    cursor.execute(
        "INSERT INTO events (timestamp, event_type, session_id, payload) VALUES (?, ?, ?, ?)",
        (ts.isoformat(), event_type, session_id, payload_str)
    )
    conn.commit()
    conn.close()


def cmd_task_list(args):
    mgr = IntentManager()
    for tid, task in mgr.tasks.items():
        print(f"- {tid}: {task.state} (created {task.created_at})")


def _process_event_locally_cli(ts, event_type, payload):
    """Process an INTENT event locally when the agent is not running.
    Uses a fresh SessionManager to handle the event and persists any completed sessions."""
    try:
        from agent.session.sessionizer import SessionManager
        sm = SessionManager()
        sm.handle_event(ts, event_type, payload)
        # persist completed sessions
        if sm.completed_sessions:
            from agent.analytics.persistence import load_sessions, save_sessions
            persisted = load_sessions()
            merged = 0
            for s in sm.completed_sessions:
                try:
                    exists = any(abs((s.start - cs.start).total_seconds()) < 1 and abs((s.end - cs.end).total_seconds()) < 1 for cs in persisted)
                except Exception:
                    exists = False
                if not exists:
                    persisted.append(s)
                    merged += 1
            if merged:
                save_sessions(persisted)
                sm.completed_sessions = []
    except Exception:
        pass


def cmd_task_start(args):
    print("ERROR: Manual task creation is disabled.")
    print("Tasks are now automatically inferred by the agent based on your activity.")
    print("Simply use your computer normally - the agent will detect and track work sessions.")
    return


def cmd_task_pause(args):
    print("ERROR: Manual task control is disabled.")
    print("Tasks are automatically managed by the agent based on your activity patterns.")
    return


def cmd_task_stop(args):
    print("ERROR: Manual task control is disabled.")
    print("Tasks are automatically managed by the agent based on your activity patterns.")
    return


def cmd_task_summary(args):
    sessions = load_sessions()
    totals = {}
    for s in sessions:
        # prefer persisted segments
        if getattr(s, "intent_segments", None):
            for intent_id, st, en in s.intent_segments:
                if st is None:
                    continue
                en = en or s.end
                totals[intent_id] = totals.get(intent_id, 0) + int((en - st).total_seconds())
        else:
            # fallback to resolver
            breakdown = resolve_session_attribution(s, os.path.join("agent", "intent", "intents.json"))
            for intent_id, seconds in breakdown.items():
                totals[intent_id] = totals.get(intent_id, 0) + seconds

    print("Task durations:")
    for intent_id, seconds in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        label = "Unattributed" if intent_id is None else intent_id
        print(f"  {label}: {seconds//60}m ({seconds}s)")


def cmd_task_unstable(args):
    """Display unstable tasks flagged for review."""
    import json
    
    try:
        from agent.task.inference import TaskInferenceEngine
        
        # Try to load persisted engine state if available
        engine = TaskInferenceEngine()
        
        # Load from storage if persisted
        try:
            from agent.storage.db import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT payload FROM events WHERE event_type = 'TASK_ENGINE_STATE' ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                data = json.loads(row[0])
                engine = TaskInferenceEngine.deserialize(data)
        except Exception:
            pass
        
        unstable_tasks = engine.get_unstable_tasks()
        
        if args.json:
            output = {
                'version': '1.0',
                'count': len(unstable_tasks),
                'unstable_tasks': unstable_tasks,
            }
            print(json.dumps(output, indent=2))
            return
        
        if not unstable_tasks:
            print("✅ All tasks are stable. No tasks flagged for review.")
            return
        
        print(f"\n⚠️  {len(unstable_tasks)} task(s) flagged for review:\n")
        
        for i, task in enumerate(unstable_tasks, 1):
            severity_icon = "🔴" if task['severity'] == 'critical' else "🟡"
            print(f"{i}. {severity_icon} Task {task['task_id'][:12]}")
            print(f"   Label: {task['label'] or '(unlabeled/latent)'}")
            print(f"   Drift: {task['drift']:.3f} (severity: {task['severity']})")
            print(f"   Members: {task['members']} sessions")
            print(f"   Confidence: {task['confidence']:.2f}")
            print(f"   Recommendation: {task['recommendation']}")
            print()
    
    except Exception as e:
        print(f"Error retrieving unstable tasks: {e}")



# =====================================================
# NEW: Session-Centric Model (SessionGate-based)
# =====================================================

def cmd_session_create(args):
    """Create a new session (not started yet)."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    name = args.name
    session_id = getattr(args, "session_id", None)
    try:
        session = mgr.create_session(name=name, session_id=session_id)
        print(f"Created session: {session.session_id} ('{session.name}')")
        if getattr(args, "json", False):
            print(json.dumps({"session_id": session.session_id, "name": session.name, "created_at": session.created_at.isoformat()}))
    except Exception as e:
        print(f"Error creating session: {e}")


def cmd_session_start(args):
    """Start a session, enabling tracking."""
    from agent.session.manager_v2 import SessionManager
    from agent.session.gate import SessionAlreadyActive
    mgr = SessionManager()
    session_id = args.session_id
    try:
        session = mgr.start_session(session_id)
        print(f"Started session: {session_id}")
        if getattr(args, "json", False):
            print(json.dumps({"session_id": session.session_id, "started_at": session.started_at.isoformat()}))
    except ValueError as e:
        print(f"Error: {e}")
    except SessionAlreadyActive as e:
        print(f"Error: {e}")


def cmd_session_pause(args):
    """Pause a session (stop collecting signals, keep session open)."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    session_id = args.session_id
    try:
        session = mgr.pause_session(session_id)
        print(f"Paused session: {session_id}")
        if getattr(args, "json", False):
            print(json.dumps({"session_id": session.session_id, "status": "paused"}))
    except Exception as e:
        print(f"Error: {e}")


def cmd_session_resume(args):
    """Resume a paused session (re-enable signal collection)."""
    from agent.session.manager_v2 import SessionManager
    from agent.session.gate import SessionAlreadyActive
    mgr = SessionManager()
    session_id = args.session_id
    try:
        session = mgr.resume_session(session_id)
        print(f"Resumed session: {session_id}")
        if getattr(args, "json", False):
            print(json.dumps({"session_id": session.session_id, "status": "active"}))
    except SessionAlreadyActive as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_session_end(args):
    """End a session, stopping tracking and finalizing all tasks."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    session_id = args.session_id
    try:
        session = mgr.end_session(session_id)
        print(f"Ended session: {session_id}")
        print(f"  Duration: {session.duration_seconds():.1f} seconds")
        print(f"  Tasks: {len(session.tasks)}")
        if getattr(args, "json", False):
            print(json.dumps({"session_id": session.session_id, "ended_at": session.ended_at.isoformat(), "duration_seconds": session.duration_seconds()}))
    except Exception as e:
        print(f"Error: {e}")


def cmd_session_list(args):
    """List all sessions."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    sessions = mgr.list_sessions()
    if not sessions:
        print("No sessions found")
        return
    
    print("Sessions:")
    for i, s in enumerate(sessions, 1):
        status = "ACTIVE" if s.is_active() else "ENDED"
        duration = s.duration_seconds() if s.started_at else 0
        print(f"  {i}. {s.session_id} | {s.name} | {status} | {duration:.1f}s | Tasks: {len(s.tasks)}")
    
    if getattr(args, "json", False):
        sessions_data = [{"session_id": s.session_id, "name": s.name, "status": "ACTIVE" if s.is_active() else "ENDED", "duration_seconds": s.duration_seconds()} for s in sessions]
        print(json.dumps({"sessions": sessions_data}))


def cmd_session_info(args):
    """Get detailed info about a session."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    session_id = args.session_id
    session = mgr.get_session(session_id)
    if not session:
        print(f"Session not found: {session_id}")
        return
    
    print(f"Session: {session_id}")
    print(f"  Name: {session.name}")
    print(f"  Created: {session.created_at.isoformat()}")
    print(f"  Started: {session.started_at.isoformat() if session.started_at else '(not started)'}")
    print(f"  Ended: {session.ended_at.isoformat() if session.ended_at else '(active)'}")
    print(f"  Duration: {session.duration_seconds():.1f} seconds")
    print(f"  Tasks: {len(session.tasks)}")
    if session.tasks:
        print("    Details:")
        for task in session.tasks.values():
            print(f"      - {task.task_id}: {task.name} ({task.state}) | {task.duration_seconds():.1f}s")


def _session_signals_dict(session):
    """Helper to serialize signal data for a session (active or ended)."""
    if not session or not session.signals:
        return None

    sb = session.signals
    metrics = sb.metrics_since(300)
    # If recent buffers are empty (e.g., reloaded session), fall back to aggregates
    if metrics.get("keys", 0) == 0 and metrics.get("clicks", 0) == 0 and metrics.get("mouse_distance", 0) == 0:
        metrics = {
            "keys": sb.keyboard_presses,
            "clicks": sb.mouse_clicks,
            "mouse_distance": sb.mouse_distance,
            "app_changes": len(sb.app_timeline),
            "active_app": sb.app_timeline[-1].app_name if sb.app_timeline else None,
            "window_seconds": 300,
            "last_ts": metrics.get("last_ts"),
        }
    return {
        "session_id": session.session_id,
        "mouse_distance": sb.mouse_distance,
        "mouse_clicks": sb.mouse_clicks,
        "keyboard_presses": sb.keyboard_presses,
        "app_timeline": [
            {
                "app_name": w.app_name,
                "window_title": w.window_title,
                "timestamp": w.timestamp.isoformat(),
                "duration_seconds": w.duration_seconds,
            }
            for w in sb.app_timeline
        ],
        "recent_metrics": metrics,
    }


def cmd_session_intensity(args):
    """Show intensity score for a session."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    session_id = args.session_id
    session = mgr.get_session(session_id)

    if not session:
        print(f"Session not found: {session_id}")
        return

    if not session.signals:
        print(f"No signals recorded for session {session_id}")
        return

    # Get metrics for different time windows
    windows = [60, 300, 600]  # 1min, 5min, 10min
    
    print(f"\nIntensity Score for Session {session_id} ({session.name})")
    print("=" * 80)
    
    for window in windows:
        metrics = session.signals.metrics_since(window)
        intensity = metrics.get("intensity", 0)
        keys = metrics.get("keys", 0)
        clicks = metrics.get("clicks", 0)
        mouse_dist = metrics.get("mouse_distance", 0)
        app_changes = metrics.get("app_changes", 0)
        
        window_label = f"{window}s" if window < 60 else f"{window//60}min"
        print(f"\nLast {window_label}:")
        print(f"  Intensity: {intensity}/100")
        print(f"  Activity: {keys} keys, {clicks} clicks, {mouse_dist:.0f}px moved, {app_changes} app changes")
    
    # Overall aggregates
    print(f"\nTotal (entire session):")
    print(f"  Keys: {session.signals.keyboard_presses}")
    print(f"  Clicks: {session.signals.mouse_clicks}")
    print(f"  Mouse distance: {session.signals.mouse_distance:.0f}px")
    print(f"  App changes: {len(session.signals.app_timeline)}")
    print(f"  Duration: {session.duration_seconds():.1f}s")
    
    if getattr(args, "json", False):
        import json
        data = {
            "session_id": session_id,
            "name": session.name,
            "windows": {}
        }
        for window in windows:
            metrics = session.signals.metrics_since(window)
            data["windows"][f"{window}s"] = metrics
        data["total"] = {
            "keyboard_presses": session.signals.keyboard_presses,
            "mouse_clicks": session.signals.mouse_clicks,
            "mouse_distance": session.signals.mouse_distance,
            "app_timeline_count": len(session.signals.app_timeline),
            "duration_seconds": session.duration_seconds()
        }
        print("\n" + json.dumps(data, indent=2))


def cmd_session_inspect(args):
    """Inspect a session, including live/recorded signal data."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    session_id = args.session_id
    session = mgr.get_session(session_id)

    if not session:
        print(f"Session not found: {session_id}")
        return

    active = mgr.gate.active_session_id == session_id
    signals = _session_signals_dict(session)

    data = {
        "session_id": session.session_id,
        "name": session.name,
        "status": "ACTIVE" if session.is_active() else "ENDED",
        "created_at": session.created_at.isoformat(),
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "duration_seconds": session.duration_seconds(),
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "state": t.state,
                "duration_seconds": t.duration_seconds(),
            }
            for t in session.tasks.values()
        ],
        "active_session_gate": active,
        "signals": signals,
    }

    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return

    print(f"Session {session.session_id} ({session.name}) - {'ACTIVE' if active else 'ENDED'}")
    print(f"  Created: {session.created_at.isoformat()}")
    print(f"  Started: {session.started_at.isoformat() if session.started_at else '-'}")
    print(f"  Ended:   {session.ended_at.isoformat() if session.ended_at else '-'}")
    print(f"  Duration: {session.duration_seconds():.1f}s")
    print(f"  Tasks: {len(session.tasks)}")
    for t in session.tasks.values():
        print(f"    - {t.task_id}: {t.name} [{t.state}] {t.duration_seconds():.1f}s")

    if not signals:
        print("  Signals: none recorded yet")
        return

    print("  Signals (aggregate):")
    print(f"    Keys: {signals['keyboard_presses']} | Clicks: {signals['mouse_clicks']} | Move px: {signals['mouse_distance']}")
    recent = signals.get("recent_metrics", {})
    print(
        f"    Recent (last {recent.get('window_seconds', 300)}s): "
        f"keys={recent.get('keys', 0)}, clicks={recent.get('clicks', 0)}, "
        f"mouse={recent.get('mouse_distance', 0)}, app_changes={recent.get('app_changes', 0)}, "
        f"active_app={recent.get('active_app')}"
    )
    if signals.get("app_timeline"):
        print("    App timeline:")
        for w in signals["app_timeline"]:
            print(
                f"      - {w['timestamp']}: {w['app_name']} | {w['window_title']} "
                f"(for {w['duration_seconds']:.1f}s)"
            )


# =====================================================
# NEW: Task v2 commands (Session-bound)
# =====================================================

def cmd_task_create_v2(args):
    """Create a task within a session."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    session_id = args.session_id
    name = args.name
    activity_type = getattr(args, "activity_type", "HYBRID")
    try:
        task = mgr.create_task(session_id=session_id, name=name, activity_type=activity_type)
        print(
            f"Created task: {task.task_id} ('{task.name}') in session {session_id} "
            f"[{task.activity_type}]"
        )
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "session_id": session_id,
                        "name": name,
                        "activity_type": task.activity_type,
                    }
                )
            )
    except Exception as e:
        print(f"Error: {e}")


def cmd_task_start_v2(args):
    """Start a task (mark it active in its session)."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    task_id = args.task_id
    try:
        task = mgr.start_task(task_id)
        print(f"Started task: {task_id} ('{task.name}')")
        if getattr(args, "json", False):
            print(json.dumps({"task_id": task_id, "session_id": task.session_id, "state": "ACTIVE"}))
    except ValueError as e:
        if "not active" in str(e).lower():
            print(f"Error: Session is not active. Start or resume the session first.")
        else:
            print(f"Error: {e}")


def cmd_task_end_v2(args):
    """End a task."""
    from agent.session.manager_v2 import SessionManager
    mgr = SessionManager()
    task_id = args.task_id
    try:
        task = mgr.end_task(task_id)
        print(f"Ended task: {task_id}")
        if getattr(args, "json", False):
            print(json.dumps({"task_id": task_id, "session_id": task.session_id, "state": "COMPLETED", "duration_seconds": task.duration_seconds()}))
    except Exception as e:
        print(f"Error: {e}")


def _parse_date_arg(s: str):
    # accepts YYYY-MM-DD or ISO
    from datetime import datetime
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            raise ValueError("Invalid date format; use YYYY-MM-DD or ISO")


def cmd_task_stats(args):
    """Show detailed stats for a single task or all tasks in a date range."""
    sessions = load_sessions()
    name = getattr(args, "name", None)
    from_dt = _parse_date_arg(args.from_date) if getattr(args, "from_date", None) else None
    to_dt = _parse_date_arg(args.to_date) if getattr(args, "to_date", None) else None

    # aggregate per task
    per_task = {}
    for s in sessions:
        # skip sessions outside range
        if from_dt and s.end < from_dt:
            continue
        if to_dt and s.start > to_dt:
            continue

        # use intent_segments if present
        if getattr(s, "intent_segments", None):
            for intent_id, st, en in s.intent_segments:
                st = st or s.start
                en = en or s.end
                if from_dt and en < from_dt:
                    continue
                if to_dt and st > to_dt:
                    continue
                if name and intent_id != name:
                    continue
                dur = int((en - st).total_seconds())
                entry = per_task.setdefault(intent_id, {"time":0, "sessions":0, "keys":0, "clicks":0})
                entry["time"] += dur
                entry["sessions"] += 1
                # approximate input allocation by distributing timeline buckets evenly
                # fallback: use session totals if precise per-segment data missing
                entry["keys"] += int(s.input_events.get("keys",0) * (dur / max(1, (s.end - s.start).total_seconds())))
                entry["clicks"] += int(s.input_events.get("clicks",0) * (dur / max(1, (s.end - s.start).total_seconds())))
        else:
            # fallback: attribute whole session to None or current intent via resolver
            from agent.intent.attribution import resolve_session_attribution
            breakdown = resolve_session_attribution(s, os.path.join("agent","intent","intents.json"))
            for intent_id, seconds in breakdown.items():
                if name and intent_id != name:
                    continue
                entry = per_task.setdefault(intent_id, {"time":0, "sessions":0, "keys":0, "clicks":0})
                entry["time"] += seconds
                entry["sessions"] += 1

    # print result
    if name:
        entry = per_task.get(name)
        if not entry:
            print(f"No data for task: {name}")
            return
        print(f"Task: {name}")
        print(f"  Time: {entry['time']//60}m ({entry['time']}s)")
        print(f"  Sessions: {entry['sessions']}")
        print(f"  Keys (approx): {entry.get('keys',0)}")
        print(f"  Clicks (approx): {entry.get('clicks',0)}")
    else:
        print("Task stats summary:")
        for intent_id, entry in sorted(per_task.items(), key=lambda x: x[1]['time'], reverse=True):
            label = 'Unattributed' if intent_id is None else intent_id
            print(f"  {label}: {entry['time']//60}m across {entry['sessions']} sessions")


def cmd_analyze_gaps(args):
    """Find gaps in intent coverage longer than threshold (minutes)."""
    sessions = load_sessions()
    threshold = int(getattr(args, 'minutes', 30))
    gaps = []
    # build timeline of all intent-covered intervals
    intervals = []
    for s in sessions:
        if getattr(s, 'intent_segments', None):
            for intent_id, st, en in s.intent_segments:
                st = st or s.start
                en = en or s.end
                intervals.append((st, en))
        else:
            # fallback: assume entire session unattributed (no intervals)
            pass
    if not intervals:
        print("No intent segments found in sessions")
        return
    intervals.sort()
    # merge and find gaps
    from datetime import timedelta
    merged = []
    cur_s, cur_e = intervals[0]
    for s,e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    for i in range(len(merged)-1):
        gap_s = merged[i][1]
        gap_e = merged[i+1][0]
        gap_seconds = (gap_e - gap_s).total_seconds()
        if gap_seconds >= threshold*60:
            gaps.append((gap_s, gap_e, int(gap_seconds)))

    if not gaps:
        print(f"No gaps >= {threshold} minutes found")
    else:
        print(f"Gaps >= {threshold} minutes:")
        for s,e,sec in gaps:
            print(f"  {s.isoformat()} -> {e.isoformat()} | {sec//60}m")


def cmd_task_report(args):
    """Produce a day-style report filtered to a single task (intent)."""
    sessions = load_sessions()
    name = getattr(args, 'name')
    # determine target date
    target_date = None
    if getattr(args, 'date', None):
        try:
            target_date = datetime.fromisoformat(args.date).date()
        except Exception:
            print("Invalid date format; use YYYY-MM-DD")
            return
    else:
        target_date = date.today()

    # aggregate metrics focused on this task for the target_date
    total_time = 0
    sessions_count = 0
    apps = {}
    input_totals = {"keys":0, "clicks":0, "mouse_distance":0}

    for s in sessions:
        # examine intent segments
        segs = getattr(s, 'intent_segments', []) or []
        for intent_id, st, en in segs:
            if intent_id != name:
                continue

            # Determine overlap of this segment with the target date
            st = st or s.start
            en = en or s.end

            try:
                seg_start_date = st.date()
                seg_end_date = en.date() if en else s.end.date()
            except Exception:
                # skip malformed segments
                continue

            # If segment doesn't overlap target_date, skip
            if seg_start_date > target_date or seg_end_date < target_date:
                continue

            # Compute actual overlap interval for accurate durations
            # Bound start to target_date at 00:00:00 and end to target_date at 23:59:59
            day_start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1)

            seg_st = st if st >= day_start else day_start
            seg_en = en if en and en < day_end else day_end

            dur = int((seg_en - seg_st).total_seconds())
            if dur <= 0:
                continue

            total_time += dur
            sessions_count += 1

            # collect apps touched during this segment --- approximate by attributing full session apps
            for a in getattr(s, 'apps', []):
                apps[a] = apps.get(a, 0) + dur

            # collect input from timeline buckets overlapping the segment
            for bucket_str, vals in getattr(s, 'timeline', {}).items():
                try:
                    # Parse bucket timestamp from ISO format
                    bucket = datetime.fromisoformat(bucket_str) if isinstance(bucket_str, str) else bucket_str
                    # Make sure both are timezone-aware for comparison
                    if bucket.tzinfo is None:
                        bucket = bucket.replace(tzinfo=timezone.utc)
                    if st.tzinfo is None:
                        st = st.replace(tzinfo=timezone.utc)
                    if en is not None and en.tzinfo is None:
                        en = en.replace(tzinfo=timezone.utc)

                    # bucket must fall within the segment overlap we've computed
                    if bucket >= seg_st and bucket < seg_en:
                        for k in input_totals.keys():
                            input_totals[k] += vals.get(k, 0)
                except Exception:
                    pass  # Skip if there's any issue with bucket comparison

    # prepare output
    is_json = getattr(args, 'json', False)
    out = {
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {
            "task": name,
            "date": str(target_date),
            "total_time_seconds": total_time,
            "sessions_count": sessions_count,
            "apps": apps,
            "input_totals": input_totals,
        }
    }

    if is_json:
        print(json.dumps(out, default=str))
        return

    # human-readable
    print(f"TASK REPORT — {name} — {target_date}")
    if total_time == 0:
        print("  No data for this task on this date")
        return
    print(f"  Total time: {total_time//60}m ({total_time}s)")
    print(f"  Sessions: {sessions_count}")
    print("\n[App usage]")
    if apps:
        for a, t in sorted(apps.items(), key=lambda x: x[1], reverse=True):
            print(f"  {a}: {t//60}m")
    else:
        print("  (no apps recorded)")
    print("\n[Input events]")
    print(f"  Keys: {input_totals['keys']}")
    print(f"  Clicks: {input_totals['clicks']}")
    print(f"  Mouse distance: {input_totals['mouse_distance']} px")


def cmd_report_insights(args):
    """Generate analytics & insights from intervals, segments, and tasks."""
    from agent.analytics.insights import generate_insights_report
    from datetime import datetime

    start_time = None
    end_time = None

    if getattr(args, 'from_date', None):
        try:
            start_time = _parse_date_arg(args.from_date)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"Invalid --from: {e}")
            return

    if getattr(args, 'to_date', None):
        try:
            end_time = _parse_date_arg(args.to_date)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"Invalid --to: {e}")
            return

    report = generate_insights_report(
        start_time=start_time,
        end_time=end_time,
        limit=getattr(args, 'limit', 5000),
        clustering_method=getattr(args, 'clustering', 'dbscan'),
        focus_threshold=getattr(args, 'focus_threshold', 0.6),
        audio_engagement_threshold=getattr(args, 'audio_threshold', 0.5),
    )

    if getattr(args, 'json', False):
        print(json.dumps(report, default=str))
        return

    data = report.get('data', {})
    if 'message' in data:
        print(data['message'])
        return

    print("INSIGHTS REPORT")
    print("=" * 80)

    rng = data.get('range', {})
    print(f"Intervals: {rng.get('interval_count', 0)} | Segments: {rng.get('segment_count', 0)} | Tasks: {rng.get('task_count', 0)}")

    prod = data.get('productivity', {})
    print(f"Focus minutes: {prod.get('focus_minutes', 0):.1f} / Active minutes: {prod.get('active_minutes', 0):.1f}")
    print(f"Focus ratio: {prod.get('focus_ratio', 0):.2f} | Avg focus score: {prod.get('avg_focus_score', 0):.2f}")

    ctx = data.get('context_switching', {})
    print(f"Context switches: {ctx.get('total_switches', 0)}")

    inter = data.get('interaction', {})
    print(f"Copy/paste density: {inter.get('copy_paste_density', 0):.2f} events/min")

    corr = data.get('correlations', {})
    print(f"Mic-focus correlation: {corr.get('mic_focus_correlation')}")
    print(f"Camera-focus correlation: {corr.get('camera_focus_correlation')}")
    print(f"Audio-focus correlation: {corr.get('audio_focus_correlation')}")

    sysres = data.get('system_resources', {})
    heavy = sysres.get('heavy_computation', {})
    print(f"Heavy computation tasks: {heavy.get('task_count', 0)} | segments: {heavy.get('segment_count', 0)}")

    av = data.get('audio_video', {})
    print(f"Meeting periods: {len(av.get('meeting_periods', []))}")
    audio_eng = av.get('audio_engagement', {})
    print(f"Audio engagement (>{audio_eng.get('threshold', 0)}): {audio_eng.get('engaged_ratio', 0):.2f}")


def cmd_task_delete(args):
    mgr = IntentManager()
    sessions = load_sessions()
    if getattr(args, "all", False):
        mgr.clear_tasks()
        for s in sessions:
            s.intent_segments = []
        save_sessions(sessions)
        print("Deleted all tasks and cleared intent segments from sessions")
        return

    name = getattr(args, "name", None)
    if not name:
        print("Provide a task name to delete or use --all")
        return
    try:
        mgr.delete_task(name)
    except Exception:
        pass
    # remove segments referencing this task
    changed = False
    for s in sessions:
        segs = getattr(s, "intent_segments", []) or []
        new_segs = [seg for seg in segs if seg[0] != name]
        if len(new_segs) != len(segs):
            s.intent_segments = new_segs
            changed = True
    if changed:
        save_sessions(sessions)
    print(f"Deleted task {name} and removed its intent segments from sessions")

def cmd_input_query(args):
    """Query input events (keys/clicks/mouse_distance) filtered by time range, task, or session."""
    sessions = load_sessions()
    # parse time range
    from_date = None
    to_date = None
    if getattr(args, 'from_date', None):
        try:
            from_date = _parse_date_arg(args.from_date)
            # ensure timezone aware
            if from_date.tzinfo is None:
                from_date = from_date.replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"Invalid from date: {e}")
            return
    if getattr(args, 'to_date', None):
        try:
            to_date = _parse_date_arg(args.to_date)
            # ensure timezone aware
            if to_date.tzinfo is None:
                to_date = to_date.replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"Invalid to date: {e}")
            return

    # restrict to a single session if requested
    target_sessions = []
    if getattr(args, 'session', None):
        idx = args.session - 1
        if idx < 0 or idx >= len(sessions):
            print("Invalid session index")
            return
        target_sessions = [sessions[idx]]
    else:
        target_sessions = sessions

    task_filter = getattr(args, 'task', None)
    type_filter = getattr(args, 'type', None)

    # aggregate totals and per-bucket timeline
    total = 0
    timeline = {}

    for s in target_sessions:
        # check if we're filtering by time or task
        in_time_range = True
        if from_date or to_date:
            # normalize datetimes to UTC for comparison
            s_start = s.start
            s_end = s.end
            if s_start and s_start.tzinfo is None:
                s_start = s_start.replace(tzinfo=timezone.utc)
            if s_end and s_end.tzinfo is None:
                s_end = s_end.replace(tzinfo=timezone.utc)
            # session must overlap time range
            if to_date and s_start > to_date:
                in_time_range = False
            if from_date and s_end < from_date:
                in_time_range = False
        if not in_time_range:
            continue

        # build task intervals for this session if task_filter set
        task_intervals = []
        if task_filter and getattr(s, 'intent_segments', None):
            for intent_id, st, en in s.intent_segments:
                if intent_id == task_filter:
                    task_intervals.append((st or s.start, en or s.end))
        
        # if task filter is set but no matching segments, skip this session
        if task_filter and not task_intervals:
            continue

        # iterate timeline buckets if available
        timeline_data = getattr(s, 'timeline', {}) or {}
        if timeline_data:
            for bucket, vals in timeline_data.items():
                # bucket is a datetime
                if from_date and bucket < from_date:
                    continue
                if to_date and bucket > to_date:
                    continue
                # if task filter, only include buckets overlapping any task interval
                if task_filter:
                    ok = False
                    for ist, ien in task_intervals:
                        if ist <= bucket < ien:
                            ok = True
                            break
                    if not ok:
                        continue
                # pick value type
                if type_filter:
                    v = vals.get(type_filter, 0)
                else:
                    # sum all
                    v = sum(vals.get(k, 0) for k in ("keys","clicks","mouse_distance"))
                if v:
                    total += v
                    timeline[bucket] = timeline.get(bucket, 0) + v
        else:
            # no timeline data; fall back to session-level input_events if no task filter
            if not task_filter:
                input_events = getattr(s, 'input_events', {})
                if type_filter:
                    total += input_events.get(type_filter, 0)
                else:
                    total += sum(input_events.get(k, 0) for k in ("keys","clicks","mouse_distance"))

    print("Input query result:")
    print(f"  Total: {total}")
    if timeline:
        print("  Timeline (per-minute buckets):")
        for b in sorted(timeline.keys()):
            print(f"    {b.isoformat()}: {timeline[b]}")
    elif total > 0:
        print("  (Data aggregated from session-level input_events)")


def cmd_session_stats(args):
    sessions = load_sessions()
    summary = aggregate_daily_summary(sessions, target_date=datetime.now().date())
    print(format_daily_report(summary))


def cmd_report_daily(args):
    sessions = load_sessions()
    summaries = []
    # group by date
    from collections import defaultdict
    groups = defaultdict(list)
    for s in sessions:
        try:
            d = s.start.date()
        except Exception:
            d = datetime.fromisoformat(s.start).date() if isinstance(s.start, str) else datetime.now().date()
        groups[d].append(s)
    for d in sorted(groups.keys()):
        summaries.append(aggregate_daily_summary(groups[d], target_date=d))
    if summaries:
        print(format_daily_report(summaries[-1]))
    else:
        print("No sessions available for daily report")


def cmd_report_show(args):
    """Flexible report viewer: show sessions, daily, trends, or combination."""
    sessions = load_sessions()

    show_sessions = getattr(args, "sessions", False)
    show_daily = getattr(args, "daily", False)
    show_trends = getattr(args, "trends", False)
    list_sessions = getattr(args, "list_sessions", False)

    # If nothing specified, show all
    if not (show_sessions or show_daily or show_trends):
        show_sessions = show_daily = show_trends = True

    # Sessions summary
    if show_sessions:
        print(f"Loaded {len(sessions)} sessions from disk")
        if list_sessions:
            for i, s in enumerate(sessions):
                try:
                    st = s.start.isoformat()
                    en = s.end.isoformat()
                except Exception:
                    st = str(getattr(s, 'start', None))
                    en = str(getattr(s, 'end', None))
                print(f"{i+1}. {st} -> {en} | apps={list(getattr(s,'apps',[]))} | segments={getattr(s,'intent_segments',[])}")

    # Daily report
    if show_daily:
        # choose date if provided
        if getattr(args, 'date', None):
            try:
                from datetime import datetime
                target_date = datetime.fromisoformat(args.date).date()
            except Exception:
                print("Invalid date format; use YYYY-MM-DD")
                target_date = None
        else:
            from datetime import date
            target_date = date.today()

        # group sessions by date and print the requested day's summary
        from collections import defaultdict
        groups = defaultdict(list)
        for s in sessions:
            try:
                d = s.start.date()
            except Exception:
                try:
                    d = datetime.fromisoformat(s.start).date()
                except Exception:
                    continue
            groups[d].append(s)

        if target_date in groups:
            summary = aggregate_daily_summary(groups[target_date], target_date=target_date)
            print(format_daily_report(summary))
        else:
            print(f"No sessions for {target_date}")

    # Trends
    if show_trends:
        # build daily summaries across all available dates
        from collections import defaultdict
        groups = defaultdict(list)
        for s in sessions:
            try:
                d = s.start.date()
            except Exception:
                continue
            groups[d].append(s)
        daily = [aggregate_daily_summary(groups[d], target_date=d) for d in sorted(groups.keys())]
        if daily:
            today = daily[-1]
            metrics = compute_trend_metrics(today, daily)
            print(format_trend_report(metrics))
        else:
            print("Not enough data for trends")


def cmd_report_trends(args):
    sessions = load_sessions()
    # build daily summaries
    from collections import defaultdict
    groups = defaultdict(list)
    for s in sessions:
        try:
            d = s.start.date()
        except Exception:
            d = datetime.now().date()
        groups[d].append(s)
    daily = [aggregate_daily_summary(groups[d], target_date=d) for d in sorted(groups.keys())]
    if not daily:
        print("Not enough data for trends")
        return
    today = daily[-1]
    metrics = compute_trend_metrics(today, daily)
    print(format_trend_report(metrics))


# =====================================================
# Predictive Intelligence Commands
# =====================================================

def cmd_predict_estimate(args):
    """Estimate completion time for a task."""
    from agent.analytics.behavioral_model import BehavioralModel
    from agent.analytics.predictor import PredictiveIntelligence
    from agent.analytics.persistence import load_behavioral_model_state
    
    task_id = args.task_id
    current_duration = args.current_duration
    current_hour = args.hour
    
    # Load behavioral model
    model_data = load_behavioral_model_state()
    if not model_data:
        print("No behavioral model data available. Need historical data to make predictions.")
        return
    
    model = BehavioralModel.deserialize(model_data)
    predictor = PredictiveIntelligence(behavioral_model=model)
    
    # Get estimate
    estimate = predictor.estimate_completion(task_id, current_duration, current_hour)
    
    if not estimate:
        print(f"Cannot estimate completion for task '{task_id}' - insufficient historical data (need 3+ sessions)")
        return
    
    if getattr(args, 'json', False):
        import json
        from dataclasses import asdict
        print(json.dumps({'version': '1.0', 'estimate': asdict(estimate)}, indent=2))
    else:
        print(f"Task: {estimate.task_id}")
        print(f"Estimated time remaining: {estimate.estimated_minutes_remaining:.1f} minutes")
        if estimate.estimated_completion_time:
            print(f"Estimated completion: {estimate.estimated_completion_time}")
        print(f"Confidence: {estimate.confidence:.2f}")
        print(f"Reason: {estimate.reason}")


def cmd_predict_risks(args):
    """Detect risks for a task or schedule."""
    from agent.analytics.behavioral_model import BehavioralModel
    from agent.analytics.predictor import PredictiveIntelligence
    from agent.analytics.persistence import load_behavioral_model_state
    
    task_id = args.task_id if hasattr(args, 'task_id') else None
    current_hour = args.hour if hasattr(args, 'hour') else None
    
    # Load behavioral model
    model_data = load_behavioral_model_state()
    if not model_data:
        print("No behavioral model data available.")
        return
    
    model = BehavioralModel.deserialize(model_data)
    predictor = PredictiveIntelligence(behavioral_model=model)
    
    risks = []
    
    # Task-specific risks
    if task_id and hasattr(args, 'duration'):
        metrics = {
            'duration_minutes': args.duration,
            'focus_continuity': getattr(args, 'continuity', 0.5),
            'apps': getattr(args, 'apps', '').split(',') if hasattr(args, 'apps') and args.apps else []
        }
        risks = predictor.detect_task_risks(task_id, metrics)
    
    # Schedule risks
    elif current_hour is not None:
        continuity = getattr(args, 'continuity', 0.5)
        risks = predictor.detect_schedule_risks(current_hour, continuity)
    
    if not risks:
        print("No risks detected" if task_id or current_hour else "Please specify --task-id or --hour")
        return
    
    if getattr(args, 'json', False):
        import json
        from dataclasses import asdict
        risk_data = [asdict(r) for r in risks]
        # Convert enum to string
        for r in risk_data:
            r['risk_level'] = r['risk_level'].value
        print(json.dumps({'version': '1.0', 'risks': risk_data}, indent=2))
    else:
        print(f"Detected {len(risks)} risk(s):")
        for i, risk in enumerate(risks, 1):
            print(f"\n{i}. [{risk.risk_level.value.upper()}] {risk.category}")
            print(f"   {risk.message}")
            print(f"   Recommendation: {risk.recommendation}")


def cmd_predict_bottlenecks(args):
    """Detect bottlenecks (stuck tasks, systemic issues)."""
    from agent.analytics.behavioral_model import BehavioralModel
    from agent.analytics.predictor import PredictiveIntelligence
    from agent.analytics.persistence import load_behavioral_model_state, load_sessions
    from datetime import datetime, timezone, timedelta
    
    # Load behavioral model
    model_data = load_behavioral_model_state()
    if not model_data:
        print("No behavioral model data available.")
        return
    
    model = BehavioralModel.deserialize(model_data)
    predictor = PredictiveIntelligence(behavioral_model=model)
    
    bottlenecks = []
    
    # Check for stuck tasks
    if hasattr(args, 'check_stuck') and args.check_stuck:
        from agent.intent.manager import IntentManager
        mgr = IntentManager()
        for task_id, task in mgr.tasks.items():
            if task.state in ["ACTIVE", "IN_PROGRESS"]:
                # Check last activity (simplified - using created_at for demo)
                last_activity = task.created_at
                bottleneck = predictor.detect_stuck_task(task_id, last_activity, task.state)
                if bottleneck:
                    bottlenecks.append(bottleneck)
    
    # Check for systemic bottlenecks
    if hasattr(args, 'check_systemic') and args.check_systemic:
        sessions = load_sessions()
        lookback_days = getattr(args, 'lookback_days', 7)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        recent_sessions = []
        for s in sessions:
            if s.start >= cutoff:
                # Convert to dict format expected by predictor
                session_dict = {
                    'hour': s.start.hour,
                    'focus_continuity': getattr(s, 'focus_continuity', 0.5),
                    'task_id': getattr(s, 'task_id', 'unknown')
                }
                recent_sessions.append(session_dict)
        
        systemic = predictor.detect_systemic_bottlenecks(recent_sessions, lookback_days)
        bottlenecks.extend(systemic)
    
    # Get all tracked bottlenecks if --active flag
    if hasattr(args, 'active') and args.active:
        bottlenecks = predictor.get_all_bottlenecks()
    
    if getattr(args, 'json', False):
        import json
        from dataclasses import asdict
        bottleneck_data = [asdict(b) for b in bottlenecks]
        # Convert enums to strings
        for b in bottleneck_data:
            if 'risk_level' in b and hasattr(b['risk_level'], 'value'):
                b['risk_level'] = b['risk_level'].value
            if 'severity' in b and hasattr(b['severity'], 'value'):
                b['severity'] = b['severity'].value
        print(json.dumps({'version': '1.0', 'bottlenecks': bottleneck_data}, indent=2))
        return
    
    if not bottlenecks:
        print("No bottlenecks detected")
        return
    else:
        print(f"Detected {len(bottlenecks)} bottleneck(s):")
        for i, bn in enumerate(bottlenecks, 1):
            print(f"\n{i}. [{bn.severity.value.upper()}] {bn.type}")
            print(f"   {bn.description}")
            print(f"   Affected tasks: {', '.join(bn.affected_tasks[:3])}" + ("..." if len(bn.affected_tasks) > 3 else ""))
            print(f"   Recommendation: {bn.recommendation}")


def cmd_predict_workload(args):
    """Estimate daily workload for planned tasks."""
    from agent.analytics.behavioral_model import BehavioralModel
    from agent.analytics.predictor import PredictiveIntelligence
    from agent.analytics.persistence import load_behavioral_model_state
    
    task_ids = args.task_ids.split(',') if args.task_ids else []
    current_hour = args.hour if hasattr(args, 'hour') else 9
    
    if not task_ids:
        print("Please provide task IDs via --task-ids (comma-separated)")
        return
    
    # Load behavioral model
    model_data = load_behavioral_model_state()
    if not model_data:
        print("No behavioral model data available.")
        return
    
    model = BehavioralModel.deserialize(model_data)
    predictor = PredictiveIntelligence(behavioral_model=model)
    
    workload = predictor.estimate_daily_workload(task_ids, current_hour)
    
    if not workload:
        print("Cannot estimate workload - insufficient data")
        return
    
    if getattr(args, 'json', False):
        import json
        print(json.dumps({'version': '1.0', 'workload': workload}, indent=2))
    else:
        print(f"Daily Workload Estimate:")
        print(f"Total time: {workload['total_hours']:.2f} hours ({workload['total_minutes']:.0f} minutes)")
        print(f"Feasible: {'Yes' if workload['feasible'] else 'No'}")
        print(f"Utilization: {workload['utilization']*100:.0f}%")
        
        if workload['insufficient_data']:
            print(f"\nTasks with insufficient data: {', '.join(workload['insufficient_data'])}")
        
        print(f"\nTask breakdown:")
        for est in workload['task_estimates']:
            print(f"  - {est['task_id']}: {est['estimated_minutes']:.0f}m (confidence: {est['confidence']:.2f})")


def cmd_agent_start(args):
    """(DEPRECATED) Agent is now embedded in the UI and starts automatically."""
    print("[INFO] Agent is now integrated into the UI.")
    print("The agent will start automatically when you launch the UI and stop when you close it.")
    print("There is no separate background process to manage.")


def cmd_agent_stop(args):
    """(DEPRECATED) Agent is now embedded in the UI."""
    print("[INFO] Agent is now integrated into the UI.")
    print("To stop the agent, simply close the UI window.")


def cmd_agent_status(args):
    """(DEPRECATED) Agent is now embedded in the UI."""
    print("[INFO] Agent is now integrated into the UI.")
    print("The agent runs while the UI is open. Close the UI to stop the agent.")


def cmd_agent_collection_status(args):
    """Check if data collection has started (engagement detected)."""
    if not is_running():
        print("Agent is not running - data collection is inactive")
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT timestamp FROM events WHERE event_type = 'COLLECTION_START' ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if row:
            start_time = row[0]
            print(f"✓ Data collection is ACTIVE")
            print(f"  Started at: {start_time}")
            
            # Count events collected
            cursor.execute(
                "SELECT COUNT(*) FROM events WHERE timestamp >= ?",
                (start_time,)
            )
            count_row = cursor.fetchone()
            count = count_row[0] if count_row else 0
            print(f"  Events collected: {count}")
        else:
            print("✗ Data collection NOT started - waiting for engagement")
            print("  Requirements: 30s warmup, 5 inputs, 10s stable window")
    finally:
        conn.close()


def cmd_state_current(args):
    """Show the latest inferred state (read-only)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT timestamp, payload FROM events WHERE event_type = 'STATE_CHANGE' ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
    except Exception:
        row = None
    finally:
        conn.close()

    if not row:
        print("Current State: unknown")
        return

    ts, payload_str = row
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except Exception:
        payload = {"raw": payload_str}

    state_to = payload.get("to") or "unknown"
    active_task = payload.get("active_task")
    active_app = payload.get("active_app")
    is_idle = payload.get("is_idle")
    session_active = payload.get("session_active")
    intensity = payload.get("intensity") if payload.get("intensity") is not None else "N/A"

    print(f"Current State: {state_to}")
    print(f"Intensity: {intensity}")
    print(f"Active Task: {active_task}")
    print(f"Active App: {active_app}")
    print(f"Idle: {is_idle}")
    print(f"Session Active: {session_active}")


def cmd_state_history(args):
    """Show recent STATE_CHANGE timeline from the events DB."""
    last = int(getattr(args, "last", 10) or 10)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT timestamp, payload FROM events WHERE event_type = 'STATE_CHANGE' ORDER BY timestamp DESC LIMIT ?",
            (last,)
        )
        rows = cursor.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    if not rows:
        print("No state history available")
        return

    # rows are newest first; reverse for chronological
    entries = []
    for ts_str, payload_str in reversed(rows):
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        try:
            payload = json.loads(payload_str) if payload_str else {}
        except Exception:
            payload = {"raw": payload_str}
        entries.append((ts, payload))

    now = datetime.now(timezone.utc)
    for i, (ts, payload) in enumerate(entries):
        state = payload.get("to") or "unknown"
        start = ts
        end = entries[i + 1][0] if i + 1 < len(entries) else now
        dur = int((end - start).total_seconds())
        # Pretty print: STATE  HH:MM → HH:MM (Xm)
        start_s = start.astimezone().strftime("%H:%M")
        end_s = end.astimezone().strftime("%H:%M") if end != now else "now"
        mins = dur // 60
        print(f"{state:16} {start_s} → {end_s} ({mins}m)")


def cmd_ui_now(args):
    """Emit current state as a UI-ready JSON object (no formatting)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT timestamp, payload FROM events WHERE event_type = 'STATE_CHANGE' ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
    except Exception:
        row = None
    finally:
        conn.close()

    if not row:
        # No STATE_CHANGE found. If the background agent isn't running, try a
        # single in-process evaluation to generate a state (helps when running
        # the UI locally without a separate agent process).
        try:
            from agent.process_manager import is_running
            if not is_running():
                try:
                    # Ensure DB initialized and run one evaluation
                    from agent.storage.db import init_db
                    from agent.inference.runner import InferenceRunner
                    init_db()
                    runner = InferenceRunner()
                    runner.evaluate_once()
                    # Re-query the DB for a new STATE_CHANGE event
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT timestamp, payload FROM events WHERE event_type = 'STATE_CHANGE' ORDER BY timestamp DESC LIMIT 1")
                    row = cursor.fetchone()
                    conn.close()
                except Exception:
                    row = None
        except Exception:
            pass

    if not row:
        print(json.dumps({"timestamp": None, "state": "unknown", "explanation": ["no STATE_CHANGE events"]}))
        return

    ts, payload_str = row
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except Exception:
        payload = {"raw": payload_str}

    out = {
        "version": "1.0",
        "timestamp": ts,
        "data": {
            "state": payload.get("to") or "unknown",
            "intensity": payload.get("intensity") if payload.get("intensity") is not None else "N/A",
            "active_task": payload.get("active_task"),
            "active_app": payload.get("active_app"),
            "session_active": payload.get("session_active"),
            "explanation": [payload.get("reason")] if payload.get("reason") else [],
        },
    }
    if getattr(args, "json", False):
        print(json.dumps(out, default=str))
    else:
        d = out["data"]
        print(f"Current State: {d['state']}")
        print(f"Intensity: {d['intensity']}")
        print(f"Active Task: {d['active_task']}")
        print(f"Active App: {d['active_app']}")
        print(f"Session Active: {d['session_active']}")
        print(f"Explanation: {d['explanation']}")


def cmd_ui_dump(args):
    """Emit the UI contract for a date; supports --json for machine consumption."""
    try:
        target_date = __import__('datetime').date.fromisoformat(args.date)
    except Exception:
        print("Invalid date format; use YYYY-MM-DD")
        return

    try:
        contract = dump_ui_contract_for_date(target_date)
    except Exception as e:
        if getattr(args, 'json', False):
            print(json.dumps({"version": "1.0", "error": True, "reason": str(e)}))
        else:
            print(f"Error building UI contract: {e}")
        return

    out = {"version": "1.1", "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), "data": contract}
    if getattr(args, 'json', False):
        print(json.dumps(out, default=str))
    else:
        # pretty print summary
        print("UI contract summary:")
        print(f"  segments: {len(contract.get('timeline', []))}")
        print(f"  summary: {contract.get('summary')}")


def cmd_ui_llm_query(args):
    """Execute a planner-validated analytics query produced by an LLM."""
    import json
    payload = None

    if getattr(args, "question", None):
        try:
            from agent.ui.llm_connector import orchestrate_question
            out = orchestrate_question(args.question)
            out["version"] = 1
            if getattr(args, "json", False):
                print(json.dumps(out, default=str))
            else:
                if not out.get("ok"):
                    print("Errors:")
                    for err in out.get("errors", []):
                        print(f"- {err}")
                    return
                print(json.dumps(out, indent=2, default=str))
            return
        except Exception as e:
            print(json.dumps({"ok": False, "errors": [str(e)], "version": 1}))
            return

    if args.payload is None:
        print(json.dumps({"ok": False, "errors": ["payload or --question required"], "version": 1}))
        return

    try:
        payload = json.loads(args.payload)
    except Exception as e:
        print(json.dumps({"ok": False, "errors": [f"invalid JSON: {e}"], "version": 1}))
        return

    try:
        from agent.ui import query_planner
        result = query_planner.plan(payload)
        out = {"version": 1, "ok": result.ok, "errors": result.errors, "data": result.data}
        if getattr(args, "json", False):
            print(json.dumps(out, default=str))
        else:
            if not result.ok:
                print("Planner errors:")
                for err in result.errors:
                    print(f"- {err}")
                return
            print(json.dumps(out, indent=2, default=str))
    except Exception as e:
        print(json.dumps({"ok": False, "errors": [str(e)], "version": 1}))


def cmd_ui_live_task(args):
    """Get current live task prediction from the most recent minute of activity."""
    try:
        from agent.session.gate import get_session_gate
        from agent.storage.db import get_latest_live_prediction
        
        gate = get_session_gate()
        
        # Check if there's an active session
        if not gate.is_active():
            out = {
                "version": "1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "task_id": None,
                    "confidence": None,
                    "distance_to_centroid": None,
                    "reason": None,
                    "session_active": False,
                    "message": "No active session",
                },
            }
            if getattr(args, "json", False):
                print(json.dumps(out, default=str))
            else:
                print("No active session - no live task prediction available")
            return
        
        # Get active session ID
        try:
            session_id = gate.get_active_session_id()
        except Exception:
            session_id = None
        
        if not session_id:
            out = {
                "version": "1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "task_id": None,
                    "confidence": None,
                    "distance_to_centroid": None,
                    "reason": None,
                    "session_active": False,
                    "message": "Could not get active session ID",
                },
            }
            if getattr(args, "json", False):
                print(json.dumps(out, default=str))
            else:
                print("Could not determine active session")
            return
        
        # Get latest live prediction for this session
        prediction = get_latest_live_prediction(session_id)
        
        out = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "task_id": prediction["task_id"] if prediction else None,
                "confidence": float(prediction["confidence"]) if prediction else None,
                "distance_to_centroid": float(prediction["distance_to_centroid"]) if prediction else None,
                "reason": prediction["reason"] if prediction else None,
                "prediction_timestamp": prediction["timestamp"] if prediction else None,
                "session_active": True,
                "session_id": session_id,
            },
        }
        
        if getattr(args, "json", False):
            print(json.dumps(out, default=str))
        else:
            d = out["data"]
            if d["task_id"]:
                print(f"Live Task Prediction: {d['task_id']}")
                print(f"Confidence: {d['confidence']:.2%}" if d['confidence'] is not None else "Confidence: N/A")
                print(f"Distance to Centroid: {d['distance_to_centroid']:.3f}" if d['distance_to_centroid'] is not None else "Distance: N/A")
                print(f"Reason: {d['reason']}" if d['reason'] else "Reason: N/A")
                print(f"Prediction Time: {d['prediction_timestamp']}" if d['prediction_timestamp'] else "")
            else:
                print("No live task prediction yet (still collecting data)")
    
    except Exception as e:
        out = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "error": str(e),
            },
        }
        if getattr(args, "json", False):
            print(json.dumps(out, default=str))
        else:
            print(f"Error getting live task prediction: {e}")


def cmd_check_consistency(args):
    """Run numerical consistency checks for a given date."""
    try:
        target_date = __import__('datetime').date.fromisoformat(args.date)
    except Exception:
        print("Invalid date format; use YYYY-MM-DD")
        return

    # load contract
    try:
        contract = dump_ui_contract_for_date(target_date)
    except Exception as e:
        print(f"Error building contract: {e}")
        return

    # compute timeline seconds
    timeline = contract.get('timeline', [])
    seg_secs = 0
    overlaps = False
    last_end = None
    for seg in timeline:
        try:
            s = datetime.fromisoformat(seg['start'])
            e = datetime.fromisoformat(seg['end'])
        except Exception:
            continue
        if last_end and s < last_end:
            overlaps = True
        last_end = e
        seg_secs += int((e - s).total_seconds())

    # compute session seconds
    sessions = load_sessions()
    day_start = datetime.combine(target_date, __import__('datetime').datetime.min.time()).replace(tzinfo=timezone.utc)
    day_end = day_start + __import__('datetime').timedelta(days=1)
    sess_secs = 0
    for s in sessions:
        try:
            s_start = s.start
            s_end = s.end
        except Exception:
            continue
        if s_start is None or s_end is None:
            continue
        if s_start.tzinfo is None:
            s_start = s_start.replace(tzinfo=timezone.utc)
        if s_end.tzinfo is None:
            s_end = s_end.replace(tzinfo=timezone.utc)
        if s_end <= day_start or s_start >= day_end:
            continue
        seg_start = max(s_start, day_start)
        seg_end = min(s_end, day_end)
        sess_secs += int((seg_end - seg_start).total_seconds())

    ok = True
    msgs = []
    if abs(sess_secs - seg_secs) > 60:
        ok = False
        msgs.append(f"Timeline total ({seg_secs}s) differs from session total ({sess_secs}s)")
    if overlaps:
        ok = False
        msgs.append("Found overlapping timeline segments")
    if seg_secs < 0 or sess_secs < 0:
        ok = False
        msgs.append("Negative durations found")

    out = {"version": "1.0", "date": str(target_date), "ok": ok, "messages": msgs, "session_seconds": sess_secs, "timeline_seconds": seg_secs}
    if getattr(args, 'json', False):
        print(json.dumps(out, default=str))
    else:
        print("Consistency check result:")
        print(f"  ok: {ok}")
        for m in msgs:
            print(f"  - {m}")
        print(f"  session_seconds: {sess_secs}")
        print(f"  timeline_seconds: {seg_secs}")


def ensure_agent_running():
    """Ensure the agent is running; auto-start if not."""
    if not is_running():
        pm_start()


def cmd_reset(args):
    """Factory reset: remove all user data (DB, JSON snapshots, runtime files, intents)."""
    force = getattr(args, 'yes', False) or getattr(args, 'force', False)

    if not force:
        print("WARNING: This will permanently delete all user data (DB, sessions, intents, runtime files).")
        resp = input("Type 'yes' to confirm: ")
        if resp.strip().lower() != 'yes':
            print("Aborting reset.")
            return

    failures = []

    # Stop agent if running
    try:
        if is_running():
            ok, msg = pm_stop()
            print(f"Stopped agent: {msg}")
    except Exception as e:
        failures.append(f"stopping agent: {e}")

    # Paths to remove
    repo_root = Path(__file__).resolve().parent
    files = [
        repo_root / 'sessions.json',
        repo_root / 'sessions.backup.json',
        Path(__file__).resolve().parent / 'storage' / 'events.db',
        Path(__file__).resolve().parent / 'intent' / 'intents.json',
    ]

    for p in files:
        try:
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                print(f"Removed: {p}")
        except Exception as e:
            failures.append(f"removing {p}: {e}")

    # Remove runtime directory (~/.productivity_agent)
    try:
        from agent.process_manager import RUNTIME_DIR
        if RUNTIME_DIR.exists():
            # Try stopping the agent again to release any held files
            try:
                if is_running():
                    ok, msg = pm_stop()
                    print(f"Stopped agent during reset: {msg}")
            except Exception:
                pass

            # Attempt to remove files inside the runtime dir individually with fallbacks
            try:
                for child in list(RUNTIME_DIR.iterdir()):
                    try:
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            try:
                                child.unlink()
                            except Exception:
                                # Try truncating the file and unlinking again
                                try:
                                    with open(child, 'w'):
                                        pass
                                    child.unlink()
                                except Exception:
                                    # Last resort: try to identify processes holding the file,
                                    # terminate them, and retry removal/rename (Windows handles)
                                    retried = False
                                    try:
                                        import psutil
                                        holders = []
                                        for p in psutil.process_iter(['pid', 'name']):
                                            try:
                                                for of in p.open_files():
                                                    if Path(of.path) == child:
                                                        holders.append(p)
                                                        break
                                            except Exception:
                                                continue

                                        if holders:
                                            for p in holders:
                                                try:
                                                    p.terminate()
                                                except Exception:
                                                    pass
                                            # wait briefly for processes to exit
                                            import time
                                            time.sleep(1)
                                            for p in holders:
                                                try:
                                                    if p.is_running():
                                                        p.kill()
                                                except Exception:
                                                    pass

                                            # retry unlink
                                            try:
                                                child.unlink()
                                                retried = True
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                    if not retried:
                                        # Try renaming as a fallback
                                        try:
                                            new_name = child.with_suffix(child.suffix + '.old')
                                            child.rename(new_name)
                                            retried = True
                                        except Exception as e:
                                            failures.append(f"removing runtime file {child}: {e}")
                    except Exception as e:
                        failures.append(f"removing runtime file {child}: {e}")

                # Finally attempt to remove the directory itself
                try:
                    RUNTIME_DIR.rmdir()
                    print(f"Removed runtime dir: {RUNTIME_DIR}")
                except Exception as e:
                    failures.append(f"removing runtime dir: {e}")
            except Exception as e:
                failures.append(f"removing runtime dir contents: {e}")
    except Exception as e:
        failures.append(f"removing runtime dir: {e}")

    if failures:
        print("Reset completed with errors:")
        for f in failures:
            print(f" - {f}")
    else:
        print("Factory reset complete. All user data removed.")


def main():
    # Auto-start agent for most commands (except agent status)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "agent":
        ensure_agent_running()
    
    parser = argparse.ArgumentParser(description="Productivity agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    # task
    task_p = subparsers.add_parser("task")
    task_sp = task_p.add_subparsers(dest="sub")
    task_sp.add_parser("list").set_defaults(func=cmd_task_list)
    start_p = task_sp.add_parser("start")
    start_p.add_argument("name")
    start_p.set_defaults(func=cmd_task_start)
    pause_p = task_sp.add_parser("pause")
    pause_p.add_argument("name")
    pause_p.set_defaults(func=cmd_task_pause)
    stop_p = task_sp.add_parser("stop")
    stop_p.add_argument("name")
    stop_p.set_defaults(func=cmd_task_stop)
    task_sp.add_parser("summary").set_defaults(func=cmd_task_summary)
    # unstable tasks command
    unstable_p = task_sp.add_parser("unstable")
    unstable_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    unstable_p.set_defaults(func=cmd_task_unstable)
    # delete command
    delete_p = task_sp.add_parser("delete")
    delete_p.add_argument("name", nargs="?", default=None)
    delete_p.add_argument("--all", action="store_true")
    delete_p.set_defaults(func=cmd_task_delete)
    # stats command
    stats_p = task_sp.add_parser("stats")
    stats_p.add_argument("name", nargs="?", default=None)
    stats_p.add_argument("--from", dest="from_date", help="Start date (YYYY-MM-DD)")
    stats_p.add_argument("--to", dest="to_date", help="End date (YYYY-MM-DD)")
    stats_p.set_defaults(func=cmd_task_stats)
    # task report (per-task daily-style report)
    report_p = task_sp.add_parser("report")
    report_p.add_argument("name")
    report_p.add_argument("--date", help="Target date (YYYY-MM-DD)")
    report_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    report_p.set_defaults(func=cmd_task_report)

    # session
    sess_p = subparsers.add_parser("session")
    sess_sp = sess_p.add_subparsers(dest="sub")
    sess_list_p = sess_sp.add_parser("list")
    sess_list_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    sess_list_p.set_defaults(func=cmd_session_list)
    sess_sp.add_parser("stats").set_defaults(func=cmd_session_stats)
    
    # NEW: Session v2 commands (session-centric model)
    create_sess_p = sess_sp.add_parser("create")
    create_sess_p.add_argument("name", help="Session name")
    create_sess_p.add_argument("--session-id", help="Custom session ID (optional)")
    create_sess_p.add_argument("--json", action="store_true")
    create_sess_p.set_defaults(func=cmd_session_create)
    
    start_sess_p = sess_sp.add_parser("start")
    start_sess_p.add_argument("session_id", help="Session ID to start")
    start_sess_p.add_argument("--json", action="store_true")
    start_sess_p.set_defaults(func=cmd_session_start)
    
    pause_sess_p = sess_sp.add_parser("pause")
    pause_sess_p.add_argument("session_id", help="Session ID to pause")
    pause_sess_p.add_argument("--json", action="store_true")
    pause_sess_p.set_defaults(func=cmd_session_pause)
    
    resume_sess_p = sess_sp.add_parser("resume")
    resume_sess_p.add_argument("session_id", help="Session ID to resume")
    resume_sess_p.add_argument("--json", action="store_true")
    resume_sess_p.set_defaults(func=cmd_session_resume)
    
    end_sess_p = sess_sp.add_parser("end")
    end_sess_p.add_argument("session_id", help="Session ID to end")
    end_sess_p.add_argument("--json", action="store_true")
    end_sess_p.set_defaults(func=cmd_session_end)
    
    info_sess_p = sess_sp.add_parser("info")
    info_sess_p.add_argument("session_id", help="Session ID to get info on")
    info_sess_p.set_defaults(func=cmd_session_info)

    inspect_sess_p = sess_sp.add_parser("inspect")
    inspect_sess_p.add_argument("session_id", help="Session ID to inspect (includes signals)")
    inspect_sess_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    inspect_sess_p.set_defaults(func=cmd_session_inspect)
    
    intensity_sess_p = sess_sp.add_parser("intensity")
    intensity_sess_p.add_argument("session_id", help="Session ID to check intensity")
    intensity_sess_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    intensity_sess_p.set_defaults(func=cmd_session_intensity)
    
    # NEW: Task v2 commands (session-bound)
    task_create_p = sess_sp.add_parser("task-create")
    task_create_p.add_argument("session_id", help="Session ID")
    task_create_p.add_argument("name", help="Task name")
    task_create_p.add_argument(
        "--activity-type",
        dest="activity_type",
        choices=["PASSIVE", "ACTIVE", "HYBRID"],
        default="HYBRID",
        help="Task activity type (default: HYBRID)",
    )
    task_create_p.add_argument("--json", action="store_true")
    task_create_p.set_defaults(func=cmd_task_create_v2)
    
    task_start_p = sess_sp.add_parser("task-start")
    task_start_p.add_argument("task_id", help="Task ID")
    task_start_p.add_argument("--json", action="store_true")
    task_start_p.set_defaults(func=cmd_task_start_v2)
    
    task_end_p = sess_sp.add_parser("task-end")
    task_end_p.add_argument("task_id", help="Task ID")
    task_end_p.add_argument("--json", action="store_true")
    task_end_p.set_defaults(func=cmd_task_end_v2)

    # analysis
    analy_p = subparsers.add_parser("analyze")
    analy_sp = analy_p.add_subparsers(dest="sub")
    gaps_p = analy_sp.add_parser("gaps")
    gaps_p.add_argument("--minutes", type=int, default=30)
    gaps_p.set_defaults(func=cmd_analyze_gaps)

    # input queries
    input_p = subparsers.add_parser("input")
    input_sp = input_p.add_subparsers(dest="sub")
    iq = input_sp.add_parser("query")
    iq.add_argument("--from", dest="from_date", help="Start datetime (ISO or YYYY-MM-DD)")
    iq.add_argument("--to", dest="to_date", help="End datetime (ISO or YYYY-MM-DD)")
    iq.add_argument("--task", help="Filter by task/intent id")
    iq.add_argument("--session", type=int, help="Filter by session index (1-based from sessions.json)")
    iq.add_argument("--type", choices=["keys","clicks","mouse_distance"], help="Type of input event to query")
    iq.set_defaults(func=cmd_input_query)

    # report
    rep_p = subparsers.add_parser("report")
    rep_sp = rep_p.add_subparsers(dest="sub")
    rep_sp.add_parser("daily").set_defaults(func=cmd_report_daily)
    rep_sp.add_parser("trends").set_defaults(func=cmd_report_trends)
    insights_p = rep_sp.add_parser("insights")
    insights_p.add_argument("--from", dest="from_date", help="Start datetime (ISO or YYYY-MM-DD)")
    insights_p.add_argument("--to", dest="to_date", help="End datetime (ISO or YYYY-MM-DD)")
    insights_p.add_argument("--limit", type=int, default=5000, help="Max intervals to analyze")
    insights_p.add_argument("--clustering", choices=["dbscan", "hierarchical", "time_windowed"], default="dbscan")
    insights_p.add_argument("--focus-threshold", type=float, default=0.6, help="Focus score threshold (0-1)")
    insights_p.add_argument("--audio-threshold", type=float, default=0.5, help="Audio engagement threshold (0-1)")
    insights_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    insights_p.set_defaults(func=cmd_report_insights)
    # flexible show command: sessions/daily/trends (show together or separately)
    show_p = rep_sp.add_parser("show")
    show_p.add_argument("--sessions", action="store_true", help="Show saved sessions summary")
    show_p.add_argument("--list-sessions", action="store_true", help="List saved sessions entries")
    show_p.add_argument("--daily", action="store_true", help="Show daily report for --date (defaults to today)")
    show_p.add_argument("--trends", action="store_true", help="Show trend analysis across available days")
    show_p.add_argument("--date", help="Target date for daily report (YYYY-MM-DD)")
    show_p.set_defaults(func=cmd_report_show)

    # predict (predictive intelligence)
    predict_p = subparsers.add_parser("predict", help="Predictive intelligence (estimates, risks, bottlenecks)")
    predict_sp = predict_p.add_subparsers(dest="sub")
    
    # predict estimate
    est_p = predict_sp.add_parser("estimate", help="Estimate task completion time")
    est_p.add_argument("task_id", help="Task ID")
    est_p.add_argument("--current-duration", type=float, required=True, help="Current duration in minutes")
    est_p.add_argument("--hour", type=int, help="Current hour (0-23) for time-of-day adjustment")
    est_p.add_argument("--json", action="store_true", help="Emit JSON output")
    est_p.set_defaults(func=cmd_predict_estimate)
    
    # predict risks
    risks_p = predict_sp.add_parser("risks", help="Detect task or schedule risks")
    risks_p.add_argument("--task-id", help="Task ID to check")
    risks_p.add_argument("--duration", type=float, help="Current task duration (minutes)")
    risks_p.add_argument("--continuity", type=float, help="Current focus continuity score")
    risks_p.add_argument("--apps", help="Comma-separated list of current apps")
    risks_p.add_argument("--hour", type=int, help="Hour to check schedule risks (0-23)")
    risks_p.add_argument("--json", action="store_true", help="Emit JSON output")
    risks_p.set_defaults(func=cmd_predict_risks)
    
    # predict bottlenecks
    bottle_p = predict_sp.add_parser("bottlenecks", help="Detect bottlenecks (stuck tasks, systemic issues)")
    bottle_p.add_argument("--check-stuck", action="store_true", help="Check for stuck tasks")
    bottle_p.add_argument("--check-systemic", action="store_true", help="Check for systemic bottlenecks")
    bottle_p.add_argument("--active", action="store_true", help="List all active bottlenecks")
    bottle_p.add_argument("--lookback-days", type=int, default=7, help="Days to look back for systemic analysis")
    bottle_p.add_argument("--json", action="store_true", help="Emit JSON output")
    bottle_p.set_defaults(func=cmd_predict_bottlenecks)
    
    # predict workload
    work_p = predict_sp.add_parser("workload", help="Estimate daily workload for planned tasks")
    work_p.add_argument("--task-ids", required=True, help="Comma-separated list of task IDs")
    work_p.add_argument("--hour", type=int, default=9, help="Starting hour (default: 9)")
    work_p.add_argument("--json", action="store_true", help="Emit JSON output")
    work_p.set_defaults(func=cmd_predict_workload)

    # agent (process management)
    agent_p = subparsers.add_parser("agent")
    agent_sp = agent_p.add_subparsers(dest="sub")
    agent_sp.add_parser("start").set_defaults(func=cmd_agent_start)
    agent_sp.add_parser("stop").set_defaults(func=cmd_agent_stop)
    agent_sp.add_parser("status").set_defaults(func=cmd_agent_status)
    agent_sp.add_parser("collection").set_defaults(func=cmd_agent_collection_status)

    # reset (factory reset)
    reset_p = subparsers.add_parser("reset")
    reset_p.add_argument("--yes", action="store_true", help="Confirm reset without prompting")
    reset_p.add_argument("--force", action="store_true", help="Alias for --yes")
    reset_p.set_defaults(func=cmd_reset)

    # state (inference visibility)
    state_p = subparsers.add_parser("state")
    state_sp = state_p.add_subparsers(dest="sub")
    state_sp.add_parser("current").set_defaults(func=cmd_state_current)
    hist_p = state_sp.add_parser("history")
    hist_p.add_argument("--last", type=int, default=10, help="Number of recent entries to show")
    hist_p.set_defaults(func=cmd_state_history)

    # check (consistency and assertions)
    check_p = subparsers.add_parser("check")
    check_sp = check_p.add_subparsers(dest="sub")
    cons_p = check_sp.add_parser("consistency")
    cons_p.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    cons_p.add_argument("--json", action="store_true", help="Emit JSON output")
    cons_p.set_defaults(func=cmd_check_consistency)

    # ui (dump UI-ready JSON contract)
    ui_p = subparsers.add_parser("ui")
    ui_sp = ui_p.add_subparsers(dest="sub")
    ui_dump = ui_sp.add_parser("dump")
    ui_dump.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ui_dump.add_argument("--json", action="store_true", help="Emit machine-readable JSON contract (versioned)")
    ui_dump.set_defaults(func=lambda args: cmd_ui_dump(args))
    ui_now = ui_sp.add_parser("now")
    ui_now.add_argument("--json", action="store_true", help="Emit machine-readable JSON (versioned)")
    ui_now.set_defaults(func=cmd_ui_now)
    ui_live_task = ui_sp.add_parser("live-task")
    ui_live_task.add_argument("--json", action="store_true", help="Emit machine-readable JSON (versioned)")
    ui_live_task.set_defaults(func=cmd_ui_live_task)
    ui_llm = ui_sp.add_parser("llm-query")
    ui_llm.add_argument("payload", nargs="?", help="LLM-produced JSON payload for analytics intent")
    ui_llm.add_argument("--question", help="Natural language question to route via local LLM")
    ui_llm.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    ui_llm.set_defaults(func=cmd_ui_llm_query)

    

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
