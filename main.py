import time
from datetime import datetime, timezone, date, timedelta
from collections import defaultdict
import os
import sys
import signal
import atexit

from agent.storage.db import init_db, get_connection
from agent.logging import log_event
from agent.signals.active_window import get_active_window
from agent.signals.idle import get_idle_seconds
from pathlib import Path
import traceback
from agent.signals.input_engagement import attach_input_hooks_with_engagement
from agent.session.sessionizer import SessionManager
from agent.session.engagement import EngagementDetector
from agent.session.data_collector import DataCollector
from agent.analytics.daily import aggregate_daily_summary, print_daily_report
from agent.analytics.persistence import load_sessions, save_sessions
from agent.analytics.trends import compute_trend_metrics, print_trend_report
from agent.analytics.behavioral_model import BehavioralModel
from agent.analytics.predictor import PredictiveIntelligence
from agent.task.inference import TaskInferenceEngine
from agent.intent.manager import IntentManager
from agent.inference.engine import InferenceEngine, InferenceContext
from agent.inference.state import StateManager
from agent.error_handling import (
    get_error_handler, 
    ComponentType, 
    ErrorSeverity,
    log_component_error,
    handle_critical_failure
)

IDLE_THRESHOLD_SECONDS = 300  # 5 minutes
SUSTAINED_ACTIVITY_THRESHOLD_SECONDS = 60  # 1 minute of sustained activity before starting session

def build_daily_summaries_from_sessions(sessions):
    """
    Build a list of DailySummary objects from sessions.
    Groups sessions by date for multi-day trend analysis.
    """
    sessions_by_date = defaultdict(list)
    
    # Group sessions by date
    for session in sessions:
        session_date = session.start.astimezone().date()
        sessions_by_date[session_date].append(session)
    
    # Aggregate each day
    daily_summaries = []
    for target_date in sorted(sessions_by_date.keys()):
        day_sessions = sessions_by_date[target_date]
        summary = aggregate_daily_summary(day_sessions, target_date=target_date)
        daily_summaries.append(summary)
    
    return daily_summaries

def main(stop_flag=None):
    """Main entry point for the productivity agent.
    
    Args:
        stop_flag: threading.Event that signals when to stop the agent loop
    """
    crash_log = Path(".agent_crash.log")

    def log_crash(prefix: str, exc: Exception):
        try:
            with crash_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} - {prefix}: {exc}\n")
                traceback.print_exc(file=f)
        except Exception:
            pass

    # Add stdout/stderr flushing to ensure logs are written immediately
    import sys
    
    try:
        init_db()
    except Exception as e:
        log_crash("init_db", e)
        return
    
    # Start data maintenance manager (90-day retention)
    try:
        from agent.storage.maintenance import start_maintenance
        start_maintenance(retention_days=90)
        print("[INFO] Data maintenance started (90-day retention)")
    except Exception as e:
        log_crash("init_maintenance", e)
        # Continue even if maintenance fails

    # Initialize engagement detector
    try:
        engagement_detector = EngagementDetector(
            warmup_seconds=SUSTAINED_ACTIVITY_THRESHOLD_SECONDS,  # 1 minute of sustained activity
            input_threshold=8,   # Require at least 8 inputs during 1-minute warmup period
            window_stability_seconds=3  # Window must be stable for 3 seconds
        )
    except Exception as e:
        return
    
    # ML Components: Task Inference, Behavioral Model, Predictive Intelligence
    # Initialize BEFORE SessionManager so we can define finalization callback
    try:
        task_inference_engine = TaskInferenceEngine()
        behavioral_model = BehavioralModel()
        predictive_intelligence = PredictiveIntelligence(
            behavioral_model=behavioral_model,
            task_inference=task_inference_engine
        )
    except Exception as e:
        log_crash("init_ml", e)
        return
    
    def finalize_session_with_ml(session):
        """
        ML finalization callback: task inference, behavioral model update, predictions.
        Called when a session ends to:
        1. Extract feature vector from session
        2. Infer task using centroid-based clustering
        3. Update behavioral model with session data
        4. Generate predictive signals (completion estimates, risk detection)
        """
        try:
            # Import feature extraction helper
            from agent.task.inference import extract_feature_vector
            
            # 1. Extract feature vector from session
            feature_vector = extract_feature_vector(session)
            
            # 2. Task inference (centroid matching)
            task_id, confidence = task_inference_engine.infer_task(session.id, feature_vector)
            if task_id:
                session.inferred_task_id = task_id  # Store task assignment on session
            
            # 3. Behavioral model update
            behavioral_model.update_from_session(session, task_id=task_id)
            
            # 4. Predictive intelligence (only if task assigned)
            if task_id:
                try:
                    # Calculate duration for predictions
                    duration_minutes = (session.end - session.start).total_seconds() / 60.0
                    current_hour = session.start.hour
                    
                    # Generate completion estimate
                    completion_estimate = predictive_intelligence.estimate_completion(
                        task_id, 
                        current_duration_minutes=duration_minutes,
                        current_hour=current_hour
                    )
                    
                    # Check for task risks
                    session_metrics = {
                        "duration_minutes": duration_minutes,
                        "input_per_minute": (session.input_events.get("keys", 0) + session.input_events.get("clicks", 0)) / max(duration_minutes, 1),
                        "context_switches": len(session.apps)
                    }
                    risk_signals = predictive_intelligence.detect_task_risks(task_id, session_metrics)
                    
                except Exception as pred_err:
                    pass
            
        except Exception as e:
            pass
    
    # Session manager (now decoupled from data collection)
    try:
        from agent.session.error_handling import get_error_handler, safe_ml_finalization
        
        error_handler = get_error_handler()
        
        # Wrap ML finalization with error handling
        def safe_finalize_session_with_ml(session):
            return safe_ml_finalization(session, finalize_session_with_ml)
        
        session_manager = SessionManager(
            idle_threshold_seconds=IDLE_THRESHOLD_SECONDS,
            ml_finalization_callback=safe_finalize_session_with_ml
        )
        
        # --- Cleanup old sessions on startup ---
        # DISABLED: Sessions are no longer auto-created or auto-stopped
        # Manual session control only
        # now = datetime.now(timezone.utc)
        # try:
        #     if session_manager.current_session:
        #         current = session_manager.current_session
        #         session_age_hours = (now - current.start).total_seconds() / 3600
        #         if session_age_hours > 24:
        #             # Session is from a previous day - end it automatically
        #             session_manager.end_session_if_active(now, reason="startup_cleanup_old_session")
        # except Exception as cleanup_err:
        #     error_handler.log_error("startup_old_session_cleanup", cleanup_err, critical=False)
        
    except Exception as e:
        log_crash("session_manager_init", e)
        return
    
    # Initialize data collector (independent of sessions)
    # Provides optional session_id_provider for decoupled session interaction:
    # - If session active: events associated with session
    # - If no session: events stored as background/unassigned
    def get_active_session_id():
        """Get current active session_id, or None if no session (background)."""
        gate = getattr(session_manager, "gate", None)
        if gate and gate.is_active():
            try:
                return gate.active_session_id
            except Exception:
                return None
        return None
    
    def notify_session_activity(keys=0, clicks=0, mouse_distance=0):
        """Notify session manager of activity for incremental feature updates."""
        session_manager.update_session_activity(keys, clicks, mouse_distance)
    
    try:
        data_collector = DataCollector(
            log_event, 
            session_id_provider=get_active_session_id,
            activity_callback=notify_session_activity
        )
    except Exception as e:
        return

    # Start interval-based signal collection (persists interval_signals)
    interval_aggregator = None
    try:
        import sys
        sys.stderr.write(f"[MAIN] Starting interval collection with 60s intervals\n")
        sys.stderr.flush()
        
        from agent.signals import interval_aggregator as ia_module
        
        # Reset singleton in case it's already running from previous test
        ia_module._interval_aggregator = None
        
        interval_aggregator = ia_module.start_interval_collection(
            interval_seconds=60.0,
            session_id_provider=get_active_session_id
        )
        sys.stderr.write(f"[MAIN] Interval collection started successfully\n")
        sys.stderr.flush()
    except Exception as e:
        log_crash("interval_collection_start", e)
        import sys
        import traceback
        sys.stderr.write(f"[MAIN] ERROR: Failed to start interval collection: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
    
    # Intent manager loads and reconciles persistent task state
    try:
        intent_manager = IntentManager()
    except Exception as e:
        return

    # Online Task Classification Manager
    # Provides real-time task assignment during active sessions
    session_task_manager = None
    try:
        from agent.task.session_task_manager import SessionTaskManager
        session_task_manager = SessionTaskManager(
            session_manager=session_manager,
            task_inference_engine=task_inference_engine
        )
    except Exception as e:
        log_crash("session_task_manager_init", e)
        session_task_manager = None

    # Inference runner (background)
    from agent.inference.runner import InferenceRunner, set_main_session_manager, set_task_inference_engine, LivePredictionRunner
    inference_runner = None
    live_prediction_runner = None
    recent_activity_runner = None
    try:
        # Pass the session manager to the inference runner
        set_main_session_manager(session_manager)
        # Pass the task inference engine for live predictions
        set_task_inference_engine(task_inference_engine)
        
        inference_runner = InferenceRunner()
        inference_runner.start()
        
        # Start live prediction runner (session-based, every 60 seconds)
        live_prediction_runner = LivePredictionRunner(interval_seconds=60.0, window_seconds=60)
        live_prediction_runner.set_task_inference_engine(task_inference_engine)
        live_prediction_runner.start()
        
        # Start recent activity task runner (activity-based, every 60 seconds)
        from agent.task.recent_activity_runner import RecentActivityTaskRunner
        recent_activity_runner = RecentActivityTaskRunner(interval_seconds=60.0, window_seconds=60)
        recent_activity_runner.set_task_inference_engine(task_inference_engine)
        recent_activity_runner.start()
        
    except Exception as e:
        log_crash("inference_runner_start", e)
        inference_runner = None
        live_prediction_runner = None
        recent_activity_runner = None

    realtime_runner = None
    try:
        from agent.realtime.integration import RealTimeIntegrationRunner
        realtime_runner = RealTimeIntegrationRunner(interval_seconds=60, lookback_minutes=120)
        realtime_runner.start()
    except Exception as e:
        log_crash("realtime_runner_start", e)
    
    # Attach input hooks with engagement detector and data collector
    
    try:
        attach_input_hooks_with_engagement(engagement_detector, data_collector, debug=False)
    except Exception as e:
        log_crash("attach_hooks", e)
        return
    

    last_active_window = None
    is_idle = False
    last_event_replay_time = 0
    REPLAY_INTERVAL = 2
    last_replayed_timestamp = None
    last_session_save_time = time.time()
    SESSION_SAVE_INTERVAL = 10
    last_signal_persist_time = time.time()
    SIGNAL_PERSIST_INTERVAL = 5  # Batch persist input signals every 5 seconds
    last_heartbeat_time = time.time()
    HEARTBEAT_INTERVAL = 5  # Log heartbeat every 5 seconds for debugging
    last_midnight_check = datetime.now(timezone.utc).date()  # Track the last date we checked for midnight rollover
    last_task_classification_time = time.time()
    TASK_CLASSIFICATION_INTERVAL = 10  # Update task classification every 10 seconds
    
    # Cache for expensive system calls to avoid blocking on fast input
    _cached_window = {"value": None, "timestamp": 0}
    _cached_idle = {"value": 0, "timestamp": 0}
    _window_cache_timeout = 0.5  # Cache window for 500ms
    _idle_cache_timeout = 0.2    # Cache idle for 200ms
    
    def get_cached_window():
        """Get cached window info or fetch fresh if cache expired."""
        now = time.time()
        if now - _cached_window["timestamp"] > _window_cache_timeout:
            try:
                _cached_window["value"] = get_active_window()
            except Exception as e:
                log_crash("get_active_window", e)
                _cached_window["value"] = None
            _cached_window["timestamp"] = now
        return _cached_window["value"]
    
    def get_cached_idle():
        """Get cached idle seconds or fetch fresh if cache expired."""
        now = time.time()
        if now - _cached_idle["timestamp"] > _idle_cache_timeout:
            try:
                _cached_idle["value"] = get_idle_seconds()
            except Exception as e:
                log_crash("get_idle_seconds", e)
                _cached_idle["value"] = 0
            _cached_idle["timestamp"] = now
        return _cached_idle["value"]

    def perform_shutdown(reason: str = "agent_stop"):
        """Perform graceful shutdown: end sessions, save state, stop threads."""
        try:
            # Stop the inference runner if running
            try:
                if inference_runner is not None:
                    inference_runner.stop()
                    inference_runner.join(timeout=5)
            except Exception as stop_err:
                log_crash("shutdown_inference_runner", stop_err)
            
            # Stop the live prediction runner
            try:
                if live_prediction_runner is not None:
                    live_prediction_runner.stop()
                    live_prediction_runner.join(timeout=5)
            except Exception as stop_err:
                log_crash("shutdown_live_prediction_runner", stop_err)

            # Stop the realtime integration runner
            try:
                if realtime_runner is not None:
                    realtime_runner.stop()
            except Exception as stop_err:
                log_crash("shutdown_realtime_runner", stop_err)

            # Stop interval signal collection
            try:
                if interval_aggregator is not None:
                    interval_aggregator.stop()
            except Exception as stop_err:
                log_crash("shutdown_interval_aggregator", stop_err)

            # --- End active session on shutdown ---
            shutdown_time = datetime.now(timezone.utc)
            
            # End active session
            try:
                if session_manager.current_session:
                    session_manager.end_session_if_active(shutdown_time, reason="agent_shutdown")
            except Exception as e:
                log_crash("shutdown_end_session", e)

            # --- Save sessions to disk ---
            all_sessions = session_manager.completed_sessions[:]

            if all_sessions:
                try:
                    save_sessions(all_sessions)
                except Exception as e:
                    log_crash("save_sessions_shutdown", e)
            
            # --- Day 5: Analytics & Aggregation Report ---
            
            # Aggregate all sessions by day
            if session_manager.completed_sessions:
                # Build daily summaries for all days (for trend comparison)
                daily_summaries = build_daily_summaries_from_sessions(session_manager.completed_sessions)
                
                # Get today's summary (last in the list)
                today_summary = daily_summaries[-1] if daily_summaries else None
                
                if today_summary:
                    print_daily_report(today_summary)
                    
                    # --- Day 6: Trend & Comparative Analysis ---
                    
                    # Compute and display trends
                    trend_metrics = compute_trend_metrics(today_summary, daily_summaries)
                    print_trend_report(trend_metrics)
        except Exception as e:
            log_crash("perform_shutdown", e)

    try:
        while True:
            # Check stop flag if provided (for thread-based execution)
            if stop_flag and stop_flag.is_set():
                print("[AGENT] Stop signal received, shutting down...", flush=True)
                perform_shutdown(reason="ui_stop")
                break
            try:
                timestamp = datetime.now(timezone.utc)

                # --- DISABLED: Check for midnight rollover and end session from previous day ---
                # current_date = timestamp.date()
                # if current_date > last_midnight_check:
                #     # Midnight has passed - end any active session to prevent multi-day sessions
                #     if session_manager.current_session:
                #         session_manager.end_session_if_active(timestamp, reason="daily_rollover")
                #     last_midnight_check = current_date

                # --- Check engagement status (use cached window) ---
                current_window = get_cached_window()
                if current_window:
                    engagement_detector.record_window_change(
                        current_window.get("process_name", ""),
                        current_window.get("window_title", "")
                    )
                
                # Check if user is engaged
                try:
                    is_engaged = engagement_detector.check_engagement()
                    # Debug: print engagement status occasionally
                    if int(time.time()) % 10 == 0:  # Every 10 seconds
                        num_inputs = len(engagement_detector.input_events)
                        warmup_age = (datetime.now() - engagement_detector.start_time).total_seconds()
                        print(f"[ENGAGEMENT] is_engaged={is_engaged}, inputs={num_inputs}, warmup_age={warmup_age:.0f}s, window={current_window.get('process_name', 'unknown')}", flush=True)
                except Exception as ce_err:
                    log_crash("check_engagement", ce_err)
                    is_engaged = False
                
                # Start data collection when engagement is detected.
                # Ensure a session exists before activating collection so collected
                # device/input events are associated with a session.
                if is_engaged and not data_collector.is_active():
                    # Attempt to start a session first; only start collection if
                    # a session is active (new or existing).
                    new_session = session_manager.start_session_if_needed(timestamp)
                    if new_session or session_manager.current_session:
                        data_collector.start_collection()
                        if new_session:
                            session_id = new_session.id or new_session.session_id
                            print(f"[SESSION] Started new session {session_id} after sustained activity", flush=True)
                
                # Only collect data if engaged
                if not data_collector.is_active():
                    # Not engaged yet - skip data collection
                    time.sleep(1)  # Shorter sleep when idle to be more responsive
                    continue
                    
                # --- Periodically save sessions to disk (for graceful shutdown compatibility) ---
                current_time = time.time()
                
                # Batch persist input signals to reduce I/O overhead
                if current_time - last_signal_persist_time >= SIGNAL_PERSIST_INTERVAL:
                    last_signal_persist_time = current_time
                    if session_manager.current_session:
                        try:
                            # Force persist the session manager state
                            session_manager._persist()
                            session_manager._save_gate_state()
                        except Exception as e:
                            pass  # Silent fail, will retry next interval
                
                # Save all sessions periodically
                if current_time - last_session_save_time >= SESSION_SAVE_INTERVAL:
                    last_session_save_time = current_time
                    # Save all sessions (completed + current)
                    all_sessions = session_manager.completed_sessions[:]
                    if session_manager.current_session:
                        all_sessions.append(session_manager.current_session)
                    if all_sessions:
                        try:
                            save_sessions(all_sessions)
                        except Exception as e:
                            # Silent fail - don't print to avoid spam during heavy input
                            pass

                # --- Update online task classification periodically ---
                # Provides real-time task assignment that appears in the UI
                if current_time - last_task_classification_time >= TASK_CLASSIFICATION_INTERVAL:
                    last_task_classification_time = current_time
                    if session_task_manager and session_manager.current_session:
                        try:
                            session_task_manager.update_active_session_task_assignment(timestamp)
                            print(f"[TASK] Updated task classification for session", flush=True)
                        except Exception as task_err:
                            print(f"[TASK] ERROR updating classification: {task_err}", flush=True)
                            pass  # Silent fail - task classification is optional/non-critical

                # --- Periodically check for new intent events from CLI ---
                # This ensures INTENT_START/INTENT_STOP from CLI are processed
                current_time = time.time()
                if current_time - last_event_replay_time >= REPLAY_INTERVAL:
                    last_event_replay_time = current_time
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        try:
                            # Get all intent events after the last one we processed
                            if last_replayed_timestamp:
                                cursor.execute(
                                    "SELECT timestamp, event_type, payload FROM events WHERE event_type IN ('INTENT_START', 'INTENT_PAUSE', 'INTENT_STOP', 'INTENT_RESUME') AND timestamp > ? ORDER BY timestamp ASC",
                                    (last_replayed_timestamp,)
                                )
                            else:
                                cursor.execute("SELECT timestamp, event_type, payload FROM events WHERE event_type IN ('INTENT_START', 'INTENT_PAUSE', 'INTENT_STOP', 'INTENT_RESUME') ORDER BY timestamp ASC")
                            rows = cursor.fetchall()
                        except Exception:
                            rows = []
                        finally:
                            conn.close()
                        
                        # Process each intent event
                        for ts_str, event_type, payload in rows:
                            try:
                                ts = datetime.fromisoformat(ts_str)
                                last_replayed_timestamp = ts_str
                                
                                pl = None
                                if payload:
                                    try:
                                        import json as _json
                                        pl = _json.loads(payload)
                                    except Exception:
                                        pl = payload
                                
                                if event_type in ("INTENT_START", "INTENT_RESUME", "INTENT_PAUSE", "INTENT_STOP"):
                                    intent_id = pl.get("intent_id") if pl else "unknown"
                                    with open(".agent_intent_debug.log", "a") as df:
                                        df.write(f"{datetime.now().isoformat()} - Processed {event_type} for {intent_id}\n")
                            except Exception as e:
                                with open(".agent_intent_debug.log", "a") as df:
                                    df.write(f"{datetime.now().isoformat()} - ERROR: {e}\n")
                    except Exception as intent_err:
                        pass  # Silent fail - don't print to avoid spam

                # --- Active window tracking (raw signal → normalized CONTEXT_SWITCH) ---
                if current_window and current_window != last_active_window:
                    process_name = current_window.get("process_name", "")
                    window_title = current_window.get("window_title", "")
                    data_collector.collect_window_change(
                        process_name,
                        window_title
                    )
                    last_active_window = current_window
                    
                    # Update session activity with app context
                    session_manager.update_session_activity(app=process_name, window_title=window_title)

                # --- Idle detection (raw signal → normalized IDLE_ENTER/IDLE_EXIT) ---
                idle_seconds = get_cached_idle()

                if not is_idle and idle_seconds >= IDLE_THRESHOLD_SECONDS:
                    # Transition from active to idle
                    data_collector.collect_idle_start(idle_seconds)
                    is_idle = True
                    
                    # Finalize task classification before ending session
                    if session_task_manager and session_manager.current_session:
                        try:
                            current_session_id = session_manager.current_session.id
                            session_task_manager.finalize_session_tasks(current_session_id, timestamp)
                        except Exception as task_err:
                            pass  # Silent fail - task finalization is optional
                    
                    # End session when idle threshold is exceeded (10 minutes)
                    session_manager.end_session_if_active(timestamp, reason="idle_threshold")
                    print(f"[SESSION] Ended session due to {IDLE_THRESHOLD_SECONDS/60:.0f} min idle", flush=True)
                    
                    # Also stop data collection when session ends
                    # This allows a fresh start when engagement is detected again
                    if hasattr(data_collector, 'stop_collection'):
                        data_collector.stop_collection()

                elif is_idle and idle_seconds < IDLE_THRESHOLD_SECONDS:
                    # Transition from idle back to active
                    data_collector.collect_idle_end(idle_seconds)
                    is_idle = False
                    
                    # Start new session when returning from idle
                    # This creates a new session boundary for the resumed work block
                    if is_engaged and data_collector.is_active():
                        print(f"[SESSION] Creating new session on idle->active transition", flush=True)
                        session_manager.start_session_if_needed(timestamp)

                # Inference now runs in background via InferenceRunner; nothing to do here

                time.sleep(0.1)  # Very short sleep to keep responsive, but avoid busy-waiting
                
                # Log heartbeat every N seconds
                if time.time() - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                    last_heartbeat_time = time.time()
                    with open(".agent_heartbeat.log", "a") as hb:
                        hb.write(f"{datetime.now().isoformat()} - HEARTBEAT - loop iteration OK\n")
                
            except Exception as loop_err:
                log_crash("main_loop", loop_err)
                time.sleep(1)
                continue

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] KeyboardInterrupt received, ending session...", flush=True)
        perform_shutdown(reason="keyboard_interrupt")
    
    except Exception as crash_err:
        print(f"\n[SHUTDOWN] Unhandled exception, ending session...", flush=True)
        log_crash("unhandled_exception", crash_err)
        perform_shutdown(reason="agent_crash")
    
    finally:
        # DISABLED: Ensure session is ended even if shutdown wasn't called
        # try:
        #     if session_manager and session_manager.current_session:
        #         print("[SHUTDOWN] Finally block - ensuring session is ended", flush=True)
        #         session_manager.end_session_if_active(
        #             datetime.now(timezone.utc), 
        #             reason="finally_block"
        #         )
        # except Exception as final_err:
        #     print(f"[SHUTDOWN] Error in finally block: {final_err}", flush=True)
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
