"""Intent Manager (Week2 Day2)

Provides a single authoritative manager for task/intents lifecycle,
persistence on every state change, and deterministic recovery rules
on startup.

Lifecycle states: CREATED, ACTIVE, PAUSED, COMPLETED, ABANDONED

This module enforces transitions via command methods only.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, Dict, List

from agent.error_handling import log_component_error, ComponentType, ErrorSeverity


INTENTS_FILE = os.path.join("agent", "intent", "intents.json")


class InvalidTransition(Exception):
    pass


@dataclass
class Task:
    id: str
    state: str
    created_at: str
    started_at: Optional[str] = None
    paused_at: Optional[str] = None
    completed_at: Optional[str] = None
    abandoned_at: Optional[str] = None
    # Ordered list of transition events: {"ts": iso, "state": STATE}
    events: List[Dict] = field(default_factory=list)


class IntentManager:
    STATES = {"CREATED", "ACTIVE", "PAUSED", "COMPLETED", "ABANDONED"}
    TERMINAL = {"COMPLETED", "ABANDONED"}

    def __init__(self, filepath: str = INTENTS_FILE):
        self.filepath = filepath
        self.tasks: Dict[str, Task] = {}
        self._load_and_reconcile()

    # ----------------- Public commands -----------------
    def create_task(self, task_id: str) -> Task:
        if task_id in self.tasks:
            raise ValueError("task already exists")
        now = datetime.now(timezone.utc).isoformat()
        t = Task(id=task_id, state="CREATED", created_at=now)
        t.events.append({"ts": now, "state": "CREATED"})
        self.tasks[task_id] = t
        self._persist()
        return t

    # Compatibility helpers using "intent" terminology (legacy callers / tests)
    def create_intent(self, name: str) -> dict:
        """Create a new intent (compat wrapper). Returns a lightweight dict with id."""
        import uuid
        tid = str(uuid.uuid4())[:8]
        t = self.create_task(tid)
        return {"id": t.id}

    def start_intent(self, intent_id: str):
        """Start an intent (compat wrapper)."""
        return self.start_task(intent_id)

    def start_task(self, task_id: str) -> Task:
        t = self._get_existing(task_id)
        if t.state == "ACTIVE":
            return t
        if t.state in self.TERMINAL:
            raise InvalidTransition(f"Cannot start terminal task {t.state}")

        # Enforce single active: if another active exists, pause it (last-start wins)
        active = self._find_active()
        now = datetime.now(timezone.utc).isoformat()
        if active and active.id != task_id:
            # Pause the currently active task and record event
            active.state = "PAUSED"
            active.paused_at = now
            active.events.append({"ts": now, "state": "PAUSED"})

        # Start the requested task
        t.state = "ACTIVE"
        t.started_at = now
        t.paused_at = None
        t.events.append({"ts": now, "state": "ACTIVE"})
        self._persist()
        return t

    def pause_task(self, task_id: str) -> Task:
        t = self._get_existing(task_id)
        if t.state != "ACTIVE":
            raise InvalidTransition("Can only pause ACTIVE tasks")
        now = datetime.now(timezone.utc).isoformat()
        t.state = "PAUSED"
        t.paused_at = now
        t.events.append({"ts": now, "state": "PAUSED"})
        self._persist()
        return t

    def resume_task(self, task_id: str) -> Task:
        t = self._get_existing(task_id)
        if t.state != "PAUSED":
            raise InvalidTransition("Can only resume PAUSED tasks")
        # prevent multiple actives: pause any other active task (last-start wins)
        active = self._find_active()
        now = datetime.now(timezone.utc).isoformat()
        if active and active.id != task_id:
            active.state = "PAUSED"
            active.paused_at = now
            active.events.append({"ts": now, "state": "PAUSED"})

        t.state = "ACTIVE"
        t.started_at = t.started_at or now
        t.paused_at = None
        t.events.append({"ts": now, "state": "ACTIVE"})
        self._persist()
        return t

    def complete_task(self, task_id: str) -> Task:
        t = self._get_existing(task_id)
        if t.state == "COMPLETED":
            return t
        if t.state in {"CREATED"}:
            raise InvalidTransition("Cannot complete a CREATED task directly")
        now = datetime.now(timezone.utc).isoformat()
        t.state = "COMPLETED"
        t.completed_at = now
        t.events.append({"ts": now, "state": "COMPLETED"})
        self._persist()
        return t

    def abandon_task(self, task_id: str) -> Task:
        t = self._get_existing(task_id)
        if t.state in self.TERMINAL:
            return t
        now = datetime.now(timezone.utc).isoformat()
        t.state = "ABANDONED"
        t.abandoned_at = now
        t.events.append({"ts": now, "state": "ABANDONED"})
        self._persist()
        return t

    def delete_task(self, task_id: str) -> None:
        """Remove a task from manager and persist change."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._persist()

    def clear_tasks(self) -> None:
        """Remove all tasks."""
        self.tasks = {}
        self._persist()

    # ----------------- Helpers -----------------
    def _get_existing(self, task_id: str) -> Task:
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task: {task_id}")
        return self.tasks[task_id]

    def _find_active(self) -> Optional[Task]:
        for t in self.tasks.values():
            if t.state == "ACTIVE":
                return t
        return None

    def get_active_task_id(self) -> Optional[str]:
        """Public helper to return the id of the currently ACTIVE task, or None."""
        t = self._find_active()
        return t.id if t else None

    # ----------------- Persistence & Recovery -----------------
    def _persist(self) -> None:
        # Atomic write: write to temp file then replace
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                payload = {
                    "version": 1,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "tasks": {tid: asdict(t) for tid, t in self.tasks.items()},
                }
                json.dump(payload, f, indent=2)
            # Try to atomically replace the target file. On Windows this can
            # fail if another process has the file open or it is read-only.
            # Attempt a small retry loop and try to relax permissions on the
            # target file before the final attempt.
            import time
            attempts = 3
            for attempt in range(1, attempts + 1):
                try:
                    os.replace(tmp_path, self.filepath)
                    break
                except PermissionError as e:
                    if attempt == attempts:
                        raise
                    # Try to make target writable if it exists, then retry
                    try:
                        if os.path.exists(self.filepath):
                            try:
                                os.chmod(self.filepath, 0o666)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    time.sleep(0.2 * attempt)
                except Exception:
                    # re-raise other exceptions
                    raise
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _load_and_reconcile(self) -> None:
        # Load tasks and apply deterministic reconciliation rules
        if not os.path.exists(self.filepath):
            self.tasks = {}
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_tasks = data.get("tasks", {})
            tasks: Dict[str, Task] = {}
            for tid, td in raw_tasks.items():
                tasks[tid] = Task(**td)
            self.tasks = tasks
        except Exception as e:
            log_component_error(
                ComponentType.INTENT,
                "load_intents",
                e,
                ErrorSeverity.ERROR,
                filepath=self.filepath
            )
            print(f"Error loading intents file: {e}")
            self.tasks = {}

        # Reconciliation rules:
        # - If multiple ACTIVE tasks, make the one with latest started_at ACTIVE, pause others
        # - If ACTIVE without started_at -> pause it
        try:
            active_tasks = [t for t in self.tasks.values() if t.state == "ACTIVE"]
            if active_tasks:
                # filter out those lacking started_at
                valid_started = [t for t in active_tasks if t.started_at]
                if not valid_started:
                    # no valid started times -> pause all
                    for t in active_tasks:
                        print(f"Reconcile: pausing ACTIVE task without started_at: {t.id}")
                        t.state = "PAUSED"
                        t.paused_at = datetime.now(timezone.utc).isoformat()
                    self._persist()
                    return

                # choose last-started wins
                last = max(valid_started, key=lambda t: t.started_at)
                for t in active_tasks:
                    if t.id == last.id:
                        # ensure it has started_at
                        if not t.started_at:
                            t.started_at = datetime.now(timezone.utc).isoformat()
                    else:
                        print(f"Reconcile: pausing extra ACTIVE task {t.id}; last-started wins ({last.id})")
                        t.state = "PAUSED"
                        t.paused_at = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            log_component_error(
                ComponentType.INTENT,
                "reconcile_intents",
                e,
                ErrorSeverity.WARNING,
                task_count=len(self.tasks)
            )
            self._persist()
