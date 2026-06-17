"""
Task model: first-class work block entity tied to sessions.

A task represents explicit user-initiated work within a session.
Cannot exist outside a session.
Exactly one task can be active per session at a time.
"""
from datetime import datetime
from typing import Optional
import uuid


class Task:
    """
    Represents a unit of work within a session.
    
    Fields:
      id: Unique identifier
      session_id: Foreign key to parent session (must exist)
      name: User-provided task name
      start_time: When user started the task
      end_time: When user stopped the task (None if active)
    """
    
    def __init__(
        self, 
        task_id: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ):
        self.id = task_id or str(uuid.uuid4())
        self.session_id = session_id or ""  # Must be set before task is used
        self.name = name or ""
        self.start_time = start_time
        self.end_time = end_time  # None while active
    
    def is_active(self) -> bool:
        """True if task is currently running (no end_time)."""
        return self.end_time is None and self.start_time is not None
    
    def duration_seconds(self) -> int:
        """Return duration in seconds. If active, returns 0 (use current time in caller)."""
        if not self.start_time or not self.end_time:
            return 0
        return int((self.end_time - self.start_time).total_seconds())
    
    def __repr__(self):
        status = "ACTIVE" if self.is_active() else "CLOSED"
        return f"Task({self.id[:8]}, {self.name}, {status})"


class TaskManager:
    """
    Manages tasks for a single active session.
    
    Invariants:
      - At most one active task per session
      - All tasks must have a session_id
      - Tasks cannot be created without an active session
    """
    
    def __init__(self):
        self.tasks: dict[str, Task] = {}  # id -> Task
        self.active_task_id: str | None = None
    
    def create_task(self, name: str, session_id: str, start_time: datetime) -> Task:
        """
        Create a new task for the given session.
        
        Precondition: session_id must exist (caller's responsibility)
        Effect: creates task, sets as active if none exists
        """
        task = Task(
            session_id=session_id,
            name=name,
            start_time=start_time,
        )
        self.tasks[task.id] = task
        
        if self.active_task_id is None:
            self.active_task_id = task.id
        
        return task
    
    def start_task(self, task_id: str, start_time: datetime) -> Task:
        """
        Mark a task as active (stopped it being paused or created).
        
        Precondition: task exists, no other task is active
        """
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task: {task_id}")
        
        if self.active_task_id is not None and self.active_task_id != task_id:
            # Pause previous task
            prev = self.tasks[self.active_task_id]
            if prev.is_active():
                prev.end_time = start_time
        
        task = self.tasks[task_id]
        task.start_time = start_time
        task.end_time = None
        self.active_task_id = task_id
        return task
    
    def end_task(self, task_id: str, end_time: datetime) -> Task:
        """
        Stop a task from running.
        
        Precondition: task exists and is active
        """
        if task_id not in self.tasks:
            raise KeyError(f"Unknown task: {task_id}")
        
        task = self.tasks[task_id]
        task.end_time = end_time
        
        if self.active_task_id == task_id:
            self.active_task_id = None
        
        return task
    
    def get_active_task(self) -> Task | None:
        """Return the currently active task, or None."""
        if self.active_task_id is None:
            return None
        return self.tasks.get(self.active_task_id)
    
    def delete_task(self, task_id: str) -> None:
        """Remove a task entirely."""
        if task_id in self.tasks:
            if self.active_task_id == task_id:
                self.active_task_id = None
            del self.tasks[task_id]
