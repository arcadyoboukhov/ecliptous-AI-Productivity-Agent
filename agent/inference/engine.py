from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agent.inference.state import ActivityState
from agent.error_handling import log_component_error, ComponentType, ErrorSeverity


@dataclass
class InferenceContext:
    timestamp: datetime
    is_idle: bool
    active_app: Optional[str]
    input_activity_score: float
    active_task: Optional[str]
    session_active: bool


class InferenceEngine:
    """Naive inference engine (Day 1).

    Behavior is intentionally minimal and fully stateless: decision
    is made only from the current context.
    """

    def evaluate(self, ctx: InferenceContext) -> ActivityState:
        try:
            if ctx.is_idle:
                return ActivityState.IDLE

            # If there's no active task, consider activity to be drifting (unaligned)
            if ctx.active_task is None:
                return ActivityState.ACTIVE_UNALIGNED

            return ActivityState.ACTIVE_ALIGNED
        except Exception as e:
            log_component_error(
                ComponentType.INFERENCE,
                "evaluate",
                e,
                ErrorSeverity.WARNING,
                context=ctx
            )
            # Safe default: return IDLE to avoid making bad inferences
            return ActivityState.IDLE
