"""Execution trace recorder: captures plan, steps, status, and timing."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path


class ExecutionTrace:
    """Records the full plan-execute-reflect lifecycle for a single goal."""

    def __init__(self, goal: str):
        self.trace_id = str(uuid.uuid4())
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.started_at = datetime.utcnow().isoformat()
        self.start_time = time.time()
        self.duration = None
        # Ch10 §1.3 — queue/worker correlation metadata injected by the task layer.
        # Callers set this after construction; engine code ignores it.
        self.metadata: dict = {}

    def log_plan(self, plan):
        self.plan = plan
        self.status = "Running"

    def log_step(self, step_num, agent, planned_input, resolved_input, output):
        self.steps.append({
            "step": step_num,
            "agent": agent,
            "planned_input": planned_input,
            "resolved_context": resolved_input,
            "output": output,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def finalize(self, status: str, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "plan": self.plan,
            "steps": self.steps,
            "status": self.status,
            "final_output": self.final_output,
            "started_at": self.started_at,
            "duration_seconds": self.duration,
            # Ch10 §1.3 — correlation metadata (task_id, worker_id, queue).
            "metadata": self.metadata,
        }

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{self.trace_id}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return out
