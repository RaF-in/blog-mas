"""Ch10 §1.3 — Celery worker package.

The worker pool is the engine room of the production architecture.  Each
worker process runs the same glass-box Context Engine logic (Planner, Agents,
Executor) that runs locally via the CLI.  The only difference is that workers
pull goals from the task queue rather than receiving them interactively.

Ch10: "When a worker becomes available, it pulls the goal from the queue and
begins execution."
"""
