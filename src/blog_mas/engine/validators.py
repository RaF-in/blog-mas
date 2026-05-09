"""Structural plan validation — fail fast before execution."""


def validate_plan(plan: list, registry) -> None:
    """Raise ValueError on the first malformed step. Otherwise return None.

    Checks:
    - plan is a non-empty list of dicts
    - each step has integer 'step' (1-indexed, sequential), string 'agent', dict 'input'
    - 'agent' exists in registry
    - any $$STEP_N_OUTPUT$$ reference points to a strictly earlier step
    """
    if not isinstance(plan, list) or not plan:
        raise ValueError("Plan must be a non-empty list.")

    for i, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i}: must be a dict.")
        if step.get("step") != i:
            raise ValueError(f"Step {i}: 'step' must equal {i} (got {step.get('step')!r}).")
        agent = step.get("agent")
        if not isinstance(agent, str) or not registry.has(agent):
            raise ValueError(f"Step {i}: unknown agent {agent!r}.")
        inp = step.get("input")
        if not isinstance(inp, dict):
            raise ValueError(f"Step {i}: 'input' must be a dict.")
        _check_refs(i, inp)


def _check_refs(step_num: int, value) -> None:
    if isinstance(value, str) and value.startswith("$$STEP_") and value.endswith("_OUTPUT$$"):
        try:
            ref_num = int(value[len("$$STEP_"):-len("_OUTPUT$$")])
        except ValueError as e:
            raise ValueError(f"Step {step_num}: malformed reference {value!r}.") from e
        if ref_num >= step_num:
            raise ValueError(
                f"Step {step_num}: forward/self reference to step {ref_num}."
            )
    elif isinstance(value, dict):
        for v in value.values():
            _check_refs(step_num, v)
    elif isinstance(value, list):
        for v in value:
            _check_refs(step_num, v)
