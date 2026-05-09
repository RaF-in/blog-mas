"""Context chaining: resolve $$STEP_N_OUTPUT$$ and $$STEP_N_OUTPUT$$.field references."""

import copy


def resolve_dependencies(input_params: dict, state: dict) -> dict:
    """Replace $$REF$$ placeholders in input_params with values from state.

    Supports two forms:
    - Whole-string: "$$STEP_1_OUTPUT$$" → state["STEP_1_OUTPUT"]
    - Dotted path: "$$STEP_1_OUTPUT$$.blog_spec" → state["STEP_1_OUTPUT"].blog_spec
      (traverses dict keys or object attributes via dot notation)

    Uses copy.deepcopy so the original plan is never mutated.
    Raises ValueError if a referenced key is not in state.
    """
    resolved = copy.deepcopy(input_params)

    def resolve(value):
        if isinstance(value, str) and value.startswith("$$"):
            # Dotted path: $$STEP_1_OUTPUT$$.field_name
            if "$$." in value:
                ref_part, _, attr_path = value[2:].partition("$$.")
                if ref_part not in state:
                    raise ValueError(
                        f"Dependency Error: Reference {ref_part} not found in execution state."
                    )
                obj = state[ref_part]
                for attr in attr_path.split("."):
                    if isinstance(obj, dict):
                        obj = obj[attr]
                    else:
                        obj = getattr(obj, attr)
                return obj

            # Whole-string: $$STEP_1_OUTPUT$$
            if value.endswith("$$"):
                ref_key = value[2:-2]
                if ref_key not in state:
                    raise ValueError(
                        f"Dependency Error: Reference {ref_key} not found in execution state."
                    )
                return state[ref_key]

        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(resolved)
