"""Conservative full-trace counterfactual guard, not a native bias-curve license."""
import math
from .identity import ContractError


def inspect_trace(logit_steps, bias, token_id):
    """Float64 toy/reference arithmetic. A changed token invalidates every later prefix.

    Native precision, processors, stopping, vocabulary and ties require independent
    qualification. This function deliberately never infers correctness from step zero.
    """
    if not logit_steps or type(bias) not in (float, int) or not math.isfinite(bias):
        raise ContractError("Finite bias and nonempty complete trace required")
    width = len(logit_steps[0])
    if type(token_id) is not int or not 0 <= token_id < width:
        raise ContractError("Token outside vocabulary")
    for step, logits in enumerate(logit_steps):
        if len(logits) != width or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits):
            raise ContractError("Malformed logit trace")
        changed = [float(x) + (bias if i == token_id else 0.) for i, x in enumerate(logits)]
        if not all(math.isfinite(x) for x in changed):
            raise ContractError("Counterfactual overflow")
        # Python max over ordered indices matches the declared first-index tie rule.
        old = max(range(width), key=lambda i: logits[i])
        new = max(range(width), key=lambda i: changed[i])
        if old != new:
            return {"status": "REGENERATE_FROM_DIVERGENCE", "step": step,
                    "native_token": old, "counterfactual_token": new,
                    "later_saved_prefixes_valid": False}
    return {"status": "UNCHANGED_ON_SUPPLIED_COMPLETE_TRACE", "steps": len(logit_steps),
            "native_applicability": "UNQUALIFIED_FLOAT64_REFERENCE_ONLY"}
