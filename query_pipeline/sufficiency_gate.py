import os

SUFFICIENCY_THRESHOLD = 0.3

SUFFICIENCY_GATE_ENABLED = os.environ.get("SUFFICIENCY_GATE_ENABLED", "true").lower() == "true"

REFUSAL_TEXT = "Insufficient context in the provided specifications to answer this."


def is_sufficient(results):
    return bool(results) and max(r["score"] for r in results) >= SUFFICIENCY_THRESHOLD
