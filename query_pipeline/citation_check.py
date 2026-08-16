import os
import re

CITATION_CHECK_ENABLED = os.environ.get("CITATION_CHECK_ENABLED", "true").lower() == "true"

CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def verify_citations(answer_text: str, sources: list[dict]) -> list[str]:
    if not CITATION_CHECK_ENABLED:
        return []

    known_spec_ids = {s["spec_id"] for s in sources}
    known_cr_numbers = {s.get("cr_number") for s in sources if s.get("cr_number")}

    flagged = []
    for match in CITATION_RE.finditer(answer_text):
        text = match.group(1)
        if text.startswith("PROPOSED CR"):
            if not any(cr and cr in text for cr in known_cr_numbers):
                flagged.append(text)
            continue
        if not any(text.startswith(spec_id) for spec_id in known_spec_ids):
            flagged.append(text)
    return flagged
