"""
Query rewriting: generates alternate phrasings of the user's question before
retrieval, to close the vocabulary gap between conversational questions and
formal 3GPP clause text. Confirmed load-bearing by direct measurement: the
same answer chunk for a real test question ranked 17th by cross-encoder
score under a colloquial phrasing ("does not retain the same EPS bearer
context") and 1st under spec-style phrasing ("EPS bearer context status IE
... indicates ... inactive") — see .claude/memory.md section 9. A single
rewrite only gets one shot at guessing the right vocabulary; generating
several diverse reformulations (multi-query retrieval, a standard IR
technique) and merging their results raises the odds one of them lands on
wording that matches the spec text.

Toggle via QUERY_REWRITE_ENABLED (default "true") and tune fan-out via
QUERY_REWRITE_COUNT (default 1) so both can be adjusted — or this whole
technique swapped for something else (e.g. contextual retrieval / chunk-
context prepending at index time) — without touching call sites in
app/retrieval.py.

QUERY_REWRITE_COUNT tested at 2 (2026-08-16) and reverted to 1: it did not
reliably reproduce the "use the exact spec vocabulary" phrasing that made
the difference in testing (the model's reformulations stayed close to the
original's framing rather than pivoting to literal IE/message names), and
the extra candidate diversity measurably diluted two previously-clean
answers (see .claude/memory.md section 9) rather than fixing the case it
targeted. Left tunable in case a better-tuned prompt is worth retrying
later, but 1 is the empirically-validated default.
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "true").lower() == "true"
QUERY_REWRITE_COUNT = int(os.environ.get("QUERY_REWRITE_COUNT", "1"))
REWRITE_MODEL = "llama-3.1-8b-instant"

REWRITE_SYSTEM_PROMPT = """You generate alternate phrasings of a user's \
question about 3GPP telecom standards, to improve retrieval against formal \
spec text.

Generate exactly {n} reformulations, one per line, no numbering or bullets.
Each must stay a QUESTION (not a statement) about exactly what the original
question asks — do not introduce new concepts, causes, or mechanisms the
original question didn't mention or imply.

Rules for each reformulation:
- Expand acronyms actually present in the question to their full form (e.g.
"EBI" -> "EPS Bearer Identity"). Only expand acronyms that are there —
never introduce an unrelated acronym or concept that isn't implied by the
original question.
- Rephrase into the formal, technical phrasing 3GPP spec clauses actually
use — definitions, value ranges, procedures, information element (IE) and
message names (e.g. SERVICE REQUEST, SERVICE ACCEPT, ATTACH ACCEPT,
TRACKING AREA UPDATE ACCEPT — use whichever real 3GPP NAS message name is
relevant to what the question is about).
- Make each of the {n} reformulations genuinely different from the others in
wording and angle: at least one should be framed around the literal
message/IE names involved (spec text often states a fact via a specific
field or message name rather than the general concept a user would use to
ask about it); another can use the general procedural framing.
- Never invent or guess a spec number, clause number, or section number you
are not certain is real — an invented one will pollute keyword search. Refer
to concepts by name only unless the original question already named a
spec/clause.
- Each reformulation: one sentence, no preamble, no explanation.
- Output ONLY the {n} reformulations, one per line. Do not answer the
question.
"""

_client = None


def client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def rewrite_queries(query: str, n: int = QUERY_REWRITE_COUNT) -> list[str]:
    """Return up to `n` retrieval-oriented reformulations of `query` (may be
    fewer if the model produces duplicates/blank lines), or an empty list if
    rewriting is disabled or fails for any reason — callers must fall back
    to the original query rather than block retrieval on this."""
    if not QUERY_REWRITE_ENABLED or n <= 0:
        return []
    try:
        resp = client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=80 * n,
            temperature=0.3,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT.format(n=n)},
                {"role": "user", "content": query},
            ],
        )
        lines = resp.choices[0].message.content.strip().splitlines()
        seen = {query.strip().lower()}
        rewrites = []
        for line in lines:
            cleaned = line.strip().lstrip("-*0123456789.) ").strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                rewrites.append(cleaned)
        return rewrites[:n]
    except Exception:
        return []
