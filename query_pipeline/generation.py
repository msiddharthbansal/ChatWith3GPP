"""
Grounded generation: takes retrieval results from app.retrieval and produces
an answer that cites clauses, refuses when context is insufficient, and
compares (never blends) paired Rel-18/Rel-19 context for comparison queries.

Generator model: Llama-3-8B-Instruct, per the Chat3GPP paper. Groq has
deprecated the original `llama3-8b-8192` model string; `llama-3.1-8b-instant`
is its direct successor and the closest currently-available match.
"""
import os
import re

from dotenv import load_dotenv
from groq import Groq

from app.retrieval import hybrid_search, compare_search, detect_releases

load_dotenv()

MODEL = "llama-3.1-8b-instant"

# Cross-encoder rerank score (roughly 0-1) below which the top retrieved
# candidate is treated as a clear no-match — retrieval found nothing
# genuinely relevant, so skip the LLM call entirely rather than let it
# reason past thin evidence. Deliberately conservative: real questions we've
# tested score >=0.70 even when the *right* chunk doesn't make top_k, so
# this only fires on queries retrieval has nothing plausible for (e.g.
# out-of-domain questions), not on the borderline cases documented in
# .claude/memory.md (section 9).
SUFFICIENCY_THRESHOLD = 0.3

# Both toggles default on but can be flipped off independently via env vars
# (no code change, no redeploy of call sites) — same pattern as
# QUERY_REWRITE_ENABLED in app/query_rewrite.py, so any of the three
# retrieval/generation-quality additions can be pulled out independently if
# it turns out not to help, without touching the other two.
SUFFICIENCY_GATE_ENABLED = os.environ.get("SUFFICIENCY_GATE_ENABLED", "true").lower() == "true"
CITATION_CHECK_ENABLED = os.environ.get("CITATION_CHECK_ENABLED", "true").lower() == "true"

SYSTEM_PROMPT = """You are a technical assistant answering questions about 3GPP \
telecom standards (Rel-18 and Rel-19), grounded strictly in the context provided \
with each question.

Rules:
1. Answer using ONLY the provided context. Do not use outside knowledge of 3GPP \
specifications, even if you recall it, since it may not match the exact version cited.
2. Every factual claim must cite its source inline as [spec_id clause_number], e.g. \
[24.501 5.4.1.2]. If a clause number is not available, cite [spec_id title] instead.
3. If the provided context does not contain enough information to answer, say so \
explicitly: "Insufficient context in the provided specifications to answer this." \
Do not guess or fill gaps from general knowledge.
4. When the context contains paired Rel-18/Rel-19 excerpts for the same clause, \
compare them explicitly and state what changed (or that nothing changed) — never \
blend the two releases into a single unattributed statement.
5. Some context is tagged "[PROPOSED CR ... — NOT ratified spec text — ...]". This \
is a Change Request: a proposal to amend a spec, not confirmed or ratified spec \
text. When citing it, say explicitly that it is a proposed change (e.g. "a \
proposed CR would add...") — never present it as if it were already part of the \
published specification.
6. Never invent a citation, spec, clause, or "proposed CR" that does not literally \
appear in the provided context — every [bracketed citation] you write must match a \
citation header that exists verbatim above. If the context only partially answers \
the question, answer the part it supports and explicitly say the rest is not \
covered by the provided context, rather than filling the gap with a fabricated \
source.
7. Context often contains ASN.1 IE definitions (common in 37.355, 38.331, and \
OpenAPI-derived content). Read ASN.1 range notation directly: "FieldName ::= \
INTEGER (X..Y)" means the field's valid range is X to Y inclusive. If a question \
asks for a range, size, or bound and the context contains this notation for the \
relevant field, that IS the answer — extract X and Y rather than treating the \
notation as unrelated boilerplate.
"""

MCQ_SYSTEM_PROMPT = """You are answering multiple-choice questions about 3GPP \
telecom standards, grounded strictly in the context provided with each question.

Respond with ONLY the single letter of the correct option (A, B, C, or D). \
No explanation, no punctuation, nothing else — just the letter.
"""

_client = None


def client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _format_chunk(c):
    if c.get("chunk_type") in ("change_request", "change_request_rationale"):
        cr_number = c.get("cr_number") or "?"
        header = (
            f"[PROPOSED CR {cr_number} — NOT ratified spec text — "
            f"{c['spec_id']} {c['release']} {c.get('clause_number') or ''} {c['title']}]"
        ).strip()
    else:
        header = f"[{c['spec_id']} {c['release']} {c.get('clause_number') or ''} {c['title']}]".strip()
    return f"{header}\n{c['content']}"


# Groq free tier caps llama-3.1-8b-instant at 6,000 tokens/minute (input +
# output combined). Keep the context comfortably under that regardless of
# how many chunks retrieval returns, dropping the lowest-ranked chunks first.
# Bumped 10000->11000 alongside the skip-continue packing change below —
# still comfortable headroom under the ~24000-char (6000 token) TPM cap.
MAX_CONTEXT_CHARS = 11000


def _trim_to_budget(chunks, max_chars):
    """Greedy-fill in score order, but skip (not break on) a chunk that
    doesn't fit and keep trying smaller/later ones — a hard break on the
    first overflow was silently discarding every lower-ranked chunk after
    it, including smaller ones that would've fit in the leftover budget
    (see .claude/memory.md section 9's Q3 diagnosis). Order of `kept` is
    still the input's score order, just with non-fitting chunks removed
    rather than truncating the whole list at the first miss."""
    kept, total = [], 0
    for c in chunks:
        piece_len = len(_format_chunk(c)) + 2
        if total + piece_len > max_chars:
            continue
        kept.append(c)
        total += piece_len
    return kept


def build_context(query: str, releases: list[str] | None = None):
    """Return (context_text, sources, is_comparison, sufficient). Pass
    `releases` to target an exact release (e.g. for MCQ evaluation) and skip
    comparison auto-detection entirely. `sufficient` is a conservative
    pre-generation confidence gate (see SUFFICIENCY_THRESHOLD) — comparison
    queries don't carry a comparable rerank score post-pairing, so they're
    always treated as sufficient when any context was found."""
    is_comparison = False
    if releases is None:
        releases, is_comparison = detect_releases(query)

    if is_comparison:
        result = compare_search(query)
        parts = []
        sources = []
        for key, by_release in result["paired"].items():
            parts.append(f"## {key}")
            for release in ("Rel-18", "Rel-19"):
                chunks = by_release.get(release, [])
                if not chunks:
                    parts.append(f"### {release}: (not found)")
                    continue
                parts.append(f"### {release}")
                for c in chunks:
                    parts.append(_format_chunk(c))
                    sources.append(c)
        for c in result["unpaired"]:
            parts.append(_format_chunk(c))
            sources.append(c)
        context = "\n\n".join(parts)
        sufficient = bool(sources)
    else:
        results = hybrid_search(query, releases=releases)
        sufficient = bool(results) and max(r["score"] for r in results) >= SUFFICIENCY_THRESHOLD
        results = _trim_to_budget(results, MAX_CONTEXT_CHARS)
        context = "\n\n".join(_format_chunk(c) for c in results)
        sources = results

    return context, sources, is_comparison, sufficient


def stream_answer(query: str, max_tokens: int = 1024):
    """Return (sources, text_stream_generator). If retrieval's top score is
    below SUFFICIENCY_THRESHOLD, skips the LLM call entirely and returns a
    fixed refusal — a programmatic backstop, not a replacement, for the
    "insufficient context" instruction in SYSTEM_PROMPT."""
    context, sources, _, sufficient = build_context(query)

    if SUFFICIENCY_GATE_ENABLED and not sufficient:
        def _refuse():
            yield "Insufficient context in the provided specifications to answer this."
        return sources, _refuse()

    user_content = f"Context:\n\n{context}\n\nQuestion: {query}"

    def _stream():
        stream = client().chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return sources, _stream()


def answer_mcq(question: str, options: dict, release: str | None = None):
    """Answer a multiple-choice question against retrieved context.
    Returns (answer_letter, context_text, sources). `release` should be a
    single release ('Rel-18' or 'Rel-19'); pass None to search both."""
    releases = [release] if release else ["Rel-18", "Rel-19"]
    context, sources, _, _ = build_context(question, releases=releases)

    options_text = "\n".join(f"{letter}) {text}" for letter, text in options.items())
    user_content = f"Context:\n\n{context}\n\nQuestion: {question}\n{options_text}"

    resp = client().chat.completions.create(
        model=MODEL,
        max_tokens=5,
        messages=[
            {"role": "system", "content": MCQ_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    return answer, context, sources


CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def verify_citations(answer_text: str, sources: list[dict]) -> list[str]:
    """Post-generation backstop for SYSTEM_PROMPT rule 6 (never fabricate a
    citation): extract every bracketed citation the model wrote and check it
    references a spec_id (or, for CRs, a cr_number) that's actually present
    in `sources`. Returns the list of citation strings that don't match
    anything retrieved — flagged, not stripped, since the underlying claim
    may still be correct even if the citation format drifted. Toggle via
    CITATION_CHECK_ENABLED; disabled returns no flags, i.e. a no-op."""
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
