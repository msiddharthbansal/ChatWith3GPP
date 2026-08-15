"""
Grounded generation: takes retrieval results from app.retrieval and produces
an answer that cites clauses, refuses when context is insufficient, and
compares (never blends) paired Rel-18/Rel-19 context for comparison queries.

Generator model: Llama-3-8B-Instruct, per the Chat3GPP paper. Groq has
deprecated the original `llama3-8b-8192` model string; `llama-3.1-8b-instant`
is its direct successor and the closest currently-available match.
"""
import os

from dotenv import load_dotenv
from groq import Groq

from app.retrieval import hybrid_search, compare_search, detect_releases

load_dotenv()

MODEL = "llama-3.1-8b-instant"

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
    header = f"[{c['spec_id']} {c['release']} {c.get('clause_number') or ''} {c['title']}]".strip()
    return f"{header}\n{c['content']}"


# Groq free tier caps llama-3.1-8b-instant at 6,000 tokens/minute (input +
# output combined). Keep the context comfortably under that regardless of
# how many chunks retrieval returns, dropping the lowest-ranked chunks first.
MAX_CONTEXT_CHARS = 10000


def _trim_to_budget(chunks, max_chars):
    kept, total = [], 0
    for c in chunks:
        piece_len = len(_format_chunk(c)) + 2
        if total + piece_len > max_chars:
            break
        kept.append(c)
        total += piece_len
    return kept


def build_context(query: str, releases: list[str] | None = None):
    """Return (context_text, sources, is_comparison). Pass `releases` to
    target an exact release (e.g. for MCQ evaluation) and skip comparison
    auto-detection entirely."""
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
    else:
        results = _trim_to_budget(hybrid_search(query, releases=releases), MAX_CONTEXT_CHARS)
        context = "\n\n".join(_format_chunk(c) for c in results)
        sources = results

    return context, sources, is_comparison


def stream_answer(query: str, max_tokens: int = 1024):
    """Return (sources, text_stream_generator)."""
    context, sources, _ = build_context(query)
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
    context, sources, _ = build_context(question, releases=releases)

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
