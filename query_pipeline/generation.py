import os

from dotenv import load_dotenv
from groq import Groq

from query_pipeline.retrieval import hybrid_search, compare_search, detect_releases
from query_pipeline.context_packing import format_chunk, trim_to_budget, MAX_CONTEXT_CHARS
from query_pipeline.sufficiency_gate import SUFFICIENCY_GATE_ENABLED, REFUSAL_TEXT, is_sufficient
from query_pipeline.prompts import SYSTEM_PROMPT, MCQ_SYSTEM_PROMPT

load_dotenv()

MODEL = "llama-3.1-8b-instant"

_client = None


def client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def build_context(query: str, releases: list[str] | None = None):
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
                    parts.append(format_chunk(c))
                    sources.append(c)
        for c in result["unpaired"]:
            parts.append(format_chunk(c))
            sources.append(c)
        context = "\n\n".join(parts)
        sufficient = bool(sources)
    else:
        results = hybrid_search(query, releases=releases)
        sufficient = is_sufficient(results)
        results = trim_to_budget(results, MAX_CONTEXT_CHARS)
        context = "\n\n".join(format_chunk(c) for c in results)
        sources = results

    return context, sources, is_comparison, sufficient


def stream_answer(query: str, max_tokens: int = 1024):
    context, sources, _, sufficient = build_context(query)

    if SUFFICIENCY_GATE_ENABLED and not sufficient:
        def _refuse():
            yield REFUSAL_TEXT
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
