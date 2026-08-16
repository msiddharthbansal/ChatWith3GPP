"""
HF Spaces entry point. Named gradio_app.py (not app.py) to avoid colliding
with the app/ package this project already uses for retrieval/generation
code — README.md's app_file points HF Spaces at this file explicitly.
"""
import spaces  # noqa: F401 — must be imported before torch anywhere in the process

import gradio as gr

from app.generation import stream_answer, answer_mcq, verify_citations
from app.retrieval import hybrid_search

DESCRIPTION = (
    "Ask questions about 3GPP TS 23.501, 23.502, 24.501, 33.501, and 29.518 "
    "(Rel-18 and Rel-19). Answers are grounded in retrieved spec text with "
    "inline clause citations — if the specs don't cover something, the "
    "assistant says so instead of guessing."
)

EXAMPLES = [
    "What are QoS rules used for?",
    "What changed in QoS rules between Rel-18 and Rel-19?",
    "How does the UE perform initial registration?",
    "What is network slicing?",
]


def respond(message, history):
    sources, stream = stream_answer(message)

    partial = ""
    for chunk in stream:
        partial += chunk
        yield partial

    flagged = verify_citations(partial, sources)
    if flagged:
        cites = ", ".join(f"`[{f}]`" for f in flagged)
        partial += f"\n\n---\n**Note:** could not verify citation(s) {cites} against retrieved sources — treat with caution."
        yield partial

    if sources:
        seen = set()
        lines = []
        for s in sources:
            key = (s["spec_id"], s["release"], s.get("clause_number"))
            if key in seen:
                continue
            seen.add(key)
            clause = s.get("clause_number") or ""
            lines.append(f"- [{s['spec_id']} {s['release']} {clause}] {s['title']}")
        partial += "\n\n---\n**Sources:**\n" + "\n".join(lines)
        yield partial


def mcq_endpoint(
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    release: str = "",
) -> str:
    """API-only endpoint for the eval harness (eval/run_eval.py). Retrieval
    runs here since it's ZeroGPU-only; the eval script calls this remotely
    rather than running retrieval locally."""
    options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}
    answer, _, _ = answer_mcq(question, options, release=release or None)
    return answer


def debug_search_endpoint(query: str, release: str = "", pool: int = 50, max_per_clause: int = 99) -> str:
    """Temporary diagnostic endpoint: returns the full reranked candidate
    pool with cross-encoder scores, one per line, for inspecting where a
    specific chunk landed. max_per_clause defaults to effectively unbounded
    (99) here so this shows the RAW rerank order, unlike production's
    max_per_clause=3 — pass max_per_clause=3 explicitly to see what
    production would actually select. Not part of the app's normal
    surface — remove before final submission."""
    releases = [release] if release else None
    results = hybrid_search(
        query, top_k=pool, releases=releases, rerank_pool=pool, max_per_clause=max_per_clause
    )
    lines = []
    for i, r in enumerate(results):
        lines.append(
            f"{i + 1}\t{r['score']:.4f}\t{len(r['content'])}chars\t{r['spec_id']} {r['release']} "
            f"{r.get('clause_number') or ''}\t{r['title']}\t{r['content'][:60]!r}"
        )
    return "\n".join(lines)


with gr.Blocks(title="Chat3GPP — 5G/3GPP Spec Assistant") as demo:
    gr.ChatInterface(
        fn=respond,
        title="Chat3GPP — 5G/3GPP Spec Assistant",
        description=DESCRIPTION,
        examples=EXAMPLES,
    )
    gr.api(mcq_endpoint, api_name="answer_mcq")
    gr.api(debug_search_endpoint, api_name="debug_search")

if __name__ == "__main__":
    # ssr_mode disabled: Gradio 6's SSR feature runs a Node.js proxy in front
    # of the Python server, and it was the last thing logged before this
    # Space's container died with no Python traceback (consistent with the
    # Node proxy itself crashing and taking the container down).
    demo.launch(ssr_mode=False)
