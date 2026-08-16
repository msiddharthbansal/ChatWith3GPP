import spaces

import gradio as gr

from query_pipeline.generation import stream_answer, answer_mcq
from query_pipeline.citation_check import verify_citations

DESCRIPTION = (
    "Ask questions about 3GPP TS 23.501, 23.502, 24.501, 33.501, and 29.518 "
    "(Rel-18 and Rel-19). Answers are grounded in retrieved spec text with "
    "inline clause citations — if the specs don't cover something, the "
    "assistant says so instead of guessing.\n\n"
    "**Limits:** this is a free-tier demo. Generation runs on Groq's free "
    "tier (rate-limited to a few thousand tokens/minute), so very long or "
    "rapid-fire queries may be slow or fail — retry if that happens. "
    "Retrieval runs on Hugging Face ZeroGPU, which can add a cold-start "
    "delay on the first query after idle periods and caps total usage per "
    "session. Answers are grounded in a fixed corpus snapshot (Rel-18 and "
    "Rel-19 only) — anything outside that scope, or requiring live/current "
    "network data, is out of scope by design."
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
    options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}
    answer, _, _ = answer_mcq(question, options, release=release or None)
    return answer


with gr.Blocks(title="Chat3GPP — 5G/3GPP Spec Assistant") as demo:
    gr.ChatInterface(
        fn=respond,
        title="Chat3GPP — 5G/3GPP Spec Assistant",
        description=DESCRIPTION,
        examples=EXAMPLES,
    )
    gr.api(mcq_endpoint, api_name="answer_mcq")

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
