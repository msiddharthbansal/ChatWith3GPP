import re

import spaces

import gradio as gr

from query_pipeline.generation import stream_answer, answer_mcq
from query_pipeline.citation_check import verify_citations

QUOTA_WAIT_RE = re.compile(r"Try again in ([\d:]+)")

DESCRIPTION = (
    "**Scope — this demo has indexed content only from:** "
    "23.501, 23.502, 23.503, 24.011, 24.301, 24.501, 29.500, 29.502, 29.518, "
    "33.501, 37.355, 38.300, 38.305, 38.331, 38.413. "
    "Rel-19 for all of them; Rel-18 also available for 23.501, 23.502, "
    "24.501, 29.518, and 33.501. A small number of proposed (not yet "
    "ratified) Change Requests are included for 24.011, 38.305, and "
    "38.331 — the assistant flags these explicitly as proposed, not "
    "settled spec text. Anything outside this list is out of scope: the "
    "assistant says so instead of guessing. Sources are shown in a "
    "collapsible panel under each answer, not inline in the text.\n\n"
    "**Limits:** Groq free tier — 30 req/min, 6,000 tokens/min. "
    "HF ZeroGPU — 5 min/day quota, 60s/query cap."
)

EXAMPLES = [
    "What are QoS rules used for?",
    "What changed in QoS rules between Rel-18 and Rel-19?",
    "How does the UE perform initial registration?",
    "What is network slicing?",
]


def respond(message, history):
    try:
        sources, stream = stream_answer(message)
    except Exception as e:
        text = str(e)
        if "quota" in text.lower():
            m = QUOTA_WAIT_RE.search(text)
            wait = m.group(1) if m else "some time"
            yield (
                "This demo has hit its free daily HF ZeroGPU compute quota. "
                f"**Try again in {wait}.**"
            )
        else:
            yield f"Something went wrong processing this query: {text}"
        return

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
        partial += (
            f"\n\n<details>\n<summary>Sources ({len(lines)})</summary>\n\n"
            + "\n".join(lines)
            + "\n\n</details>"
        )
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


with gr.Blocks(title="3GPP RAG Assistant") as demo:
    gr.ChatInterface(
        fn=respond,
        title="3GPP RAG Assistant",
        description=DESCRIPTION,
        examples=EXAMPLES,
    )
    gr.api(mcq_endpoint, api_name="answer_mcq")

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
