"""
HF Spaces entry point. Named gradio_app.py (not app.py) to avoid colliding
with the app/ package this project already uses for retrieval/generation
code — README.md's app_file points HF Spaces at this file explicitly.
"""
import gradio as gr

from app.generation import stream_answer

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


demo = gr.ChatInterface(
    fn=respond,
    title="Chat3GPP — 5G/3GPP Spec Assistant",
    description=DESCRIPTION,
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
