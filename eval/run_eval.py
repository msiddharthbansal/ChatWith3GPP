"""
Evaluation harness: runs eval/questions.json (a TeleQnA-style MCQ set,
matching how the Chat3GPP paper reports accuracy) against the deployed
Space's /answer_mcq API endpoint and reports accuracy overall, per-spec,
and per-release.

Retrieval only runs inside the Space (ZeroGPU-only — see app/embeddings.py),
so this script calls the live Space rather than importing app.* locally.
"""
import json
from collections import defaultdict
from pathlib import Path

from gradio_client import Client

SPACE = "msiddharth/3gpp-chatbot"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def run_mcq_eval(client: Client):
    questions = json.loads(QUESTIONS_PATH.read_text())

    results = []
    for q in questions:
        release = q["release"] if q["release"] in ("Rel-18", "Rel-19") else ""
        raw = client.predict(
            question=q["question"],
            option_a=q["options"]["A"],
            option_b=q["options"]["B"],
            option_c=q["options"]["C"],
            option_d=q["options"]["D"],
            release=release,
            api_name="/answer_mcq",
        )
        predicted = raw.strip().upper()[:1]
        correct = predicted == q["answer"]
        results.append(
            {
                "id": q["id"],
                "spec_id": q["spec_id"],
                "release": q["release"],
                "question": q["question"],
                "expected": q["answer"],
                "predicted": predicted,
                "raw_response": raw,
                "correct": correct,
            }
        )
        status = "✓" if correct else "✗"
        print(f"[{status}] Q{q['id']:>2} ({q['spec_id']} {q['release']}): expected {q['answer']}, got {predicted}")

    return results


def summarize(results):
    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}")

    by_spec = defaultdict(lambda: [0, 0])
    by_release = defaultdict(lambda: [0, 0])
    for r in results:
        by_spec[r["spec_id"]][0] += r["correct"]
        by_spec[r["spec_id"]][1] += 1
        by_release[r["release"]][0] += r["correct"]
        by_release[r["release"]][1] += 1

    print("\nBy spec:")
    for spec, (c, t) in sorted(by_spec.items()):
        print(f"  {spec}: {c}/{t} = {c/t:.1%}")

    print("\nBy release:")
    for rel, (c, t) in sorted(by_release.items()):
        print(f"  {rel}: {c}/{t} = {c/t:.1%}")


if __name__ == "__main__":
    client = Client(SPACE)
    results = run_mcq_eval(client)
    summarize(results)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved detailed results to {RESULTS_PATH}")
