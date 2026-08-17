import os

from dotenv import load_dotenv
from groq import Groq

from query_pipeline.prompts import REWRITE_SYSTEM_PROMPT
from query_pipeline.glossary_lookup import lookup_glossary

load_dotenv()

QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "true").lower() == "true"
QUERY_REWRITE_COUNT = int(os.environ.get("QUERY_REWRITE_COUNT", "1"))
REWRITE_MODEL = "llama-3.1-8b-instant"

_client = None


def client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def rewrite_queries(query: str, n: int = QUERY_REWRITE_COUNT) -> list[str]:
    if not QUERY_REWRITE_ENABLED or n <= 0:
        return []
    try:
        glossary_matches = lookup_glossary(query)
        user_content = query
        if glossary_matches:
            user_content = (
                "Official 3GPP terminology that may be relevant (a term can have "
                "more than one meaning across different specs — use judgment, "
                "don't assume the first one applies):\n"
                + "\n".join(f"- {m}" for m in glossary_matches)
                + f"\n\nQuestion: {query}"
            )
        resp = client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=80 * n,
            temperature=0.3,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT.format(n=n)},
                {"role": "user", "content": user_content},
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
