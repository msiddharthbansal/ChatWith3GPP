MAX_CONTEXT_CHARS = 11000


def format_chunk(c):
    if c.get("chunk_type") in ("change_request", "change_request_rationale"):
        cr_number = c.get("cr_number") or "?"
        header = (
            f"[PROPOSED CR {cr_number} — NOT ratified spec text — "
            f"{c['spec_id']} {c['release']} {c.get('clause_number') or ''} {c['title']}]"
        ).strip()
    else:
        header = f"[{c['spec_id']} {c['release']} {c.get('clause_number') or ''} {c['title']}]".strip()
    return f"{header}\n{c['content']}"


def dedupe_by_clause(ranked_results, max_per_clause):
    clause_counts = {}
    deduped = []
    for c in ranked_results:
        key = (c["spec_id"], c.get("clause_number"))
        if key[1] is not None:
            count = clause_counts.get(key, 0)
            if count >= max_per_clause:
                continue
            clause_counts[key] = count + 1
        deduped.append(c)
    return deduped


def trim_to_budget(chunks, max_chars=MAX_CONTEXT_CHARS):
    kept, total = [], 0
    for c in chunks:
        piece_len = len(format_chunk(c)) + 2
        if total + piece_len > max_chars:
            continue
        kept.append(c)
        total += piece_len
    return kept
