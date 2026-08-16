import re
from pathlib import Path

from chunk_docx import (
    docx_to_markdown,
    split_into_sections,
    recursive_split,
    write_chunks_jsonl,
    SKIP_HEADING_TITLES,
)

CHANGE_MARKER_RE = re.compile(
    r"^[\s=*]*(?:first|next|start|end)(?:\s+of)?\s+changes?[\s=*]*$",
    re.IGNORECASE,
)

COVER_SHEET_PATTERNS = {
    "spec_id": re.compile(r"\*\*(\d{2}\.\d{3})\*\*\s*\|?\s*\*\*CR\*\*"),
    "cr_number": re.compile(r"\*\*CR\*\*\s*\|?\s*\*\*(\S+)\*\*"),
    "current_version": re.compile(r"Current version:\*\*\s*\|?\s*\*\*([\d.]+)\*\*"),
    "category": re.compile(r"\*\*\*Category:\*\*\*\s*\|?\s*>?\s*\*\*([A-F])\*\*"),
    "release": re.compile(r"\*\*\*Release:\*\*\*(?:(?!\n\+)[\s\S])*?(Rel-\d+)"),
    "title": re.compile(r"\*\*\*Title:\*\*\*\s*\|?\s*>?\s*(.+)"),
    "clauses_affected": re.compile(r"\*\*\*Clauses affected:\*\*\*\s*\|?\s*>?\s*(.+)"),
}

FREE_TEXT_FIELDS = {
    "reason_for_change": "Reason for change:",
    "summary_of_change": "Summary of change:",
    "consequences_if_not_approved": "Consequences if not approved:",
}


def split_change_sections(markdown: str):
    lines = markdown.splitlines()
    marker_idx = [i for i, l in enumerate(lines) if CHANGE_MARKER_RE.match(l.strip())]
    if not marker_idx:
        return markdown, ""
    cover_text = "\n".join(lines[: marker_idx[0]])
    skip = set(marker_idx)
    body_text = "\n".join(l for i, l in enumerate(lines) if i > marker_idx[0] and i not in skip)
    return cover_text, body_text


def _clean_cell_text(text: str) -> str:
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"^[>\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*\*?", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def parse_cover_sheet(cover_text: str) -> dict:
    meta = {}
    for field, pattern in COVER_SHEET_PATTERNS.items():
        m = pattern.search(cover_text)
        if m:
            meta[field] = _clean_cell_text(m.group(1))

    lines = cover_text.splitlines()
    for field, label in FREE_TEXT_FIELDS.items():
        start = next((i for i, l in enumerate(lines) if label in l), None)
        if start is None:
            continue
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if re.match(r"^\+[-=]+\+", lines[i]) and i > start + 1
            ),
            len(lines),
        )
        block = "\n".join(lines[start:end])
        block = block.split(label, 1)[1]
        meta[field] = _clean_cell_text(block)
    return meta


def chunk_cr_document(
    markdown: str,
    source_file: str,
    max_chars: int,
    overlap: int = 0,
    min_chars: int = 40,
    spec_id_override: str | None = None,
    release_override: str | None = None,
):
    cover_text, body_text = split_change_sections(markdown)
    meta = parse_cover_sheet(cover_text)
    spec_id_source = "cover_sheet"

    if not meta.get("spec_id") and spec_id_override:
        meta["spec_id"] = spec_id_override
        spec_id_source = "manual_override"
    if not meta.get("release") and release_override:
        meta["release"] = release_override

    if not meta.get("spec_id") or not meta.get("release"):
        return None, meta

    chunks = []

    rationale_parts = [
        f"{k.replace('_', ' ').title()}: {v}"
        for k, v in meta.items()
        if k in FREE_TEXT_FIELDS and v
    ]
    if rationale_parts:
        chunks.append(
            {
                "spec_id": meta["spec_id"],
                "release": meta["release"],
                "current_version": meta.get("current_version"),
                "cr_number": meta.get("cr_number"),
                "category": meta.get("category"),
                "title": meta.get("title") or f"CR {meta.get('cr_number', '')} rationale",
                "clause_number": None,
                "clauses_affected": meta.get("clauses_affected"),
                "chunk_index": 0,
                "chunk_type": "change_request_rationale",
                "content": "\n".join(rationale_parts),
                "source_file": source_file,
                "spec_id_source": spec_id_source,
            }
        )

    sections = split_into_sections(body_text)
    for clause_num, title, title_path, body in sections:
        if title and title.lower().strip() in SKIP_HEADING_TITLES:
            continue
        if len(body) < min_chars:
            continue
        pieces = recursive_split(body, max_chars=max_chars, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "spec_id": meta["spec_id"],
                    "release": meta["release"],
                    "current_version": meta.get("current_version"),
                    "cr_number": meta.get("cr_number"),
                    "category": meta.get("category"),
                    "clause_number": clause_num,
                    "title": title,
                    "title_path": title_path,
                    "clauses_affected": meta.get("clauses_affected"),
                    "chunk_index": i,
                    "chunk_type": "change_request",
                    "content": piece,
                    "embedding_text": f"[PROPOSED CHANGE, CR {meta.get('cr_number', '')}] {title_path}\n\n{piece}",
                    "source_file": source_file,
                    "spec_id_source": spec_id_source,
                }
            )

    return chunks, meta


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Chunk a 3GPP Change Request .docx into tagged JSONL.")
    ap.add_argument("--docx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, required=True)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument(
        "--spec-id-override",
        help="Use when the cover sheet's spec field is blank but the target "
        "spec is identifiable from the body content (clause numbering, "
        "IE/ASN.1 naming, etc.) — e.g. a pre-submission draft.",
    )
    ap.add_argument("--release-override")
    args = ap.parse_args()

    markdown = docx_to_markdown(args.docx)
    chunks, meta = chunk_cr_document(
        markdown,
        source_file=args.docx.name,
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chars=args.min_chars,
        spec_id_override=args.spec_id_override,
        release_override=args.release_override,
    )
    if chunks is None:
        print(f"SKIPPED {args.docx.name}: could not parse spec_id/release from cover sheet. Extracted meta: {meta}")
        return
    write_chunks_jsonl(chunks, args.out)
    print(f"{args.docx.name} -> {meta.get('spec_id')} CR {meta.get('cr_number')} ({meta.get('release')}): {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    _cli()
