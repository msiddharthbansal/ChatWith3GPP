"""
chunk_docx.py

Splits a 3GPP .docx spec into clause-tagged chunks. Usable two ways:

CLI:
    python chunk_docx.py \\
        --docx data/extracted/rel18/23501-ic0.docx \\
        --spec-id 23.501 --release "Rel-18" --version 18.12.0 \\
        --out data/processed/rel18/23.501.jsonl \\
        --max-chars 1250 --overlap 100

Import:
    from pathlib import Path
    from chunk_docx import docx_to_markdown, chunk_document, write_chunks_jsonl

    markdown = docx_to_markdown(Path("data/extracted/rel18/23501-ic0.docx"))
    chunks = chunk_document(
        markdown,
        spec_id="23.501", release="Rel-18", version="18.12.0",
        source_file="23501-ic0.docx",
        max_chars=1250, overlap=100,
    )
    write_chunks_jsonl(chunks, Path("data/processed/rel18/23.501.jsonl"))
"""
import json
import re
import subprocess
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Headings that are pure boilerplate / navigation noise in every 3GPP spec.
# Matched case-insensitively against the heading TEXT (after any clause number).
SKIP_HEADING_TITLES = {
    "foreword",
    "contents",
    "table of contents",
    "change history",
}

# 3GPP clause numbering pattern, e.g. "4.2.3" or "5.15.3.2" or "A.1" (annexes)
CLAUSE_NUM_RE = re.compile(r"^(?P<num>[A-Z]?\d+(?:\.\d+)*)\s+(?P<title>.+)$")

# Annex title headings, e.g. "Annex F (informative): Change history"
ANNEX_TITLE_RE = re.compile(r"^Annex\s+\S+\s*\([^)]*\)\s*:\s*(?P<title>.+)$", re.IGNORECASE)


def _skip_title(title: str) -> bool:
    """True if this heading is boilerplate to drop, matching either the
    bare title (e.g. "Foreword") or an Annex-wrapped one (e.g. "Annex F
    (informative): Change history")."""
    t = title.lower().strip()
    if t in SKIP_HEADING_TITLES:
        return True
    m = ANNEX_TITLE_RE.match(title.strip())
    return bool(m and m.group("title").lower().strip() in SKIP_HEADING_TITLES)


def docx_to_markdown(docx_path: Path) -> str:
    """Convert a .docx to markdown via pandoc, preserving heading levels
    and rendering tables as markdown/grid tables."""
    result = subprocess.run(
        ["pandoc", "-t", "markdown", "--wrap=none", str(docx_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_heading(line: str):
    """Return (level, clause_num, title) for a markdown heading line, or
    None if the line isn't a heading."""
    m = re.match(r"^(#{1,})\s+(.*)$", line.strip())
    if not m:
        return None
    level = len(m.group(1))
    text = m.group(2).strip()
    # pandoc 3.x's markdown writer auto-appends {#id .class} attribute
    # blocks to headings that map to non-numbered/custom Word styles
    # (e.g. "Contents {#contents .TT}") — strip before clause/title parsing.
    text = re.sub(r"\s*\{#[^}]*\}\s*$", "", text).strip()
    cm = CLAUSE_NUM_RE.match(text)
    if cm:
        return level, cm.group("num"), cm.group("title").strip()
    return level, None, text


def split_into_sections(markdown: str):
    """Walk the markdown line by line, tracking a heading stack, and return
    one section per heading: (clause_num, title, title_path, body_text)."""
    lines = markdown.splitlines()
    stack = []  # list of (level, clause_num, title)
    sections = []
    current_body = []

    def flush():
        if not stack:
            return
        level, clause_num, title = stack[-1]
        title_path = " > ".join(t for _, _, t in stack)
        body = "\n".join(current_body).strip()
        sections.append((clause_num, title, title_path, body))

    for line in lines:
        parsed = parse_heading(line)
        if parsed:
            flush()
            current_body = []
            level, clause_num, title = parsed
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, clause_num, title))
        else:
            current_body.append(line)
    flush()
    return sections


def recursive_split(text: str, max_chars: int, overlap: int = 0):
    """
    Bound an oversized clause section to max_chars using LangChain's
    RecursiveCharacterTextSplitter (applied as a second pass *within* each
    heading-based section, so clause/title metadata is preserved on every
    resulting sub-chunk). Tries paragraph breaks, then line breaks, then
    sentences, then words, only hard-cutting as a last resort.

    max_chars: hard ceiling per chunk (Chat3GPP paper uses ~1250).
    overlap:   characters shared between adjacent sub-chunks (paper uses 0;
               ~80-150 is a reasonable middle ground to avoid boundary
               truncation without much index bloat).
    """
    if len(text) <= max_chars:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_document(
    markdown: str,
    spec_id: str,
    release: str,
    version: str,
    source_file: str,
    max_chars: int,
    overlap: int = 0,
    min_chars: int = 40,
):
    """
    Split a markdown document into clause-tagged chunks.

    max_chars: passed straight to recursive_split for any oversized section.
    overlap:   passed straight to recursive_split.
    min_chars: sections shorter than this are dropped (stray/empty headings).
    """
    sections = split_into_sections(markdown)
    chunks = []
    for clause_num, title, title_path, body in sections:
        if title and _skip_title(title):
            continue
        if len(body) < min_chars:
            continue
        # pandoc renders tables either as pipe-tables or as ASCII grid/simple
        # tables (dash-separator rows) depending on column-width heuristics —
        # detect both forms.
        is_table = bool(
            re.search(r"^\|.*\|$", body, re.MULTILINE)
            or re.search(r"^\s*-{5,}\s*$", body, re.MULTILINE)
        )
        pieces = recursive_split(body, max_chars=max_chars, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "spec_id": spec_id,
                    "release": release,
                    "version": version,
                    "clause_number": clause_num,
                    "title": title,
                    "title_path": title_path,
                    "chunk_index": i,
                    "chunk_type": "table" if is_table else "text",
                    "content": piece,
                    "embedding_text": f"{title_path}\n\n{piece}",
                    "source_file": source_file,
                }
            )
    return chunks


def write_chunks_jsonl(chunks: list[dict], out_path: Path):
    """Write chunks to a .jsonl file, one JSON object per line. Creates
    parent directories if needed. Returns out_path for chaining."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Chunk a 3GPP .docx spec into clause-tagged JSONL.")
    ap.add_argument("--docx", type=Path, required=True, help="Path to the .docx file")
    ap.add_argument("--spec-id", required=True, help='e.g. "23.501"')
    ap.add_argument("--release", required=True, help='e.g. "Rel-18"')
    ap.add_argument("--version", required=True, help='e.g. "18.12.0"')
    ap.add_argument("--out", type=Path, required=True, help="Output .jsonl path")
    ap.add_argument("--max-chars", type=int, required=True,
                     help="Hard ceiling per chunk (Chat3GPP paper uses ~1250)")
    ap.add_argument("--overlap", type=int, default=0,
                     help="Character overlap between adjacent sub-chunks (default 0, paper's setting)")
    ap.add_argument("--min-chars", type=int, default=40,
                     help="Drop sections shorter than this (default 40)")
    args = ap.parse_args()

    markdown = docx_to_markdown(args.docx)
    chunks = chunk_document(
        markdown,
        spec_id=args.spec_id, release=args.release, version=args.version,
        source_file=args.docx.name,
        max_chars=args.max_chars, overlap=args.overlap, min_chars=args.min_chars,
    )
    write_chunks_jsonl(chunks, args.out)
    print(f"{args.spec_id} ({args.release}): {len(chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    _cli()
