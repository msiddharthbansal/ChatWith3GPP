import re
import subprocess
from pathlib import Path

DEFINITIONS_START_RE = re.compile(r"^3\s+Terms and definitions\s*$")
DEFINITIONS_END_RE = re.compile(r"^4\s+Abbreviations\s*$")
ABBREV_END_RE = re.compile(r"^5\s+Equations\s*$")

FOOTER_RE = re.compile(r"^\s*ETSI\s*$|^\x0c?3GPP TR 21\.905 version")
NAV_HEADER_RE = re.compile(r"^[A-Z0-9](-[0-9])?$")


def pdf_to_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _extract_section(lines, start_re, end_re):
    start = next((i for i, l in enumerate(lines) if start_re.match(l.strip())), None)
    end = next((i for i, l in enumerate(lines) if end_re.match(l.strip())), len(lines))
    if start is None:
        return []
    return lines[start + 1 : end]


def parse_definitions(lines):
    clean = [l.rstrip("\n") for l in lines if not FOOTER_RE.match(l.strip())]

    paragraphs = []
    current = []
    for l in clean:
        if l.strip() == "":
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(l)
    if current:
        paragraphs.append(current)

    entries = []
    for para in paragraphs:
        text = " ".join(p.strip() for p in para).strip()
        if not text or NAV_HEADER_RE.match(text):
            continue
        if text.upper().startswith("NOTE"):
            if entries:
                entries[-1]["content"] += " " + text
            continue
        if ":" not in text:
            continue
        term, definition = text.split(":", 1)
        term = term.strip()
        definition = definition.strip()
        if not term or not definition:
            continue
        entries.append({"term": term, "content": f"{term}: {definition}"})
    return entries


def parse_abbreviations(lines):
    entries = []
    last_abbrev = None
    for raw in lines:
        l = raw.rstrip("\n")
        stripped = l.strip()
        if not stripped or FOOTER_RE.match(stripped) or NAV_HEADER_RE.match(stripped):
            continue
        lead = len(l) - len(l.lstrip(" "))
        if lead >= 10:
            if last_abbrev:
                entries.append({"term": last_abbrev, "content": f"{last_abbrev}: {stripped}"})
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        abbrev, expansion = parts
        last_abbrev = abbrev
        entries.append({"term": abbrev, "content": f"{abbrev}: {expansion}"})
    return entries


def chunk_glossary_pdf(pdf_path: Path, release: str, version: str, source_file: str):
    text = pdf_to_text(pdf_path)
    lines = text.splitlines()

    def_lines = _extract_section(lines, DEFINITIONS_START_RE, DEFINITIONS_END_RE)
    abbr_lines = _extract_section(lines, DEFINITIONS_END_RE, ABBREV_END_RE)

    definitions = parse_definitions(def_lines)
    abbreviations = parse_abbreviations(abbr_lines)

    chunks = []
    for i, e in enumerate(definitions):
        chunks.append(
            {
                "spec_id": "21.905",
                "release": release,
                "version": version,
                "clause_number": "3",
                "title": e["term"],
                "title_path": f"Terms and definitions > {e['term']}",
                "chunk_index": i,
                "chunk_type": "glossary_term",
                "content": e["content"],
                "embedding_text": f"{e['term']}\n\n{e['content']}",
                "source_file": source_file,
            }
        )
    for i, e in enumerate(abbreviations):
        chunks.append(
            {
                "spec_id": "21.905",
                "release": release,
                "version": version,
                "clause_number": "4",
                "title": e["term"],
                "title_path": f"Abbreviations > {e['term']}",
                "chunk_index": i,
                "chunk_type": "glossary_abbreviation",
                "content": e["content"],
                "embedding_text": f"{e['term']}\n\n{e['content']}",
                "source_file": source_file,
            }
        )
    return chunks


def write_chunks_jsonl(chunks: list[dict], out_path: Path):
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out_path


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Chunk TR 21.905 (Vocabulary for 3GPP Specifications) PDF into term/abbreviation JSONL.")
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    chunks = chunk_glossary_pdf(args.pdf, release=args.release, version=args.version, source_file=args.pdf.name)
    write_chunks_jsonl(chunks, args.out)
    definitions = sum(1 for c in chunks if c["chunk_type"] == "glossary_term")
    abbreviations = sum(1 for c in chunks if c["chunk_type"] == "glossary_abbreviation")
    print(f"{args.pdf.name}: {definitions} terms, {abbreviations} abbreviations -> {args.out}")


if __name__ == "__main__":
    _cli()
