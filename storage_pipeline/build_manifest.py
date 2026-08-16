import json
import re
from datetime import datetime, timezone
from pathlib import Path


def decode_release_char(ch: str) -> int:
    ch = ch.lower()
    if ch.isdigit():
        return int(ch)
    if "a" <= ch <= "z":
        return 10 + (ord(ch) - ord("a"))
    raise ValueError(f"Unrecognised version-code character: {ch!r}")


FILENAME_RE = re.compile(r"^(\d{5})-([0-9a-zA-Z]{3})\.zip$")


def parse_zip_filename(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    digits, code = m.groups()
    spec_id = f"{digits[:2]}.{digits[2:]}"
    series = digits[:2]
    major = decode_release_char(code[0])
    minor = decode_release_char(code[1])
    patch = decode_release_char(code[2])
    return {
        "spec_id": spec_id,
        "series": series,
        "release_num": major,
        "release_label": f"Rel-{major}",
        "version": f"{major}.{minor}.{patch}",
    }


def build_manifest(raw_dir: Path) -> list[dict]:
    raw_dir = Path(raw_dir)
    entries = []
    for zip_path in sorted(raw_dir.rglob("*.zip")):
        parsed = parse_zip_filename(zip_path.name)
        if not parsed:
            print(f"  [skip] couldn't parse filename: {zip_path.name}")
            continue
        entries.append(
            {
                **parsed,
                "filename": zip_path.name,
                "relative_path": str(zip_path.relative_to(raw_dir.parent)),
                "size_bytes": zip_path.stat().st_size,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return entries


def save_manifest(entries: list[dict], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} entries to {out_path}")
    return out_path


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Build a manifest of downloaded 3GPP spec zips.")
    ap.add_argument("--raw-dir", type=Path, required=True, help="Folder to scan recursively for *.zip")
    ap.add_argument("--out", type=Path, required=True, help="Output manifest.json path")
    args = ap.parse_args()

    entries = build_manifest(args.raw_dir)
    save_manifest(entries, args.out)
    for e in entries:
        print(f"  {e['spec_id']:>8}  {e['release_label']:<7}  v{e['version']:<10}  {e['filename']}")


if __name__ == "__main__":
    _cli()
