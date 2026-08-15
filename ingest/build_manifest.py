"""
build_manifest.py

Scans data/raw/<release>/ folders for 3GPP zip files (e.g. 23501-ic0.zip)
and records, for every file: spec_id, series, release, version, source
filename, and a recorded timestamp. Usable two ways:

CLI:
    python build_manifest.py --raw-dir data/raw --out data/manifest.json

Import:
    from pathlib import Path
    from build_manifest import build_manifest, save_manifest

    entries = build_manifest(Path("data/raw"))
    save_manifest(entries, Path("data/manifest.json"))
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# 3GPP version-code first character -> release number.
# 0-9 map to themselves; a-z map to 10-35 (a=10, b=11, ... j=19, k=20, ...)
def decode_release_char(ch: str) -> int:
    ch = ch.lower()
    if ch.isdigit():
        return int(ch)
    if "a" <= ch <= "z":
        return 10 + (ord(ch) - ord("a"))
    raise ValueError(f"Unrecognised version-code character: {ch!r}")


FILENAME_RE = re.compile(r"^(\d{5})-([0-9a-zA-Z]{3})\.zip$")


def parse_zip_filename(filename: str):
    """
    '23501-ic0.zip' -> {
        'spec_id': '23.501',
        'series': '23',
        'release_num': 18,
        'release_label': 'Rel-18',
        'version': '18.12.0',
    }
    """
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
    """Scan raw_dir recursively for *.zip files and return a manifest list."""
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
    """Write manifest entries to JSON. Returns out_path for chaining."""
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
