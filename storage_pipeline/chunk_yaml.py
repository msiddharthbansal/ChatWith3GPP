"""
chunk_yaml.py

Splits a 3GPP OpenAPI service-definition YAML directory (e.g. the
Namf_*.yaml files for TS 29.518) into operation- and schema-level chunks,
preserving raw YAML structure. Usable two ways:

CLI:
    python chunk_yaml.py \\
        --yaml-dir data/extracted/rel18/29518-ie0 \\
        --spec-id 29.518 --release "Rel-18" --version 18.14.0 \\
        --out data/processed/rel18/29.518_api.jsonl

Import:
    from pathlib import Path
    from chunk_yaml import chunk_yaml_directory, write_chunks_jsonl

    chunks = chunk_yaml_directory(
        Path("data/extracted/rel18/29518-ie0"),
        spec_id="29.518", release="Rel-18", version="18.14.0",
    )
    write_chunks_jsonl(chunks, Path("data/processed/rel18/29.518_api.jsonl"))
"""
import json
import re
from pathlib import Path

import yaml


def service_name_from_filename(path: Path) -> str:
    """'TS29518_Namf_Communication.yaml' -> 'Namf_Communication'"""
    stem = path.stem
    m = re.search(r"(Namf_[A-Za-z]+)", stem)
    return m.group(1) if m else stem


def chunk_yaml_file(path: Path, spec_id: str, release: str, version: str):
    """Chunk a single OpenAPI YAML file into operation- and schema-level chunks."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"  [warn] failed to parse {path.name}: {e}")
            return []

    if not isinstance(doc, dict):
        return []

    service = service_name_from_filename(path)
    chunks = []

    # One chunk per path + HTTP method (the actual API operations)
    for api_path, methods in (doc.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for http_method, operation in methods.items():
            if http_method.startswith("x-") or not isinstance(operation, dict):
                continue
            snippet = {api_path: {http_method: operation}}
            chunks.append(
                {
                    "spec_id": spec_id,
                    "release": release,
                    "version": version,
                    "service": service,
                    "section_type": "operation",
                    "operation_path": api_path,
                    "http_method": http_method.upper(),
                    "title": operation.get("summary")
                    or operation.get("operationId")
                    or f"{http_method.upper()} {api_path}",
                    "content": yaml.dump(snippet, sort_keys=False, allow_unicode=True),
                    "chunk_type": "openapi_yaml",
                    "source_file": path.name,
                }
            )

    # One chunk per schema definition (data model)
    schemas = ((doc.get("components") or {}).get("schemas")) or {}
    for schema_name, schema_def in schemas.items():
        snippet = {schema_name: schema_def}
        chunks.append(
            {
                "spec_id": spec_id,
                "release": release,
                "version": version,
                "service": service,
                "section_type": "schema",
                "operation_path": None,
                "http_method": None,
                "title": schema_name,
                "content": yaml.dump(snippet, sort_keys=False, allow_unicode=True),
                "chunk_type": "openapi_yaml",
                "source_file": path.name,
            }
        )

    return chunks


def chunk_yaml_directory(yaml_dir: Path, spec_id: str, release: str, version: str):
    """Chunk every .yaml/.yml file in a directory (e.g. all Namf_*.yaml
    service definitions for TS 29.518) and return the combined chunk list."""
    yaml_dir = Path(yaml_dir)
    all_chunks = []
    yaml_files = sorted(yaml_dir.glob("*.yaml")) + sorted(yaml_dir.glob("*.yml"))
    for yf in yaml_files:
        cs = chunk_yaml_file(yf, spec_id, release, version)
        all_chunks.extend(cs)
        print(f"  {yf.name}: {len(cs)} chunks")
    return all_chunks


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

    ap = argparse.ArgumentParser(description="Chunk a 3GPP OpenAPI YAML directory into operation/schema-level JSONL.")
    ap.add_argument("--yaml-dir", type=Path, required=True, help="Folder containing one or more .yaml files")
    ap.add_argument("--spec-id", required=True, help='e.g. "29.518"')
    ap.add_argument("--release", required=True, help='e.g. "Rel-18"')
    ap.add_argument("--version", required=True, help='e.g. "18.14.0"')
    ap.add_argument("--out", type=Path, required=True, help="Output .jsonl path")
    args = ap.parse_args()

    chunks = chunk_yaml_directory(args.yaml_dir, args.spec_id, args.release, args.version)
    write_chunks_jsonl(chunks, args.out)
    print(f"{args.spec_id} ({args.release}) API: {len(chunks)} total chunks -> {args.out}")


if __name__ == "__main__":
    _cli()
