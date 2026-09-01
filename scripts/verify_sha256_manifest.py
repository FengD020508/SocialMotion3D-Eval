#!/usr/bin/env python3
"""Verify a JSON SHA-256 manifest after moving its payload to another host."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root-name", default="runpod_private_payload")
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for row in rows:
        parts = PureWindowsPath(row["Path"]).parts
        try:
            relative = Path(*parts[parts.index(args.source_root_name) + 1 :])
        except ValueError as error:
            raise ValueError(f"source root not found in manifest path: {row['Path']}") from error
        target = args.root / relative
        actual = _digest(target) if target.is_file() else None
        if actual != row["Hash"].upper():
            failures.append({"path": str(target), "expected": row["Hash"], "actual": actual})
    print(json.dumps({"checked": len(rows), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
