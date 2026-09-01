#!/usr/bin/env python3
"""Install the tracked IDD-PeD pipeline patches into a GENMO checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genmo-root", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "pipeline" / "genmo" / "scripts"
    destination = args.genmo_root.resolve() / "scripts"
    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    print(f"installed {copied} tracked pipeline files into {destination}")


if __name__ == "__main__":
    main()
