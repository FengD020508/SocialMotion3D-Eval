#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from socialmotion3d_eval.e3 import run_e3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E3 DROID/MegaSAM vs OBD pilot")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_e3(args.config)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

