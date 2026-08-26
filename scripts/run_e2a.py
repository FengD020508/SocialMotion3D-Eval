#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from socialmotion3d_eval.e2a import run_e2a


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E2a fixed-human controlled ego pilot")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_e2a(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

