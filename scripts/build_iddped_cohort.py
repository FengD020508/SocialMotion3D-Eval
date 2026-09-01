#!/usr/bin/env python3
"""Create the full IDD-PeD E1a/E2a event and E3 scene manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from socialmotion3d_eval.iddped_cohort import build_cohort, write_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    cohort = build_cohort(args.annotation_root, args.context_frames, args.fps)
    write_cohort(cohort, args.output)
    print(json.dumps(cohort["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
