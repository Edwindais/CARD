#!/usr/bin/env python3
"""Convert the CARD selection result into the inference parameter file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cardiac", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def selected_params(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {key: float(value) for key, value in payload["selected"]["params"].items()}


def main() -> None:
    args = parse_args()
    tuples = {
        "acdc": selected_params(args.cardiac),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tuples, indent=2) + "\n")
    print(f"Selected tuples assembled: {args.output}")


if __name__ == "__main__":
    main()
