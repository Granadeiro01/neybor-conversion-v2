"""Write SNAPSHOT.json for the configured raw CSV export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neybor.config import SNAPSHOT_DIR
from neybor.io import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Neybor raw CSV snapshot")
    parser.add_argument("--comment", default="Frozen for thesis pipeline run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = write_manifest(SNAPSHOT_DIR, comment=args.comment)
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
