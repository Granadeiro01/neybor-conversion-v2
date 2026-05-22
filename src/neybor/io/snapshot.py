"""Snapshot freezing — hash CSV files and write a manifest.

The thesis claims reproducibility from the production state at a specific timestamp.
This module makes that claim verifiable: re-running the pipeline against a snapshot
will produce identical results, and the manifest catches accidental changes (e.g.
Excel re-saving a CSV and silently shifting column types).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 16) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(snapshot_dir: Path, comment: str | None = None) -> Path:
    """Walk snapshot_dir and write SNAPSHOT.json with hash + size for every CSV."""
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot_dir}")

    files_info = []
    for path in sorted(snapshot_dir.glob("*.csv")):
        files_info.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "snapshot_dir": str(snapshot_dir),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "comment": comment or "",
        "files": files_info,
    }

    out_path = snapshot_dir / "SNAPSHOT.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def verify_manifest(snapshot_dir: Path) -> tuple[bool, list[str]]:
    """Re-hash files and compare to the manifest. Returns (ok, list_of_problems)."""
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / "SNAPSHOT.json"
    if not manifest_path.exists():
        return False, [f"No SNAPSHOT.json in {snapshot_dir}. Run write_manifest() first."]

    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []

    for entry in manifest["files"]:
        path = snapshot_dir / entry["name"]
        if not path.exists():
            problems.append(f"Missing file: {entry['name']}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            problems.append(
                f"Hash mismatch for {entry['name']}: "
                f"expected {entry['sha256'][:12]}..., got {actual_hash[:12]}..."
            )

    return len(problems) == 0, problems
