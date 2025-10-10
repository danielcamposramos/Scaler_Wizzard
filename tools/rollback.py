#!/usr/bin/env python3
"""Rollback script for Scaler Wizard phases.

Restores the last known-good checkpoint and unloads adapters when the cockpit
issues a rollback command.  Keeps implementation lightweight so it can be
called from the CLI prototype, the TransformerLab plugin, or manual terminals.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def load_state(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def unload_adapters(run_dir: Path) -> None:
    adapters_path = run_dir / "adapters"
    if adapters_path.exists():
        # Placeholder for actual adapter cleanup.
        for item in adapters_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)


def restore_checkpoint(checkpoint_dir: Path, target_dir: Path) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in checkpoint_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            dest.write_bytes(item.read_bytes())


def rollback_phase(run_dir: Path, phase: int) -> None:
    metadata_path = run_dir / "run_metadata.json"
    metadata = load_state(metadata_path)

    current_phase = metadata.get("phase", 0)
    if phase >= current_phase:
        raise ValueError(f"Cannot roll back to phase {phase}; current phase is {current_phase}.")

    checkpoint_dir = Path(metadata["checkpoints"][f"phase_{phase}"])
    unload_adapters(run_dir)
    restore_checkpoint(checkpoint_dir, run_dir / "model")

    metadata["phase"] = phase
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f'Rollback complete; you are now at phase {phase}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback Scaler Wizard run to a previous phase.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing checkpoints.")
    parser.add_argument("--target-phase", type=int, required=True, help="Phase index to roll back to.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rollback_phase(args.run_dir, args.target_phase)
    except Exception as exc:
        print(f"Rollback failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
