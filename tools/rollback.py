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
    """Loads the run state from a JSON metadata file.

    Args:
        metadata_path (Path): The path to the `run_metadata.json` file.

    Returns:
        dict: The loaded metadata as a dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def unload_adapters(run_dir: Path) -> None:
    """Removes any existing adapter files from the run directory.

    This is a placeholder for a more sophisticated adapter cleanup process. It
    currently removes the contents of the `adapters` directory.

    Args:
        run_dir (Path): The main directory for the scaling run.
    """
    adapters_path = run_dir / "adapters"
    if adapters_path.exists():
        # Placeholder for actual adapter cleanup.
        for item in adapters_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)


def restore_checkpoint(checkpoint_dir: Path, target_dir: Path) -> None:
    """Restores a model checkpoint from a source to a target directory.

    Args:
        checkpoint_dir (Path): The directory containing the checkpoint to restore.
        target_dir (Path): The directory where the checkpoint will be restored to.

    Raises:
        FileNotFoundError: If the checkpoint directory does not exist.
    """
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
    """Performs the rollback to a specified phase.

    This function orchestrates the rollback by:
    1. Loading the run metadata.
    2. Unloading the current adapters.
    3. Restoring the checkpoint files for the target phase.
    4. Updating the metadata to reflect the new current phase.

    Args:
        run_dir (Path): The main directory for the scaling run.
        phase (int): The phase number to roll back to.

    Raises:
        ValueError: If the target phase is not earlier than the current phase.
    """
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
    """Parses command-line arguments for the rollback script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Rollback Scaler Wizard run to a previous phase.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing checkpoints.")
    parser.add_argument("--target-phase", type=int, required=True, help="Phase index to roll back to.")
    return parser.parse_args()


def main() -> int:
    """The main entry point for the rollback script.

    Parses arguments, calls the rollback logic, and handles exceptions.

    Returns:
        int: An exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    try:
        rollback_phase(args.run_dir, args.target_phase)
    except Exception as exc:
        print(f"Rollback failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
