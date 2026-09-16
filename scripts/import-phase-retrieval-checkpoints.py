"""Import an immutable bundle of converged presentation checkpoints."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multi_distance_phase_retrival_huhn.experiment_io import DEFAULT_DATASET, import_checkpoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    provenance = import_checkpoints(args.source, args.cache, args.dataset)
    print(f"Imported {len(provenance['checkpoint_sha256'])} unchanged checkpoints into {args.cache}")
