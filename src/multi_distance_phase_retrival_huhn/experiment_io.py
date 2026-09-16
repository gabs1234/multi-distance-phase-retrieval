"""Paths, checkpoint validation, and provenance for the measured comparison."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT / "holograms_beads_updated.npz"
DEFAULT_CACHE = PROJECT / ".cache/phase-retrieval/converged"
DEFAULT_OUTPUT = PROJECT / "figures/comparison"
RUNS = ("ctf_single", "ctf_single_tikh", "ctf_multi", "ctf_multi_tikh",
        "ctf_warm", "pgd_free", "pgd_single", "pgd_multi", "ap_continuous",
        "pgd_warm", "pgd_tikh", "nltikh")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(dataset=DEFAULT_DATASET):
    # Plotting and notebook prose do not invalidate scientific checkpoints.
    names = ("src/multi_distance_phase_retrival_huhn/phase_retrieval.py",
             "src/multi_distance_phase_retrival_huhn/experiment_io.py",
             "scripts/converge-phase-retrieval.py")
    return {"dataset": sha256(dataset),
            **{name: sha256(PROJECT / name) for name in names},
            **{name + "_version": version(name) for name in ("torch", "deepinv", "numpy")}}


def validate_checkpoint(path, metadata, dataset, *, current_only=False, hashes=None):
    path = Path(path)
    bundle = path.parent / "imported-provenance.json"
    if bundle.exists():
        if current_only:
            raise ValueError("Imported checkpoints are immutable historical results; "
                             "use a separate --cache directory for fresh solver runs")
        provenance = json.loads(bundle.read_text())
        if provenance["dataset_sha256"] != sha256(dataset):
            raise ValueError(f"Dataset changed since checkpoint import: {path}")
        if provenance["checkpoint_sha256"].get(path.name) != sha256(path):
            raise ValueError(f"Imported checkpoint changed: {path}")
    elif metadata["source_hashes"] != (hashes or fingerprint(dataset)):
        raise ValueError(f"Stale checkpoint: {path}; use a new cache directory")


def load_results(cache=DEFAULT_CACHE, dataset=DEFAULT_DATASET):
    """Load certified full-field maps, retaining their original run metadata."""
    cache, dataset = Path(cache), Path(dataset)
    with np.load(dataset) as archive:
        shape = archive["holograms"].shape[-2:]
    hashes = None if (cache / "imported-provenance.json").exists() else fingerprint(dataset)
    maps, metrics = {}, {}
    for run in RUNS:
        path = cache / f"{run}.npz"
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            validate_checkpoint(path, metadata, dataset, hashes=hashes)
            phase = archive["phase"].copy()
        if not metadata.get("converged"):
            raise ValueError(f"Refusing unconverged result: {run}")
        if phase.shape != shape or not np.isfinite(phase).all():
            raise ValueError(f"Invalid phase array: {run}")
        if metadata.get("constraint") == "nonpositive" and phase.max() > 0:
            raise ValueError(f"Nonpositive constraint violated: {run}")
        if run == "nltikh":
            metadata["step"] = None
            metadata["initial_step"] = metadata["histories"]["step_size"][0]
            metadata["step_policy"] = "alternating BB with nonmonotone line search"
        name = "ap" if run == "ap_continuous" else run
        maps[name], metrics[name] = phase, metadata
    original_hashes = metrics["nltikh"]["source_hashes"]
    if any(metric["source_hashes"] != original_hashes for metric in metrics.values()):
        raise ValueError("Checkpoint bundle mixes different scientific sources")
    return maps, metrics, original_hashes


def import_checkpoints(source, destination, dataset=DEFAULT_DATASET):
    """Copy existing presentation results without rewriting their provenance.

    An explicit import records byte hashes and the original source fingerprints.
    It enables rendering and verification, never resuming with changed solvers.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Import destination must be new: {destination}")
    data_hash = sha256(dataset)
    records = {}
    for name in RUNS:
        path = source / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if not metadata.get("converged"):
                raise ValueError(f"Cannot import unconverged result: {name}")
            if data_hash not in metadata["source_hashes"].values():
                raise ValueError(f"Dataset does not match {name}")
        records[path.name] = sha256(path)
    provenance = {
        "format_version": 1, "dataset_sha256": data_hash,
        "origin": str(source), "checkpoint_sha256": records,
        "note": "Original arrays and metadata copied byte-for-byte. Historical "
                "results, not claimed to have been computed with the current source. "
                "Use a separate cache for new runs; verification is read-only.",
    }
    destination.mkdir(parents=True)
    for filename in records:
        shutil.copy2(source / filename, destination / filename)
    (destination / "imported-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    load_results(destination, dataset)
    return provenance
