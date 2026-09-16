"""Compare regularized four-distance CTF with the full Huhn-style NLTikh run.

Read validated checkpoints only. No reconstruction, phase offset adjustment,
or experimental-data export is performed.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-retrieval-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from multi_distance_phase_retrival_huhn.comparison_plots import (
    DEFAULT_ROI, PHASE_COLORMAP, PIXEL_SIZE, configure_plots,
    nonlinear_phase_limits, scale_bar,
)
from multi_distance_phase_retrival_huhn.experiment_io import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT, load_results, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi", nargs=3, type=int, default=DEFAULT_ROI,
                        metavar=("TOP", "LEFT", "SIZE"))
    args = parser.parse_args()
    maps, metrics, hashes = load_results(args.cache, args.dataset)
    names = ("ctf_multi_tikh", "nltikh")
    if any(metrics[name]["planes"] != 4 for name in names):
        raise ValueError("This comparison requires the same four measured distances")
    top, left, size = args.roi
    height, width = maps[names[0]].shape
    if size < 1 or min(top, left) < 0 or top + size > height or left + size > width:
        raise ValueError("ROI is outside the reconstructed field")
    crop = np.s_[top:top + size, left:left + size]
    low, high = nonlinear_phase_limits(maps)
    # Unlike the unregularized CTF comparisons, this regularized result fits
    # comfortably in the nonlinear display range. Expand only if necessary.
    low = min(low, float(np.floor(maps[names[0]].min())))
    high = max(high, float(np.ceil(maps[names[0]].max())))
    configure_plots()
    figure = plt.figure(figsize=(11.5, 3.9), layout="constrained")
    grid = figure.add_gridspec(2, 5, width_ratios=[1, 1.45, 1, 1.45, 0.07],
                              height_ratios=[0.15, 1])
    labels = ("Regularized CTF · 4 distances", "Huhn-style nonlinear Tikhonov")
    pictures = []
    for index, (name, label) in enumerate(zip(names, labels, strict=True)):
        header = figure.add_subplot(grid[0, 2*index:2*index+2])
        header.text(0.5, 0.5, label, ha="center", va="center", fontsize=16)
        header.set_axis_off()
        phase = maps[name]
        for detail in (False, True):
            axis = figure.add_subplot(grid[1, 2*index + int(detail)])
            array = phase[crop] if detail else phase
            picture = axis.imshow(array, cmap=PHASE_COLORMAP, vmin=low, vmax=high,
                                  interpolation="nearest")
            pictures.append(picture)
            axis.set_title("Same sphere ROI" if detail else "Full field", fontsize=12)
            axis.set_axis_off()
            if not detail:
                axis.add_patch(Rectangle((left - 0.5, top - 0.5), size, size,
                    fill=False, edgecolor="#D55E00", linewidth=1.2))
            scale_bar(axis, array.shape[0], length_um=10 if detail else 50)
    figure.colorbar(pictures[0], cax=figure.add_subplot(grid[1, -1]),
                   label="phase [rad] · shared scale", ticks=np.arange(low, high + 1, 1))
    # Full fields and both crops must retain exactly the same normalization.
    assert all(picture.get_clim() == (low, high) for picture in pictures)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "06-ctf-vs-nltikh-summary.png"
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    residuals = {name: metrics[name]["residual"] for name in names}
    record = {
        "file": destination.name, "sha256": sha256(destination),
        "renderer_sha256": sha256(Path(__file__)),
        "source_hashes": hashes,
        "checkpoint_sha256": {name: sha256(args.cache / f"{name}.npz") for name in names},
        "methods": list(names), "planes": 4,
        "phase_scale": {"colormap": PHASE_COLORMAP, "limits_rad": [low, high],
                        "policy": "Shared full-field range; no clipping or offset adjustment"},
        "phase_reference": {"ctf_multi_tikh": "zero mean (unobservable DC set to zero)",
                            "nltikh": "stored nonpositive constrained phase"},
        "roi": {"top": top, "left": left, "size": size,
                "indexing": "zero-based, end-exclusive", "width_um": size*PIXEL_SIZE*1e6},
        "relative_intensity_residual": residuals,
        "residual_ratio_ctf_over_nltikh": residuals[names[0]] / residuals[names[1]],
        "residual_definition": "|| |P exp(i phi)|^2 - I ||_2 / ||I - 1||_2, same four planes",
        "phase_ground_truth": None,
        "nltikh_iterations": metrics["nltikh"]["iterations"],
        "ctf_initialization_iterations": metrics["ctf_warm"]["iterations"],
        "comparison_scope": "Best of the four direct CTF baselines shown versus the full "
                            "Huhn-style method. Model, priors, and optimization all change. "
                            "Not a claim of optimal hyperparameters or minimum phase error.",
    }
    (args.output / "method-summary-manifest.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
