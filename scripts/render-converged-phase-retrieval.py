"""Regenerate every presentation plot from the converged native-field runs."""

import hashlib
import argparse
import os
import json
from pathlib import Path
import struct
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-retrieval-mpl")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
from multi_distance_phase_retrival_huhn import comparison_plots as details
from multi_distance_phase_retrival_huhn.experiment_io import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT, load_results,
)

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
from multi_distance_phase_retrival_huhn.phase_retrieval import (
    angular_frequency_radius, ctf_transfer_functions, huhn_regularization_filter,
    make_fresnel_propagators,
)
from multi_distance_phase_retrival_huhn.hologram_plots import plot_hologram_line_cuts

CACHE = DEFAULT_CACHE
DATASET = DEFAULT_DATASET
OUTPUT = DEFAULT_OUTPUT
LABELS = {
    "pgd_free": "Free PGD", "pgd_single": "Nonpositive PGD · 1 distance",
    "pgd_multi": "PGD · 4 distances", "ap": "AP · continuous phase",
    "pgd_warm": "PGD + CTF start", "pgd_tikh": "PGD + Tikhonov", "nltikh": "NLTikh · BB",
}
FULL_FILES = ("01-ctf-phase", "02-single-distance-phase", "03-distance-comparison-phase",
              "04-optimizer-phase", "05-nltikh-phase")
COLORS = ("#666666", "#0072B2", "#A04F00", "#7B4B91")
ASSETS = []


def save_figure(figure, name, **data):
    path = OUTPUT / f"{name}.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    ASSETS.append({"file": path.name, **data})


def phase_comparisons(maps, metrics):
    nonlinear_limits = details.nonlinear_phase_limits(maps)
    for filename, (_, title, methods) in zip(FULL_FILES, details.GROUPS, strict=True):
        figure, asset = details.phase_comparison_figure(maps, metrics, methods,
            title.replace(" · tight ROI", " · full field"), nonlinear_limits)
        save_figure(figure, filename, **asset)


def direct_curves(axis, curves, *, log_x=True, log_y=True, ylabel, xlabel):
    endpoints, xx, yy = [], [], []
    for index, (label, x, y) in enumerate(curves):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if log_y:
            good &= y > 0
        x, y = x[good], y[good]
        color = COLORS[index % len(COLORS)]
        axis.plot(x, y, color=color, lw=1.7, linestyle=("-", "--", "-.", ":")[index % 4])
        axis.plot(x[-1], y[-1], "o", color=color, ms=4)
        endpoints.append((label, x[-1], y[-1], color))
        xx.extend(x)
        yy.extend(y)
    if log_x:
        axis.set_xscale("symlog", linthresh=10, linscale=0.5)
    if log_y:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.spines["bottom"].set_bounds(min(xx), max(xx))
    axis.spines["left"].set_bounds(min(yy), max(yy))
    axis.margins(x=0.04, y=0.1)
    if log_x:
        lower, upper = min(xx), max(xx)
        factors = (1, 2, 5) if upper / max(lower, 1) < 100 else (1,)
        ticks = [0] if lower == 0 else []
        ticks += [factor * 10**power for power in range(0, 8) for factor in factors
                  if max(lower, 10) <= factor * 10**power <= upper]
        axis.xaxis.set_major_locator(ticker.FixedLocator(ticks))
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))
    axis.figure.canvas.draw()
    # Place direct labels in a reserved column and separate close endpoints.
    y_low, y_high = axis.get_ylim()
    transform = np.log10 if log_y else lambda value: value
    low, high = transform(y_low), transform(y_high)
    positions = [(transform(point[2]) - low) / (high - low) for point in endpoints]
    order = sorted(range(len(positions)), key=positions.__getitem__)
    for rank, index in enumerate(order):
        positions[index] = max(0.06, positions[index])
        if rank:
            positions[index] = max(positions[index], positions[order[rank - 1]] + 0.105)
    overshoot = max(positions) - 0.94
    if overshoot > 0:
        positions = [position - overshoot for position in positions]
    for (label, x, y, color), position in zip(endpoints, positions, strict=True):
        axis.annotate(label, (x, y), xytext=(1.025, position), textcoords="axes fraction",
                      ha="left", va="center", fontsize=11, color=color,
                      arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7})


def convergence_figure(metrics, filename, title, methods, *, work=False):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6))
    figure.subplots_adjust(left=0.085, right=0.81, bottom=0.23, top=0.80, wspace=0.95)
    data_curves, stopping_curves = [], []
    for name in methods:
        result = metrics[name]
        history = result["histories"]
        x = history["operator_calls"] if work else np.arange(len(history["data_residual"]))
        data_curves.append((LABELS[name], x, history["data_residual"]))
        if name == "ap":
            checks = result["checks"]
            positions = [check["iteration"] for check in checks]
            if work:
                positions = [history["operator_calls"][index] for index in positions]
            values = [check["max_last_20_updates"] / 1e-5 for check in checks]
        else:
            checks = result["checks"]
            positions = [check["iteration"] for check in checks]
            if work:
                positions = [history["operator_calls"][index] for index in positions]
            values = [check["relative_projected_gradient"] / 1e-3 for check in checks]
        stopping_curves.append((LABELS[name], positions, values))
    xlabel = "Forward / adjoint propagations" if work else "Iteration"
    direct_curves(axes[0], data_curves, xlabel=xlabel, ylabel="Relative intensity residual")
    direct_curves(axes[1], stopping_curves, xlabel=xlabel, ylabel="Stopping measure / tolerance")
    axes[0].set_title("Agreement with measured intensities", fontsize=13)
    axes[1].set_title("Convergence threshold reached", fontsize=13)
    axes[1].axhline(1, color="#777", lw=1, ls="--")
    axes[1].text(0.04, 1, " threshold", transform=axes[1].get_yaxis_transform(), fontsize=10, va="bottom")
    figure.suptitle(title, x=0.03, ha="left", fontsize=20)
    note = "PGD / NLTikh: projected-gradient ratio ≤ 10⁻³"
    if "ap" in methods:
        note += " · AP: last 20 relative phase updates ≤ 10⁻⁵"
    note += "\nAll: 100-step objective change / E(0) ≤ 10⁻⁴ · horizontal axes use a log scale above 10"
    figure.text(0.04, 0.035, note, fontsize=9, va="bottom")
    save_figure(figure, filename, methods=methods, x="propagation work" if work else "iterations")


def radial_average(radius, values, bins=72):
    radius, values = np.asarray(radius).ravel(), np.asarray(values).ravel()
    edges = np.linspace(0, radius.max(), bins + 1)
    indices = np.clip(np.digitize(radius, edges) - 1, 0, bins - 1)
    counts = np.bincount(indices, minlength=bins)
    sums = np.bincount(indices, weights=values, minlength=bins)
    return (edges[1:] + edges[:-1]) / 2, sums / np.maximum(counts, 1)


def data_and_filters():
    with np.load(DATASET) as archive:
        holograms = archive["holograms"].copy()
        numbers = tuple(float(value) for value in archive["fresnelNumbers"])
    figure = plot_hologram_line_cuts(holograms, numbers, wavelength=details.WAVELENGTH,
                                    pixel_size=details.PIXEL_SIZE)
    save_figure(figure, "00-measured-holograms", cut_row=1024, band_width=1)
    props = make_fresnel_propagators(holograms.shape[-2:], numbers,
        wavelength=details.WAVELENGTH, pixel_size=details.PIXEL_SIZE)
    radius = angular_frequency_radius(holograms.shape[-2:]).numpy()
    transfers = ctf_transfer_functions(props).numpy()
    curves = [(f"{count} distance{'s' if count > 1 else ''}",
               *radial_average(radius, np.square(transfers[:count]).sum(axis=0))) for count in (1, 4)]
    figure, axis = plt.subplots(figsize=(10, 5.7))
    figure.subplots_adjust(left=0.1, right=0.74, bottom=0.15, top=0.85)
    direct_curves(axis, curves, log_x=False, xlabel="Radial angular frequency [rad/pixel]",
                  ylabel="Radial mean of Σ hⱼ²")
    figure.suptitle("Four distances improve CTF conditioning", x=0.03, ha="left", fontsize=20)
    save_figure(figure, "01-ctf-conditioning")
    alpha = huhn_regularization_filter(holograms.shape[-2:], numbers,
                alpha_low=1e-3, alpha_high=1e-1, alpha_beyond_na=8.0).numpy()
    x, y = radial_average(radius, alpha)
    figure, axis = plt.subplots(figsize=(10, 5.7))
    figure.subplots_adjust(left=0.11, right=0.77, bottom=0.25, top=0.82)
    direct_curves(axis, [("α(ξ)", x, y)], log_x=False,
                  xlabel="Radial angular frequency [rad/pixel]", ylabel="Tikhonov weight")
    first_maximum = np.pi * np.sqrt(2 * np.mean(numbers))
    na_cutoff = np.pi * min(holograms.shape[-2:]) * np.mean(numbers)
    axis.axvline(first_maximum, color="#777", lw=0.8, ls="--")
    axis.annotate("First CTF maximum", (first_maximum, 0.055), xytext=(45, -45),
                  textcoords="offset points", fontsize=12, arrowprops={"arrowstyle": "-", "color": "#777"})
    # For this native field the NA cutoff is beyond the sampled radial range.
    figure.text(0.1, 0.05, f"αlow = 0.001 · αhigh = 0.1 · αbeyond NA = 8\n"
                f"NA cutoff = {na_cutoff:.2f} rad/pixel; sampled radial maximum = {radius.max():.2f} rad/pixel", fontsize=11)
    figure.suptitle("Frequency regularization suppresses weakly constrained modes", x=0.03, ha="left", fontsize=18)
    save_figure(figure, "05-nltikh-filter", numerical_aperture_cutoff=na_cutoff)


def main():
    global CACHE, DATASET, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi", nargs=3, type=int, default=details.DEFAULT_ROI,
                        metavar=("TOP", "LEFT", "SIZE"))
    args = parser.parse_args()
    CACHE, DATASET, OUTPUT = args.cache.resolve(), args.dataset.resolve(), args.output.resolve()
    ASSETS.clear()
    torch.set_num_threads(4)
    maps, metrics, hashes = load_results(CACHE, DATASET)
    details.configure_plots()
    metadata = {"source_hashes": hashes, "metrics": metrics,
        "run": "native-resolution reconstructions to numerical convergence",
        "convergence": {"relative_projected_gradient_tolerance": 1e-3,
            "ap_relative_phase_update_tolerance": 1e-5, "check_every": 100,
            "objective_change_over_100_steps_relative_to_zero": 1e-4,
            "ctf_admm_primal_and_dual_tolerance": 1e-3},
        "precision": "float32 / complex64", "phase_ground_truth": None,
        "measurement_residual": "|| |P exp(i phi)|^2 - I ||_2 / ||I - 1||_2",
        "measurement_residual_scope": "Each method's own measured planes; nonlinear model for every panel, including CTF.",
        "ap_correction": "Principal arg followed by clamping caused jumps from below -pi to zero. The rerun tracks phase increments continuously before applying phi <= 0. The original presentation AP run used the principal branch and was replaced by the continuous-phase run.",
        "checkpoint_provenance": (json.loads((CACHE / "imported-provenance.json").read_text())
            if (CACHE / "imported-provenance.json").exists() else "current-source run"),
        "runner_sha256": hashlib.sha256((ROOT / "scripts/converge-phase-retrieval.py").read_bytes()).hexdigest()}
    details.render(maps, metadata, tuple(args.roi), dataset=DATASET, output=OUTPUT)
    details_metadata = json.loads((OUTPUT / "details-manifest.json").read_text())
    ASSETS.extend(details_metadata["figures"])
    phase_comparisons(maps, metrics)
    for filename, title, methods, work in (
        ("02-single-distance-convergence", "Single-distance PGD · converged runs", ["pgd_free", "pgd_single"], False),
        ("03-distance-comparison-convergence", "One versus four distances · converged PGD", ["pgd_single", "pgd_multi"], False),
        ("04-optimizer-iterations", "AP versus PGD · convergence by iteration", ["pgd_multi", "ap"], False),
        ("04-optimizer-work", "AP versus PGD · convergence by propagation work", ["pgd_multi", "ap"], True),
        ("05-nltikh-convergence", "Nonlinear Tikhonov ablation · converged runs", ["pgd_multi", "pgd_warm", "pgd_tikh", "nltikh"], True),
    ):
        convergence_figure(metrics, filename, title, methods, work=work)
    data_and_filters()
    # This legacy filename remains usable by any saved slide links.
    (OUTPUT / "05-full-field-close-up.png").write_bytes((OUTPUT / "05-full-field-paper-roi.png").read_bytes())
    final_asset = next(asset for asset in ASSETS if asset["file"] == "05-full-field-paper-roi.png")
    ASSETS.append({**final_asset, "file": "05-full-field-close-up.png", "alias_of": "05-full-field-paper-roi.png"})
    for asset in ASSETS:
        contents = (OUTPUT / asset["file"]).read_bytes()
        asset["sha256"] = hashlib.sha256(contents).hexdigest()
        asset["width"], asset["height"] = struct.unpack(">II", contents[16:24])
    manifest = {**details_metadata, "figures": ASSETS,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "details_renderer_sha256": hashlib.sha256(Path(details.__file__).read_bytes()).hexdigest()}
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    manifest = clean(manifest)
    for filename in ("manifest.json", "details-manifest.json"):
        (OUTPUT / filename).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    summary = {name: {key: value for key, value in result.items() if key != "histories"}
               for name, result in metrics.items()}
    (OUTPUT / "convergence-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Regenerated {len(ASSETS)} figures from converged native phase arrays.")


if __name__ == "__main__":
    main()
