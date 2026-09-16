"""Shared full-field and ROI comparison plots; no reconstruction is run here."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
import numpy as np

from .phase_retrieval import fresnel_numbers_to_distances

WAVELENGTH = 1.5498e-10
PIXEL_SIZE = 196e-9
# Approximate pixel bounds obtained by matching Fig. 1(c) to the native phase map.
DEFAULT_ROI = (583, 870, 192)
PAPER_REGISTRATION = {
    "figure": "Figure 1(c), _page_8_Figure_0.jpeg",
    "panel_xyxy": [37, 568, 466, 998],
    "red_box_xyxy": [246, 690, 286, 730],
    "masked_affine_correlation": 0.9553759,
    "mapped_native_xyxy": [870.15, 582.96, 1061.11, 773.90],
    "note": "Approximate match to the printed figure; the paper supplies no pixel coordinates.",
}

GROUPS = (
    ("01-ctf-roi", "CTF · tight ROI", (
        ("ctf_single", "1 distance · α = 0"),
        ("ctf_single_tikh", "1 distance · α = 0.01"),
        ("ctf_multi", "4 distances · α = 0"),
        ("ctf_multi_tikh", "4 distances · α = 0.01"),
    )),
    ("02-single-distance-roi", "Single-distance retrieval · tight ROI", (
        ("ctf_single", "Linear CTF"),
        ("pgd_free", "Nonlinear · free"),
        ("pgd_single", "Nonlinear · φ ≤ 0"),
    )),
    ("03-distance-comparison-roi", "One versus four distances · tight ROI", (
        ("pgd_single", "1 distance"),
        ("pgd_multi", "4 distances"),
    )),
    ("04-optimizer-roi", "AP versus PGD · tight ROI", (
        ("pgd_multi", "DeepInv PGD"),
        ("ap", "AP · continuous phase"),
    )),
    ("05-nltikh-roi", "Nonlinear Tikhonov ablation · tight ROI", (
        ("pgd_multi", "PGD baseline"),
        ("pgd_warm", "+ constrained CTF start"),
        ("pgd_tikh", "+ frequency Tikhonov"),
        ("nltikh", "+ BB / nonmonotone search"),
    )),
)

NONLINEAR_METHODS = ("pgd_free", "pgd_single", "pgd_multi", "ap", "pgd_warm", "pgd_tikh", "nltikh")
PHASE_COLORMAP = "magma"


def configure_plots():
    plt.rcParams.update({
        "figure.facecolor": "#fffff8", "axes.facecolor": "#fffff8",
        "savefig.facecolor": "#fffff8", "font.family": "serif", "font.size": 14,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlelocation": "left",
    })


def phase_limits(arrays):
    values = np.concatenate([array.ravel() for array in arrays])
    low, high = np.quantile(values, [0.01, 0.99])
    if high <= max(1e-8, 0.03 * abs(low)):
        high = 0.0
    return float(low), float(high)


def nonlinear_phase_limits(maps):
    """One full-field range for every iterative method, including positive free PGD.

    Round outward to whole radians. Using extrema keeps every nonlinear value
    visible; neither comparison membership nor ROI bounds can change the scale.
    """
    arrays = [np.asarray(maps[name]) for name in NONLINEAR_METHODS]
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonlinear phase maps must be finite 2D arrays")
    low = float(np.floor(min(array.min() for array in arrays)))
    high = float(np.ceil(max(array.max() for array in arrays)))
    return low, high if high > low else low + 1.0


def phase_scale_note(limits):
    return f"Nonlinear phase: {limits[0]:g} to {limits[1]:g} rad across all full-field and ROI views"


def colorbar_extension(arrays, limits):
    below = any(array.min() < limits[0] for array in arrays)
    above = any(array.max() > limits[1] for array in arrays)
    return "both" if below and above else "min" if below else "max" if above else "neither"


def phase_comparison_figure(maps, metrics, methods, title, nonlinear_limits, *, roi=None):
    """Render full-field and ROI comparisons through the same scale policy."""
    names = [name for name, _ in methods]
    nonlinear = [name for name in names if name in NONLINEAR_METHODS]
    linear = [name for name in names if name not in NONLINEAR_METHODS]
    mixed = bool(linear and nonlinear)
    linear_limits = phase_limits([maps[name] for name in linear]) if linear else None
    limits = {name: nonlinear_limits if name in nonlinear else linear_limits for name in names}
    bounds = np.s_[:, :] if roi is None else np.s_[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[2]]
    arrays = [maps[name][bounds] for name in names]
    figure = plt.figure(figsize=(11.5, 6.0), layout="constrained")
    if mixed:
        # CTF gets a separate horizontal colorbar; the two nonlinear panels share
        # the same bar and normalization used on every other nonlinear slide.
        grid = figure.add_gridspec(2, len(methods), height_ratios=[1, 0.045])
        axes = [figure.add_subplot(grid[0, index]) for index in range(len(methods))]
    else:
        rows, cols = (2, 2) if len(methods) == 4 else (1, len(methods))
        grid = figure.add_gridspec(rows, cols + 1, width_ratios=[1] * cols + [0.045])
        axes = [figure.add_subplot(grid[index // cols, index % cols]) for index in range(len(methods))]
    pictures = {}
    for index, ((name, label), array, axis) in enumerate(zip(methods, arrays, axes, strict=True)):
        low, high = limits[name]
        picture = axis.imshow(array, cmap=PHASE_COLORMAP, vmin=low, vmax=high, interpolation="nearest")
        pictures[name] = picture
        count = metrics[name].get("iterations", 0)
        stop = f"{count:,} iterations" if count else "direct inverse"
        if roi is None:
            stop += f" · residual {metrics[name]['residual']:.4f}"
        axis.set_title(f"{label}\n{stop}", fontsize=12)
        axis.set_axis_off()
        if index == 0:
            scale_bar(axis, array.shape[0], length_um=10 if roi else 50)
    if mixed:
        for group, label in ((linear, "CTF phase [rad] · separate scale"),
                             (nonlinear, "Nonlinear phase [rad] · shared across slides")):
            indices = [names.index(name) for name in group]
            figure.colorbar(pictures[group[0]], cax=figure.add_subplot(grid[1, min(indices):max(indices) + 1]),
                            orientation="horizontal", label=label,
                            extend=colorbar_extension([maps[name] for name in group], limits[group[0]]))
    else:
        figure.colorbar(pictures[names[0]], cax=figure.add_subplot(grid[:, -1]), label="phase [rad]",
                        extend=colorbar_extension([maps[name] for name in names], limits[names[0]]))
    figure.suptitle(title, x=0.03, ha="left", fontsize=20)
    context = (f"Same {roi[2]} × {roi[2]} pixel ROI · {roi[2] * PIXEL_SIZE * 1e6:.1f} μm square" if roi
               else "2048 × 1920 pixels · residuals use the nonlinear intensity model")
    if nonlinear:
        context += "\n" + phase_scale_note(nonlinear_limits)
    figure.supxlabel(context, fontsize=10)
    asset = {"methods": names, "colormap": PHASE_COLORMAP,
             "color_limits_rad_by_method": {name: list(limits[name]) for name in names}}
    if not mixed:
        asset["color_limits_rad"] = list(limits[names[0]])
    if nonlinear:
        asset["nonlinear_color_limits_rad"] = list(nonlinear_limits)
    if linear:
        asset["ctf_color_limits_rad"] = list(linear_limits)
    return figure, asset


def scale_bar(axis, size, *, length_um=10):
    length = length_um * 1e-6 / PIXEL_SIZE
    left, bottom = size * 0.08, size * 0.91
    outline = [path_effects.withStroke(linewidth=4, foreground="#111")]
    axis.plot([left, left + length], [bottom, bottom], color="#fffff8", lw=2,
              path_effects=outline)
    axis.text(left, bottom - size * 0.025, f"{length_um} μm", color="#fffff8", fontsize=10,
              path_effects=[path_effects.withStroke(linewidth=2, foreground="#111")])


def render(maps, metadata, roi, *, dataset, output):
    OUTPUT = Path(output)
    DATASET = Path(dataset)
    configure_plots()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    top, left, size = roi
    bounds = np.s_[top:top + size, left:left + size]
    height, width = maps["nltikh"].shape
    if size < 1 or top < 0 or left < 0 or top + size > height or left + size > width:
        raise ValueError("ROI is outside the native reconstruction.")
    nonlinear_limits = nonlinear_phase_limits(maps)
    assets = []
    for filename, title, methods in GROUPS:
        figure, asset = phase_comparison_figure(maps, metadata["metrics"], methods, title,
                                               nonlinear_limits, roi=roi)
        figure.savefig(OUTPUT / f"{filename}.png", dpi=200)
        plt.close(figure)
        assets.append({"file": f"{filename}.png", **asset})

    with np.load(DATASET) as archive:
        measurement = archive["holograms"][0]
        number = float(archive["fresnelNumbers"][0])
    distance = fresnel_numbers_to_distances((number,), WAVELENGTH, PIXEL_SIZE)[0]
    figure, axis = plt.subplots(figsize=(8.0, 7.6), layout="constrained")
    low, high = np.quantile(measurement, [0.01, 0.99])
    image = axis.imshow(measurement, cmap="gray", vmin=low, vmax=high)
    axis.add_patch(Rectangle((left - 0.5, top - 0.5), size, size,
                            fill=False, edgecolor="#D55E00", linewidth=1.4))
    axis.set_title(f"Measured hologram · first distance\n2048 × 1920 pixels · z = {distance * 1000:.2f} mm",
                   fontsize=16)
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025, label="normalized intensity")
    figure.savefig(OUTPUT / "00-single-distance-data.png", dpi=220)
    plt.close(figure)
    assets.append({"file": "00-single-distance-data.png", "plane": 0,
                   "intensity_display_limits": [float(low), float(high)]})

    figure = plt.figure(figsize=(11.5, 6.0), layout="constrained")
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 0.045])
    full_axis, detail_axis = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    low, high = nonlinear_limits
    image = full_axis.imshow(maps["nltikh"], cmap=PHASE_COLORMAP, vmin=low, vmax=high,
                             interpolation="nearest")
    full_axis.add_patch(Rectangle((left - 0.5, top - 0.5), size, size,
                                 fill=False, edgecolor="#D55E00", linewidth=1.4))
    full_axis.set_title("Full field · 2048 × 1920", fontsize=15)
    detail_axis.imshow(maps["nltikh"][bounds], cmap=PHASE_COLORMAP, vmin=low, vmax=high,
                       interpolation="nearest")
    detail_axis.set_title(f"Matched Figure 1 ROI · {size} × {size}", fontsize=15)
    for axis in (full_axis, detail_axis):
        axis.set_axis_off()
    scale_bar(detail_axis, size)
    figure.colorbar(image, cax=figure.add_subplot(grid[:, -1]), label="phase [rad]")
    figure.suptitle("Full reconstruction · nonlinear Tikhonov", fontsize=20, x=0.03, ha="left")
    if metadata.get("run"):
        count = metadata["metrics"]["nltikh"]["iterations"]
        figure.supxlabel(f"{count:,} nonlinear iterations · numerical stopping criteria reached\n"
                         + phase_scale_note(nonlinear_limits), fontsize=10)
    figure.savefig(OUTPUT / "05-full-field-paper-roi.png", dpi=200)
    plt.close(figure)
    assets.append({"file": "05-full-field-paper-roi.png", "methods": ["nltikh"],
                   "colormap": PHASE_COLORMAP, "color_limits_rad": [low, high],
                   "nonlinear_color_limits_rad": [low, high],
                   "color_limits_rad_by_method": {"nltikh": [low, high]}})

    metadata = {**metadata, "roi": {"top": top, "left": left, "height": size, "width": size,
                 "indexing": "zero-based, end-exclusive", "size_um": size * PIXEL_SIZE * 1e6},
                "roi_reference": ("Huhn et al. (2022), Figure 1 red-dashed region; approximately matched to the measured data."
                                  if tuple(roi) == DEFAULT_ROI else "Custom ROI"),
                "paper_registration": PAPER_REGISTRATION,
                "rendering": "Native phase arrays cropped after full-field reconstruction. All iterative nonlinear methods share one full-field range, rounded outward to whole radians, across comparisons and ROIs. CTF uses separate full-field 1–99% limits; colorbar extensions mark saturation.",
                "nonlinear_phase_scale": {"colormap": PHASE_COLORMAP,
                    "color_limits_rad": list(nonlinear_limits), "methods": list(NONLINEAR_METHODS),
                    "policy": "Extrema across every nonlinear full-field array, rounded outward to whole radians; no nonlinear clipping, phase shift, or ROI renormalization."},
                "figures": assets}
    (OUTPUT / "details-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Rendered {len(assets)} figures for ROI {roi}.", flush=True)

