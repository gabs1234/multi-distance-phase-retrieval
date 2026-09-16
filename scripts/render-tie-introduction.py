"""Render the measured-data TIE example and the single-distance CTF introduction.

Run with uv run python. Existing converged
reconstructions are read from their checkpoints; no iterative solver is rerun.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-retrieval-mpl")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
sys.path.insert(0, str(PROJECT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.patches import Rectangle
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import torch
from multi_distance_phase_retrival_huhn.phase_retrieval import make_fresnel_propagators

from multi_distance_phase_retrival_huhn.direct_methods import (
    fresnel_phase, tie_reconstruct, homogeneous_ctf_ict, predict_intensity, polystyrene_8kev,
)
from multi_distance_phase_retrival_huhn.experiment_io import (
    DEFAULT_CACHE, DEFAULT_DATASET, DEFAULT_OUTPUT, validate_checkpoint,
)

OUTPUT = DEFAULT_OUTPUT
CACHE = DEFAULT_CACHE.parent / "direct"
CONVERGED_CACHE = DEFAULT_CACHE
DATASET = DEFAULT_DATASET
PIXEL = 196e-9
WAVELENGTH = 1.5498e-10
ROI = (583, 870, 192)
BACKGROUND = "#fffff8"
ALPHA = .01
ASSETS = []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "font.size": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.titlelocation": "left",
        "axes.edgecolor": "#888888", "xtick.direction": "in",
        "ytick.direction": "in", "svg.fonttype": "path",
    })


def limits(arrays):
    low, high = np.quantile(np.concatenate([a.ravel() for a in arrays]), [0.01, 0.99])
    return float(low), float(high)


def scale_bar(axis, height, length_um):
    left, bottom = height * 0.06, height * 0.91
    outline = [effects.withStroke(linewidth=3, foreground="#111111")]
    axis.plot([left, left + length_um * 1e-6 / PIXEL], [bottom, bottom],
              color=BACKGROUND, lw=2, path_effects=outline)
    axis.text(left, bottom - height * 0.025, f"{length_um} μm", color=BACKGROUND,
              fontsize=11, path_effects=outline)


def save(figure, name, **metadata):
    path = OUTPUT / name
    figure.savefig(path, dpi=200)
    plt.close(figure)
    ASSETS.append({"file": name, "sha256": sha256(path), **metadata})


def tie_figure(phase, observed, predicted, residual, *, zoom):
    top, left, size = ROI
    crop = np.s_[top:top + size, left:left + size] if zoom else np.s_[:, :]
    phase_range = limits([phase])
    intensity_range = limits([observed, predicted])
    fig = plt.figure(figsize=(11.5, 6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.045])
    phase_axis = None
    for index, (array, label, color_range, cmap) in enumerate((
        (phase, "TIE + Tikhonov · α = 0.01", phase_range, "magma"),
        (observed, "Measured hologram", intensity_range, "gray"),
        (predicted, "Fresnel prediction from TIE", intensity_range, "gray"),
    )):
        axis = fig.add_subplot(grid[0, index])
        picture = axis.imshow(array[crop], cmap=cmap, vmin=color_range[0], vmax=color_range[1],
                              interpolation="nearest")
        axis.set_title(label, fontsize=12)
        axis.set_axis_off()
        scale_bar(axis, size if zoom else phase.shape[0], 10 if zoom else 50)
        if not zoom:
            axis.add_patch(Rectangle((left - .5, top - .5), size, size,
                                     fill=False, edgecolor="#E69F00", linewidth=1))
        if index == 0:
            phase_axis = picture
        elif index == 2:
            fig.colorbar(picture, cax=fig.add_subplot(grid[1, 1:]),
                         orientation="horizontal", label="normalized intensity · shared scale")
    fig.colorbar(phase_axis, cax=fig.add_subplot(grid[1, 0]),
                 orientation="horizontal", label="phase [rad]")
    detail = "same sphere ROI" if zoom else "full field"
    fig.suptitle(f"TIE misses fine fringes · {detail}", x=.025, ha="left", fontsize=21)
    fig.supxlabel(f"First distance · full-field nonlinear intensity residual {residual:.3f}"
                  "\nLow frequencies stabilized · no phase ground truth", fontsize=12)
    save(fig, f"00-tie-{'roi' if zoom else 'full'}.png", phase_limits_rad=phase_range,
         intensity_limits=intensity_range, roi=ROI if zoom else None)


def ctf_figure(arrays, residuals, *, zoom):
    top, left, size = ROI
    crop = np.s_[top:top + size, left:left + size] if zoom else np.s_[:, :]
    color_range = limits(arrays)
    fig = plt.figure(figsize=(11.5, 6), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .045])
    for index, (array, label, residual) in enumerate(zip(
            arrays, ("CTF · α = 0", "CTF + Tikhonov · α = 0.01"), residuals, strict=True)):
        axis = fig.add_subplot(grid[0, index])
        picture = axis.imshow(array[crop], cmap="magma", vmin=color_range[0], vmax=color_range[1],
                              interpolation="nearest")
        axis.set_title(f"{label}\nNonlinear intensity residual {residual:.3f}", fontsize=14)
        axis.set_axis_off()
        scale_bar(axis, size if zoom else array.shape[0], 10 if zoom else 50)
        if not zoom:
            axis.add_patch(Rectangle((left - .5, top - .5), size, size,
                                     fill=False, edgecolor="#E69F00", linewidth=1))
    fig.colorbar(picture, cax=fig.add_subplot(grid[0, -1]), label="phase [rad]")
    fig.suptitle(f"Single-distance CTF · {'same sphere ROI' if zoom else 'full field'}",
                 x=.025, ha="left", fontsize=21)
    fig.supxlabel("Same first hologram · direct inverses · shared phase scale"
                  "\nResiduals measure intensity consistency, not phase accuracy", fontsize=12)
    save(fig, f"01-ctf-single-{'roi' if zoom else 'full'}.png", color_limits_rad=color_range,
         methods=["ctf_single", "ctf_single_tikh"], roi=ROI if zoom else None)


def null_plot(numbers):
    # A line cut preserves exact zeros; radial averaging can conceal them.
    u = np.linspace(0, 1.82, 1400)
    fig, axis = plt.subplots(figsize=(9, 2.6), layout="constrained")
    axis.axhline(0, color="#888888", lw=.6)
    for index, color, style in ((0, "#666666", "-"), (3, "#326899", "--")):
        ratio = numbers[0] / numbers[index]
        h = 2 * np.sin(np.pi * ratio * u*u)
        axis.plot(u, h, color=color, ls=style, lw=1.7)
        roots = np.sqrt(np.arange(4) / ratio)
        axis.plot(roots, np.zeros_like(roots), ls="none", marker="o" if index == 0 else "s",
                  ms=5, mfc=BACKGROUND, mec=color)
        axis.annotate(f"distance {index + 1}", (u[-1], h[-1]), (7, 0),
                      textcoords="offset points", color=color, va="center", fontsize=12)
    axis.annotate("shared DC zero", (0, 0), (13, 19), textcoords="offset points",
                  fontsize=11, arrowprops={"arrowstyle": "-", "color": "#888888"})
    axis.set_xlim(-.015, 2.05)
    axis.set_ylim(-2.2, 2.2)
    axis.set_xticks([0, .5, 1, 1.5])
    axis.set_yticks([-2, 0, 2])
    axis.spines["bottom"].set_bounds(0, u[-1])
    axis.spines["left"].set_bounds(-2, 2)
    axis.set_xlabel(r"$u = q\sqrt{\lambda z_1}$")
    axis.set_ylabel(r"$h_j = 2\sin\chi_j$")
    save(fig, "01-ctf-null-space.svg", distance_indices=[1, 4],
         actual_distance_ratio=float(numbers[0] / numbers[3]))


def ict_comparison(observed, number, material):
    gamma = material["gamma"]
    filtered, ict, contact = homogeneous_ctf_ict(observed, number, gamma=gamma, alpha=ALPHA)
    maps = [filtered, ict]
    residuals = []
    for phase in maps:
        predicted = predict_intensity(phase, number, gamma=gamma)
        residuals.append(float(np.linalg.norm(predicted - observed) / np.linalg.norm(observed - 1)))
    difference = ict - filtered
    phase_range = limits(maps)
    diff_bound = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference**2)))
    # Algebraic recovery of the filtered contact intensity is exact, before propagation.
    if not np.allclose(np.exp(2 * ict / gamma), contact, atol=1e-12, rtol=1e-12):
        raise ValueError("ICT logarithm does not recover the filtered contact intensity")
    for zoom in (False, True):
        top, left, size = ROI
        crop = np.s_[top:top+size, left:left+size] if zoom else np.s_[:, :]
        fig = plt.figure(figsize=(11.5, 6), layout="constrained")
        grid = fig.add_gridspec(2, 3, height_ratios=[1, .045])
        for index, (phase, title) in enumerate(zip(maps + [difference], [
                f"Homogeneous CTF\nResidual {residuals[0]:.4f}",
                f"ICT\nResidual {residuals[1]:.4f}",
                f"ICT − CTF\nFull-field RMS {rms:.4f} rad"], strict=True)):
            axis = fig.add_subplot(grid[0, index])
            picture = axis.imshow(phase[crop], interpolation="nearest",
                cmap="magma" if index < 2 else "RdBu_r",
                vmin=phase_range[0] if index < 2 else -diff_bound,
                vmax=phase_range[1] if index < 2 else diff_bound)
            axis.set_title(title, fontsize=12)
            axis.set_axis_off()
            scale_bar(axis, size if zoom else phase.shape[0], 10 if zoom else 50)
            if not zoom:
                axis.add_patch(Rectangle((left-.5, top-.5), size, size,
                                         fill=False, edgecolor="#E69F00", linewidth=1))
            if index == 1:
                fig.colorbar(picture, cax=fig.add_subplot(grid[1, :2]), orientation="horizontal",
                             label="phase [rad] · shared CTF / ICT scale")
            elif index == 2:
                fig.colorbar(picture, cax=fig.add_subplot(grid[1, 2]), orientation="horizontal",
                             label="difference [rad] · separate scale")
        fig.suptitle(f"CTF and ICT nearly coincide · {'same sphere ROI' if zoom else 'full field'}",
                     x=.025, ha="left", fontsize=21)
        fig.supxlabel(f"Same first hologram · γ = {gamma:.1f} · matched α = 0.01"
                       "\nResiduals use the same nonlinear, absorbing-object model", fontsize=12)
        save(fig, f"01-ctf-ict-{'roi' if zoom else 'full'}.png", phase_limits_rad=phase_range,
             difference_limits_rad=[-diff_bound, diff_bound], roi=ROI if zoom else None)
    metadata = {"material": material, "alpha_phase_scale": ALPHA,
        "regularization": "same ||s||^2 penalty, s = gamma/2 * (I0 - 1); reference intensity = 1",
        "equivalent_alpha_intensity_scale": ALPHA * gamma**2 / 4,
        "constraint": "none", "iterations": 0, "converged": True,
        "contact_intensity_range": [float(contact.min()), float(contact.max())],
        "ctf_residual": residuals[0], "ict_residual": residuals[1],
        "phase_difference_rms_rad": rms, "phase_difference_max_abs_rad": diff_bound,
        "residual_model": "|B exp(phi/gamma + i phi)|^2, first distance, stored data unchanged",
        "note": "A physical tabulated material ratio is introduced for both new methods."
                " Earlier notebook CTF and subsequent nonlinear runs retain their pure-phase model."}
    np.savez_compressed(CACHE / "ctf-ict-single.npz", ctf=filtered, ict=ict,
                        metadata=json.dumps(metadata))
    return metadata


def main():
    global CACHE, CONVERGED_CACHE, DATASET, OUTPUT, ROI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE.parent / "direct")
    parser.add_argument("--converged-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi", nargs=3, type=int, default=ROI, metavar=("TOP", "LEFT", "SIZE"))
    args = parser.parse_args()
    CACHE, CONVERGED_CACHE = args.cache.resolve(), args.converged_cache.resolve()
    DATASET, OUTPUT, ROI = args.dataset.resolve(), args.output.resolve(), tuple(args.roi)
    ASSETS.clear()
    configure()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    data_hash = sha256(DATASET)
    with np.load(DATASET) as archive:
        observed = archive["holograms"][0].astype(np.float64)
        numbers = np.asarray(archive["fresnelNumbers"], dtype=float).ravel()
    height, width = observed.shape
    top, left, size = ROI
    if min(top, left) < 0 or size < 1 or top + size > height or left + size > width:
        raise ValueError("ROI lies outside the measured image")
    chi = fresnel_phase(observed.shape, numbers[0])
    h_tie = 2 * chi
    contrast = observed - 1
    unregularized_phase = tie_reconstruct(observed, numbers[0], alpha=0)
    phase = tie_reconstruct(observed, numbers[0], alpha=ALPHA)
    predicted = predict_intensity(phase, numbers[0])
    residual = float(np.linalg.norm(predicted - observed) / np.linalg.norm(contrast))

    # Check the inverse closes its stated linear model, with only DC projected out.
    recovered_contrast = ifft2(h_tie * fft2(unregularized_phase, norm="ortho"), norm="ortho").real
    closure_error = float(np.linalg.norm(recovered_contrast - contrast + contrast.mean())
                          / np.linalg.norm(contrast))
    if closure_error > 1e-9 or abs(phase.mean()) > 1e-10 or not np.isfinite(phase).all():
        raise ValueError("TIE inverse failed its linear-model or phase-gauge check")

    # Independently verify sign, grid conventions and propagation using DeepInv.
    torch.set_num_threads(4)
    prop = make_fresnel_propagators((height, width), [float(numbers[0])],
        wavelength=WAVELENGTH, pixel_size=PIXEL, dtype=torch.cdouble)[0]
    with torch.no_grad():
        wave = torch.from_numpy(np.exp(1j * phase))[None]
        deepinv_prediction = prop.A(wave).abs().square().squeeze().numpy()
    propagation_error = float(np.linalg.norm(deepinv_prediction - predicted)
                              / np.linalg.norm(predicted))
    if propagation_error > 1e-9:
        raise ValueError(f"Fresnel prediction differs from DeepInv: {propagation_error}")

    arrays, residuals = [], []
    for name in ("ctf_single", "ctf_single_tikh"):
        path = CONVERGED_CACHE / f"{name}.npz"
        with np.load(path) as archive:
            metadata = json.loads(str(archive["metadata"]))
            validate_checkpoint(path, metadata, DATASET)
            if not metadata["converged"]:
                raise ValueError(f"Unconverged CTF checkpoint: {name}")
            arrays.append(archive["phase"].copy())
            residuals.append(metadata["residual"])
    for zoom in (False, True):
        tie_figure(phase, observed, predicted, residual, zoom=zoom)
        ctf_figure(arrays, residuals, zoom=zoom)
    null_plot(numbers)
    ict_metadata = ict_comparison(observed, numbers[0], polystyrene_8kev())

    distances = PIXEL**2 / (WAVELENGTH * numbers)
    periods = np.asarray([1, 5, 15, 30], dtype=float)
    period_chi = np.pi * PIXEL**2 / (numbers[:, None] * (periods[None, :] * 1e-6)**2)
    metadata = {
        "source_data_sha256": data_hash, "renderer_sha256": sha256(Path(__file__)),
        "direct_methods_sha256": sha256(PROJECT / "src/multi_distance_phase_retrival_huhn/direct_methods.py"),
        "shape": list(observed.shape), "pixel_size_m": PIXEL, "wavelength_m": WAVELENGTH,
        "fresnel_numbers": numbers.tolist(), "effective_distances_m": distances.tolist(),
        "sampled_axis_nyquist_period_m": 2 * PIXEL,
        "sampled_corner_angle_rad": float(WAVELENGTH / (np.sqrt(2) * PIXEL)),
        "first_position_field_ray_angle_rad": float(np.hypot(height, width) * PIXEL / (2 * .156)),
        "axis_nyquist_chi": (np.pi / (4 * numbers)).tolist(),
        "periods_um": periods.tolist(), "chi_by_distance_and_period": period_chi.tolist(),
        "roi": {"top": ROI[0], "left": ROI[1], "height": ROI[2], "width": ROI[2]},
        "method": "single-distance pure-phase TIE, alpha=0.01, periodic FFT, DC zero",
        "phase_gauge": "zero mean; no nonpositive constraint",
        "converged": True, "iterations": 0, "stop": "closed-form inverse",
        "measurement_preprocessing": "stored values unchanged; no per-image normalization or crop",
        "nonlinear_intensity_residual": residual,
        "residual_definition": "|| |B_z exp(i phi_TIE)|^2 - y_1 ||_2 / ||y_1 - 1||_2",
        "linear_model_closure_error_excluding_dc": closure_error,
        "deepinv_propagation_relative_error": propagation_error,
        "phase_ground_truth": None, "ict_comparison": ict_metadata, "figures": ASSETS,
    }
    np.savez_compressed(CACHE / "tie-single.npz", phase=phase,
                        unregularized_phase=unregularized_phase, metadata=json.dumps(metadata))
    (OUTPUT / "tie-introduction-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figures": len(ASSETS), "tie_residual": residual,
        "linear_closure_error": closure_error, "deepinv_propagation_error": propagation_error,
        "ctf_residuals": residuals, "ict_comparison": ict_metadata}, indent=2))


if __name__ == "__main__":
    main()
