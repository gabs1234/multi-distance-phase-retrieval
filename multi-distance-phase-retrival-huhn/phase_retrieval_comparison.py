import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    from time import perf_counter
    import math
    import os

    import deepinv as dinv
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.patches import Rectangle

    from multi_distance_phase_retrival_huhn.hologram_plots import plot_hologram_line_cuts
    from multi_distance_phase_retrival_huhn.phase_retrieval import (
        alternating_projections,
        angular_frequency_radius,
        constrained_ctf_reconstruct,
        ctf_reconstruct,
        ctf_transfer_functions,
        fresnel_numbers_to_distances,
        huhn_nltikh,
        huhn_regularization_filter,
        make_fresnel_propagators,
        projected_gradient_descent,
        relative_data_residual,
        simulate_intensities,
        support_referenced_phase_nrmse,
    )

    COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    WAVELENGTH = 1.5498e-10
    PIXEL_SIZE = 196e-9
    DEFAULT_FRESNEL_NUMBERS = (1.59e-3, 1.57e-3, 1.49e-3, 1.33e-3)
    DEVICE = dinv.utils.get_device()
    NOTEBOOK_AUTORUN = os.environ.get("PHASE_RETRIEVAL_AUTORUN") == "1"
    NOTEBOOK_SOURCE = os.environ.get("PHASE_RETRIEVAL_SOURCE", "synthetic")
    NOTEBOOK_FULL_FIELD = os.environ.get("PHASE_RETRIEVAL_FULL_FIELD") == "1"
    NOTEBOOK_FULL_FIELD_ITERATIONS = int(
        os.environ.get("PHASE_RETRIEVAL_FULL_FIELD_ITERATIONS", "5")
    )
    NOTEBOOK_FULL_FIELD_CTF_ITERATIONS = int(
        os.environ.get("PHASE_RETRIEVAL_FULL_FIELD_CTF_ITERATIONS", "10")
    )

    plt.rcParams.update(
        {
            "figure.facecolor": "#fffff8",
            "axes.facecolor": "#fffff8",
            "savefig.facecolor": "#fffff8",
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titlelocation": "left",
        }
    )
    return (
        COLORS,
        DEFAULT_FRESNEL_NUMBERS,
        DEVICE,
        NOTEBOOK_AUTORUN,
        NOTEBOOK_FULL_FIELD,
        NOTEBOOK_FULL_FIELD_CTF_ITERATIONS,
        NOTEBOOK_FULL_FIELD_ITERATIONS,
        NOTEBOOK_SOURCE,
        PIXEL_SIZE,
        Path,
        perf_counter,
        WAVELENGTH,
        alternating_projections,
        angular_frequency_radius,
        constrained_ctf_reconstruct,
        ctf_reconstruct,
        ctf_transfer_functions,
        dinv,
        fresnel_numbers_to_distances,
        huhn_nltikh,
        huhn_regularization_filter,
        make_fresnel_propagators,
        mo,
        np,
        plot_hologram_line_cuts,
        plt,
        projected_gradient_descent,
        relative_data_residual,
        simulate_intensities,
        support_referenced_phase_nrmse,
        torch,
    )


@app.cell(hide_code=True)
def _(DEVICE, dinv, mo):
    mo.md(rf"""
    # Controlled comparison of multi-distance Fresnel phase retrieval

    This notebook changes **one scientific axis at a time**: linearization,
    constraints, measurement diversity, optimizer, then the ingredients of
    Huhn's nonlinear Tikhonov method. The nonlinear physics and fixed-step
    PGD paths use the editable **DeepInv {dinv.__version__}** installation on
    **{DEVICE}**.

    The measured bead stack is useful for data consistency, but it has no
    phase ground truth. Synthetic data are therefore the default whenever an
    accuracy claim is made.

    | Direct contrast | Held fixed | Changed axis |
    | --- | --- | --- |
    | CTF α = 0 → scalar α | model, data count | regularization |
    | CTF one → four planes | model, α | measurement diversity |
    | CTF → nonlinear, unconstrained | one plane, data, no prior | forward approximation |
    | nonlinear free → feasible | model, plane, solver, start | constraints |
    | nonlinear one → four planes | solver, start, constraints, mean scaling | measurement diversity |
    | multi-distance PGD → AP | data, physics, start, constraints, iterations | optimizer |
    | Huhn ablation edges | all preceding ingredients | initialization, then filter, then adaptive policy |
    """)
    return


@app.cell
def _(
    COLORS,
    ctf_transfer_functions,
    np,
    plt,
    relative_data_residual,
    support_referenced_phase_nrmse,
    torch,
):
    def centered_crop(images, size):
        height, width = images.shape[-2:]
        if size > min(height, width):
            raise ValueError(f"crop {size} exceeds image shape {(height, width)}")
        top = (height - size) // 2
        left = (width - size) // 2
        return images[..., top : top + size, left : left + size]

    def make_bead_phantom(size, max_phase, seed):
        """Projected-sphere phantom inside a deliberately loose known support."""

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        axis = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        support = xx.square() + yy.square() <= 0.78**2
        thickness = torch.zeros_like(xx)
        for _ in range(11):
            angle = 2.0 * torch.pi * torch.rand((), generator=generator)
            center_radius = 0.50 * torch.sqrt(torch.rand((), generator=generator))
            cx = center_radius * torch.cos(angle)
            cy = center_radius * torch.sin(angle)
            radius = 0.055 + 0.075 * torch.rand((), generator=generator)
            radial_square = ((xx - cx) / radius).square() + ((yy - cy) / radius).square()
            sphere = torch.sqrt((1.0 - radial_square).clamp_min(0.0))
            thickness = torch.maximum(thickness, sphere)
        phase = -float(max_phase) * thickness / thickness.amax().clamp_min(1e-12)
        phase = torch.where(support, phase, torch.zeros_like(phase))
        return phase[None, None], support

    def tensor_image(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().squeeze().cpu().numpy()
        return np.asarray(value)

    def image_grid(
        images,
        labels,
        *,
        colorbar_label,
        cmap="magma",
        zero_ceiling=False,
        context_image=None,
        crop_bounds=None,
    ):
        arrays = [tensor_image(image) for image in images]
        finite = np.concatenate([array[np.isfinite(array)].ravel() for array in arrays])
        vmin, vmax = np.quantile(finite, [0.01, 0.99])
        if zero_ceiling and vmax <= max(1e-8, 0.03 * abs(vmin)):
            vmax = 0.0
        if np.isclose(vmin, vmax):
            vmin, vmax = float(vmin) - 1.0, float(vmax) + 1.0
        _context = None if context_image is None else tensor_image(context_image)
        if _context is not None:
            if _context.ndim != 2 or not np.isfinite(_context).all():
                raise ValueError("context_image must be a finite 2D image")
            if crop_bounds is None or len(crop_bounds) != 4:
                raise ValueError("crop_bounds=(top, left, height, width) is required")
            _top, _left, _height, _width = map(int, crop_bounds)
            if (
                _top < 0
                or _left < 0
                or _height != arrays[0].shape[-2]
                or _width != arrays[0].shape[-1]
                or _top + _height > _context.shape[0]
                or _left + _width > _context.shape[1]
            ):
                raise ValueError(
                    "crop_bounds do not locate the reconstruction in context_image"
                )
            _context_vmin, _context_vmax = np.quantile(_context, [0.01, 0.99])
            if np.isclose(_context_vmin, _context_vmax):
                _context_vmin, _context_vmax = (
                    float(_context_vmin) - 0.01,
                    float(_context_vmax) + 0.01,
                )
        figure, axes = plt.subplots(
            1, len(arrays), figsize=(3.15 * len(arrays), 3.0), squeeze=False
        )
        last_image = None
        for axis, array, label in zip(axes[0], arrays, labels, strict=True):
            last_image = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(label, fontsize=10)
            axis.set_axis_off()
            if _context is not None:
                _inset = axis.inset_axes([0.02, 0.02, 0.29, 0.29])
                _inset.imshow(
                    _context,
                    cmap="gray",
                    vmin=_context_vmin,
                    vmax=_context_vmax,
                    origin="upper",
                    interpolation="nearest",
                )
                _inset.add_patch(
                    Rectangle(
                        (_left - 0.5, _top - 0.5),
                        _width,
                        _height,
                        fill=False,
                        edgecolor="#D55E00",
                        linewidth=1.6,
                    )
                )
                _inset.set_title("full FOV", fontsize=7, color="#D55E00", pad=1)
                _inset.set_xticks([])
                _inset.set_yticks([])
                for _spine in _inset.spines.values():
                    _spine.set_visible(True)
                    _spine.set_color("#D55E00")
                    _spine.set_linewidth(1.0)
        figure.colorbar(
            last_image, ax=list(axes[0]), shrink=0.76, pad=0.018, label=colorbar_label
        )
        figure.subplots_adjust(left=0.01, right=0.96, bottom=0.02, top=0.88, wspace=0.05)
        return figure

    def curve_figure(curves, *, xlabel, ylabel, log_y=True):
        figure, axis = plt.subplots(figsize=(6.4, 3.35))
        endpoints = []
        for index, (label, pair) in enumerate(curves.items()):
            x_values = np.asarray(pair[0], dtype=float)
            y_values = np.asarray(pair[1], dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if log_y:
                valid &= y_values > 0
            x_values, y_values = x_values[valid], y_values[valid]
            if not len(x_values):
                continue
            color = COLORS[index % len(COLORS)]
            axis.plot(x_values, y_values, color=color, linewidth=1.8)
            endpoints.append((label, x_values[-1], y_values[-1], color))
        if log_y:
            axis.set_yscale("log")
        for index, (label, x_value, y_value, color) in enumerate(endpoints):
            offset = 7 * (index - (len(endpoints) - 1) / 2)
            axis.annotate(
                label,
                (x_value, y_value),
                xytext=(6, offset),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.margins(x=0.17, y=0.12)
        figure.tight_layout()
        return figure

    def full_field_reconstruction_figure(phase, detail_size):
        array = tensor_image(phase)
        height, width = array.shape
        detail_height = min(int(detail_size), height)
        detail_width = min(int(detail_size), width)
        top = (height - detail_height) // 2
        left = (width - detail_width) // 2
        finite = array[np.isfinite(array)]
        vmin, vmax = np.quantile(finite, [0.01, 0.99])
        if vmax <= max(1e-8, 0.03 * abs(vmin)):
            vmax = 0.0
        if np.isclose(vmin, vmax):
            vmin, vmax = float(vmin) - 1.0, float(vmax) + 1.0
        figure, axis = plt.subplots(figsize=(7.2, 6.4))
        phase_image = axis.imshow(array, cmap="magma", vmin=vmin, vmax=vmax)
        axis.add_patch(
            Rectangle(
                (left - 0.5, top - 0.5),
                detail_width,
                detail_height,
                fill=False,
                edgecolor="#D55E00",
                linewidth=1.2,
            )
        )
        axis.set_title(
            f"Native {height} × {width} phase · center detail inset", fontsize=12
        )
        axis.set_axis_off()
        inset = axis.inset_axes([0.62, 0.03, 0.36, 0.36])
        inset.imshow(
            array[top : top + detail_height, left : left + detail_width],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        inset.set_title(
            f"center {detail_height} × {detail_width}",
            fontsize=8,
            color="#D55E00",
            pad=2,
        )
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#D55E00")
            spine.set_linewidth(1.0)
        figure.colorbar(phase_image, ax=axis, shrink=0.78, pad=0.018, label="phase [rad]")
        figure.subplots_adjust(left=0.02, right=0.93, bottom=0.03, top=0.91)
        return figure

    def radial_average(radius, values, bins=72):
        radial = tensor_image(radius).ravel()
        data = tensor_image(values).ravel()
        edges = np.linspace(0.0, radial.max(), bins + 1)
        indices = np.clip(np.digitize(radial, edges) - 1, 0, bins - 1)
        sums = np.bincount(indices, weights=data, minlength=bins)
        counts = np.bincount(indices, minlength=bins)
        means = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)
        return 0.5 * (edges[:-1] + edges[1:]), means

    def ctf_linear_residual(estimate, measurements, propagators):
        transfer = ctf_transfer_functions(propagators, dtype=estimate.dtype).to(
            estimate.device
        )
        contrast = measurements - 1.0
        predicted = torch.fft.ifft2(
            transfer[:, None, None]
            * torch.fft.fft2(estimate, norm="ortho"),
            norm="ortho",
        ).real
        denominator = torch.linalg.vector_norm(contrast).clamp_min(1e-12)
        return float(torch.linalg.vector_norm(predicted - contrast) / denominator)

    def phase_nrmse(estimate, truth, support):
        if truth is None or support is None:
            return None
        return support_referenced_phase_nrmse(estimate, truth, support)

    def result_row(label, result, truth, support, measurements, propagators):
        error = phase_nrmse(result.estimate, truth, support)
        return {
            "method": label,
            "residual": relative_data_residual(
                result.estimate, measurements, propagators
            ),
            "nrmse": error,
            "iterations": result.iterations,
            "calls": result.operator_calls[-1],
            "seconds": result.elapsed_seconds,
            "stop": result.stop_reason,
        }

    def markdown_table(rows, columns):
        heading = "| " + " | ".join(label for label, _key, _formatter in columns) + " |"
        rule = "| " + " | ".join("---" for _ in columns) + " |"
        lines = [heading, rule]
        for row in rows:
            cells = []
            for _label, key, formatter in columns:
                value = row.get(key)
                if value is None:
                    cell = "—"
                else:
                    cell = formatter(value)
                cells.append(str(cell).replace("|", "\\|"))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    METRIC_COLUMNS = (
        ("Method", "method", str),
        ("relative intensity residual", "residual", lambda value: f"{value:.3e}"),
        ("phase NRMSE", "nrmse", lambda value: f"{value:.3f}"),
        ("iterations", "iterations", str),
        ("equiv. propagations", "calls", str),
        ("time [s]", "seconds", lambda value: f"{value:.2f}"),
        ("stop", "stop", str),
    )
    return (
        METRIC_COLUMNS,
        centered_crop,
        ctf_linear_residual,
        curve_figure,
        full_field_reconstruction_figure,
        image_grid,
        make_bead_phantom,
        markdown_table,
        phase_nrmse,
        radial_average,
        result_row,
    )


@app.cell(hide_code=True)
def _(
    NOTEBOOK_FULL_FIELD_CTF_ITERATIONS,
    NOTEBOOK_FULL_FIELD_ITERATIONS,
    NOTEBOOK_SOURCE,
    mo,
):
    _source_default = (
        "Measured bead data"
        if NOTEBOOK_SOURCE == "measured"
        else "Synthetic validation (has truth)"
    )
    source_control = mo.ui.dropdown(
        options={
            "Synthetic validation (has truth)": "synthetic",
            "Measured bead data": "measured",
        },
        value=_source_default,
        label="data source",
    )
    size_control = mo.ui.dropdown(
        options={"96 × 96 (fast)": 96, "128 × 128": 128, "192 × 192": 192},
        value="128 × 128",
        label="comparison crop size",
    )
    phase_control = mo.ui.slider(
        start=0.05, stop=1.0, step=0.05, value=0.45, label="synthetic max |phase| [rad]"
    )
    noise_control = mo.ui.slider(
        start=0.0, stop=0.05, step=0.0025, value=0.005, label="synthetic noise σ"
    )
    iteration_control = mo.ui.slider(
        start=5, stop=80, step=5, value=25, label="iterations per iterative method"
    )
    step_control = mo.ui.slider(
        start=0.025, stop=0.35, step=0.025, value=0.20, label="DeepInv PGD step (mean loss)"
    )
    ctf_alpha_control = mo.ui.dropdown(
        options={"10⁻⁴": 1e-4, "10⁻³": 1e-3, "10⁻²": 1e-2, "10⁻¹": 1e-1},
        value="10⁻²",
        label="scalar CTF α",
    )
    seed_control = mo.ui.number(start=0, stop=9999, value=7, label="synthetic seed")
    run_control = mo.ui.run_button(label="Run all five stages")
    full_field_iteration_control = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=max(1, min(50, NOTEBOOK_FULL_FIELD_ITERATIONS)),
        label="full-field NLTikh iterations",
    )
    full_field_ctf_control = mo.ui.slider(
        start=1,
        stop=100,
        step=1,
        value=max(1, min(100, NOTEBOOK_FULL_FIELD_CTF_ITERATIONS)),
        label="full-field CTF-init iterations",
    )
    full_field_run_control = mo.ui.run_button(
        label="Run optional native-field NLTikh"
    )

    mo.vstack(
        [
            mo.md("## Experiment controls"),
            mo.hstack([source_control, size_control, iteration_control], widths="equal"),
            mo.hstack([phase_control, noise_control, step_control], widths="equal"),
            mo.hstack([ctf_alpha_control, seed_control, run_control], widths="equal"),
            mo.callout(
                "The button gates the expensive FFT iterations. Change controls, then run again.",
                kind="info",
            ),
            mo.md("### Optional measured-data full-field reconstruction"),
            mo.hstack(
                [
                    full_field_iteration_control,
                    full_field_ctf_control,
                    full_field_run_control,
                ],
                widths="equal",
            ),
            mo.callout(
                "This runs only the final Huhn-style NLTikh method on the native "
                "2048 × 1920 field. It has 240× as many pixels as the default "
                "128 × 128 crop and can require substantially more memory and time. "
                "The low defaults are a feasibility check; increase them until the "
                "reported residual and phase are stable for scientific use.",
                kind="warn",
            ),
        ]
    )
    return (
        ctf_alpha_control,
        full_field_ctf_control,
        full_field_iteration_control,
        full_field_run_control,
        iteration_control,
        noise_control,
        phase_control,
        run_control,
        seed_control,
        size_control,
        source_control,
        step_control,
    )


@app.cell
def _(
    NOTEBOOK_AUTORUN,
    ctf_alpha_control,
    iteration_control,
    mo,
    noise_control,
    phase_control,
    run_control,
    seed_control,
    size_control,
    source_control,
    step_control,
):
    mo.stop(
        not (run_control.value or NOTEBOOK_AUTORUN),
        mo.callout("Choose settings and press **Run all five stages**.", kind="warn"),
    )
    experiment_settings = {
        "source": source_control.value,
        "size": int(size_control.value),
        "max_phase": float(phase_control.value),
        "noise": float(noise_control.value),
        "iterations": int(iteration_control.value),
        "step": float(step_control.value),
        "ctf_alpha": float(ctf_alpha_control.value),
        "seed": int(seed_control.value),
    }
    return (experiment_settings,)


@app.cell
def _(
    DEFAULT_FRESNEL_NUMBERS,
    DEVICE,
    PIXEL_SIZE,
    Path,
    WAVELENGTH,
    centered_crop,
    experiment_settings,
    make_bead_phantom,
    make_fresnel_propagators,
    np,
    simulate_intensities,
    torch,
):
    _size = experiment_settings["size"]
    if experiment_settings["source"] == "synthetic":
        full_measurements = None
        crop_bounds = None
        phase_truth, support_mask = make_bead_phantom(
            _size, experiment_settings["max_phase"], experiment_settings["seed"]
        )
        phase_truth = phase_truth.to(DEVICE)
        support_mask = support_mask.to(DEVICE)
        fresnel_numbers = DEFAULT_FRESNEL_NUMBERS
        propagators = make_fresnel_propagators(
            (_size, _size),
            fresnel_numbers,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            device=DEVICE,
        )
        with torch.no_grad():
            clean_measurements = simulate_intensities(phase_truth, propagators)
        _generator = torch.Generator(device="cpu").manual_seed(
            experiment_settings["seed"] + 1000
        )
        _noise = torch.randn(
            clean_measurements.shape, generator=_generator, dtype=clean_measurements.dtype
        ).to(DEVICE)
        measurements = (
            clean_measurements + experiment_settings["noise"] * _noise
        ).clamp_min(0.0)
        source_note = (
            "Synthetic projected spheres: phase truth and a deliberately loose circular "
            "support are known. Additive Gaussian noise is applied to normalized intensity."
        )
    else:
        _data_path = Path(__file__).with_name("holograms_beads_updated.npz")
        with np.load(_data_path) as _archive:
            _full_stack = torch.from_numpy(_archive["holograms"]).float()
            fresnel_numbers = tuple(float(value) for value in _archive["fresnelNumbers"])
        _full_height, _full_width = _full_stack.shape[-2:]
        crop_bounds = (
            (_full_height - _size) // 2,
            (_full_width - _size) // 2,
            _size,
            _size,
        )
        full_measurements = _full_stack
        _crop = centered_crop(_full_stack, _size)
        measurements = _crop[:, None, None].to(DEVICE)
        clean_measurements = None
        phase_truth = None
        support_mask = None
        propagators = make_fresnel_propagators(
            (_size, _size),
            fresnel_numbers,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            device=DEVICE,
        )
        source_note = (
            "Measured holograms from the local holograms_beads_updated.npz, center-cropped "
            "with their stored intensity values preserved. The accompanying demo documents "
            "dark/flat-field correction, registration, and common magnification; the archive "
            "does not contain the raw calibration fields. No additional correction or "
            "per-crop mean normalization is applied. The sample fills the field of view, "
            "so no support mask is asserted. Every reconstruction panel below includes "
            "a first-plane full-FOV inset with this crop outlined; phase outside the "
            "outline has not been reconstructed. "
            "Cropping plus FFT-periodic boundaries can create edge-model mismatch."
        )

    measurements_single = measurements[:1]
    propagators_single = propagators[:1]
    fresnel_numbers_single = fresnel_numbers[:1]
    initial_phase = torch.zeros(
        1, 1, _size, _size, dtype=measurements.dtype, device=DEVICE
    )
    constraint_text = "nonpositive phase + loose known support" if support_mask is not None else "nonpositive phase only"
    return (
        constraint_text,
        crop_bounds,
        fresnel_numbers,
        full_measurements,
        initial_phase,
        measurements,
        measurements_single,
        phase_truth,
        propagators,
        propagators_single,
        source_note,
        support_mask,
    )


@app.cell(hide_code=True)
def _(mo, source_note):
    mo.vstack(
        [
            mo.md("## Data entering every stage · fringes across distances"),
            mo.callout(source_note, kind="info"),
            mo.md(
                "Choose the same row or column through every hologram. These display "
                "controls update the profiles without rerunning the reconstruction solvers."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(measurements, mo):
    cut_orientation = mo.ui.dropdown(
        options=["horizontal", "vertical"], value="horizontal", label="cut direction"
    )
    cut_band = mo.ui.dropdown(
        options={"1 pixel": 1, "3 pixels": 3, "5 pixels": 5, "9 pixels": 9},
        value="1 pixel", label="averaging band",
    )
    cut_position = mo.ui.slider(
        start=0, stop=min(measurements.shape[-2:]) - 1,
        value=measurements.shape[-2] // 2, step=1, show_value=True,
        label="row / column index (zero-based)",
    )
    mo.hstack([cut_orientation, cut_position, cut_band], wrap=True)
    return cut_band, cut_orientation, cut_position


@app.cell(hide_code=True)
def _(
    PIXEL_SIZE,
    WAVELENGTH,
    cut_band,
    cut_orientation,
    cut_position,
    crop_bounds,
    fresnel_numbers,
    fresnel_numbers_to_distances,
    measurements,
    full_measurements,
    mo,
    plot_hologram_line_cuts,
):
    _figure = plot_hologram_line_cuts(
        measurements[:, 0, 0].detach().cpu().numpy(), fresnel_numbers,
        wavelength=WAVELENGTH, pixel_size=PIXEL_SIZE,
        orientation=cut_orientation.value, index=int(cut_position.value),
        band_width=int(cut_band.value),
        context_images=(
            None if full_measurements is None else full_measurements.numpy()
        ),
        crop_bounds=crop_bounds,
    )
    _distances = fresnel_numbers_to_distances(fresnel_numbers, WAVELENGTH, PIXEL_SIZE)
    _span = max(_distances) / min(_distances) - 1.0
    mo.vstack(
        [
            _figure,
            mo.md(rf"""
            For measured data, each left panel is the complete 2048 × 1920 registered
            hologram. The orange rectangle marks the reconstruction crop, which is enlarged
            in the inset; the blue marker locates the line cut within that crop. For
            synthetic data, the left panel is the complete simulated field. A shaded band
            appears when averaging is enabled.
            Profiles use the stored/simulated intensities without offsets, rescaling, or
            smoothing along the cut. Full-field thumbnails and crop insets each share their
            own pooled 1–99% display window; **the profiles are not clipped**. Dotted gray repeats the smallest-z
            profile in each later panel to expose fringe shifts and contrast changes.

            Distances are the **equivalent plane-wave** values used by DeepInv,
            $z_\mathrm{{eff}} = p^2 / (\lambda F)$, with $p = {PIXEL_SIZE * 1e9:.0f}$ nm.
            They span {min(_distances) * 1e3:.2f}–{max(_distances) * 1e3:.2f} mm
            ({_span:.1%} increase). These closely spaced planes can show subtle changes;
            fringe amplitudes need not increase monotonically on every cut.
            """),
        ]
    )
    return


@app.cell
def _(crop_bounds, full_measurements, image_grid):
    _context = None if full_measurements is None else full_measurements[0]

    def reconstruction_grid(
        images, labels, *, colorbar_label, cmap="magma", zero_ceiling=False
    ):
        return image_grid(
            images,
            labels,
            colorbar_label=colorbar_label,
            cmap=cmap,
            zero_ceiling=zero_ceiling,
            context_image=_context,
            crop_bounds=crop_bounds,
        )

    return (reconstruction_grid,)


@app.cell(hide_code=True)
def _(
    METRIC_COLUMNS,
    angular_frequency_radius,
    ctf_linear_residual,
    ctf_reconstruct,
    ctf_transfer_functions,
    curve_figure,
    experiment_settings,
    initial_phase,
    markdown_table,
    measurements,
    measurements_single,
    mo,
    phase_nrmse,
    phase_truth,
    propagators,
    propagators_single,
    radial_average,
    reconstruction_grid,
    support_mask,
):
    _alpha = experiment_settings["ctf_alpha"]
    stage1_results = {
        "single, unregularized": ctf_reconstruct(
            measurements_single, propagators_single, alpha=0.0
        ),
        "single, scalar Tikhonov": ctf_reconstruct(
            measurements_single, propagators_single, alpha=_alpha
        ),
        "multi, unregularized": ctf_reconstruct(
            measurements, propagators, alpha=0.0
        ),
        "multi, scalar Tikhonov": ctf_reconstruct(
            measurements, propagators, alpha=_alpha
        ),
    }

    _rows = []
    for _label, _estimate in stage1_results.items():
        _is_single = _label.startswith("single")
        _used_measurements = measurements_single if _is_single else measurements
        _used_propagators = propagators_single if _is_single else propagators
        _rows.append(
            {
                "method": _label,
                "residual": ctf_linear_residual(
                    _estimate, _used_measurements, _used_propagators
                ),
                "nrmse": phase_nrmse(_estimate, phase_truth, support_mask),
                "iterations": "closed form",
                "calls": 0,
                "seconds": None,
                "stop": "linear inverse",
            }
        )

    _radius = angular_frequency_radius(
        tuple(initial_phase.shape[-2:]),
        device=initial_phase.device,
        dtype=initial_phase.dtype,
    )
    _conditioning = {}
    for _label, _props in (
        ("single distance", propagators_single),
        ("four distances", propagators),
    ):
        _transfer = ctf_transfer_functions(_props, dtype=initial_phase.dtype)
        _denominator = _transfer.square().sum(dim=0)
        _conditioning[_label] = radial_average(_radius, _denominator)

    _phase_figure = reconstruction_grid(
        list(stage1_results.values()),
        list(stage1_results.keys()),
        colorbar_label="phase [rad]",
        zero_ceiling=False,
    )
    _conditioning_figure = curve_figure(
        _conditioning,
        xlabel="radial angular frequency [rad px⁻¹]",
        ylabel="radial mean of Σ hⱼ²",
        log_y=True,
    )
    _table_columns = tuple(
        ("linear CTF residual" if key == "residual" else label, key, formatter)
        for label, key, formatter in METRIC_COLUMNS
        if key != "seconds"
    )
    mo.vstack(
        [
            mo.md(
                rf"""
                ## 1 · Linear CTF baseline

                This is a 2 × 2 factorial check. Along each row or column only one
                axis changes: **one vs four distances**, or **α = 0 vs scalar
                α = {_alpha:.0e}**. The unregularized inverse uses a numerical
                Moore–Penrose cutoff, including the unobservable DC phase mode.
                """
            ),
            _phase_figure,
            mo.md(
                "The conditioning curve shows why additional distances help: their "
                "CTF zeros generally do not coincide."
            ),
            _conditioning_figure,
            mo.md(markdown_table(_rows, _table_columns)),
        ]
    )
    return (stage1_results,)


@app.cell(hide_code=True)
def _(
    METRIC_COLUMNS,
    constraint_text,
    curve_figure,
    experiment_settings,
    initial_phase,
    markdown_table,
    measurements_single,
    mo,
    phase_nrmse,
    phase_truth,
    projected_gradient_descent,
    propagators_single,
    relative_data_residual,
    reconstruction_grid,
    result_row,
    stage1_results,
    support_mask,
):
    stage2_free = projected_gradient_descent(
        measurements_single,
        propagators_single,
        initial_phase=initial_phase,
        step_size=experiment_settings["step"],
        max_iter=experiment_settings["iterations"],
        reduction="mean",
    )
    stage2_constrained = projected_gradient_descent(
        measurements_single,
        propagators_single,
        initial_phase=initial_phase,
        nonpositive=True,
        support=support_mask,
        step_size=experiment_settings["step"],
        max_iter=experiment_settings["iterations"],
        reduction="mean",
    )
    _stage2_results = {
        "no constraints": stage2_free,
        constraint_text: stage2_constrained,
    }
    _ctf_estimate = stage1_results["single, unregularized"]
    _images = [_ctf_estimate] + [
        result.estimate for result in _stage2_results.values()
    ]
    _labels = ["linear CTF, no regularization"] + list(_stage2_results)
    if phase_truth is not None:
        _images = [phase_truth] + _images
        _labels = ["synthetic truth (reference)"] + _labels
    _rows = [
        {
            "method": "linear CTF, no regularization",
            "residual": relative_data_residual(
                _ctf_estimate, measurements_single, propagators_single
            ),
            "nrmse": phase_nrmse(_ctf_estimate, phase_truth, support_mask),
            "iterations": "closed form",
            "calls": 0,
            "seconds": None,
            "stop": "linear inverse",
        }
    ] + [
        result_row(
            label,
            result,
            phase_truth,
            support_mask,
            measurements_single,
            propagators_single,
        )
        for label, result in _stage2_results.items()
    ]
    _curves = {
        label: (range(len(result.data_residual)), result.data_residual)
        for label, result in _stage2_results.items()
    }
    mo.vstack(
        [
            mo.md(
                r"""
                ## 2 · Nonlinear single-distance reconstruction

                The first contrast replaces only the CTF approximation by the full
                DeepInv map $\phi \mapsto |P_z e^{i\phi}|^2$: both use one plane,
                no regularization, and no physical constraints. A nonlinear model
                necessarily replaces the closed-form inverse with an iterative solve,
                so computation is reported rather than silently treated as identical.
                The second contrast keeps the nonlinear run fixed and changes only the
                feasible set.
                """
            ),
            reconstruction_grid(
                _images, _labels, colorbar_label="phase [rad]", zero_ceiling=True
            ),
            curve_figure(
                _curves,
                xlabel="DeepInv PGD iteration",
                ylabel="relative intensity residual",
                log_y=True,
            ),
            mo.md(markdown_table(_rows, METRIC_COLUMNS)),
        ]
    )
    return (stage2_constrained,)


@app.cell(hide_code=True)
def _(
    METRIC_COLUMNS,
    constraint_text,
    curve_figure,
    experiment_settings,
    initial_phase,
    markdown_table,
    measurements,
    measurements_single,
    mo,
    phase_truth,
    projected_gradient_descent,
    propagators,
    propagators_single,
    reconstruction_grid,
    result_row,
    stage2_constrained,
    support_mask,
):
    stage3_multi = projected_gradient_descent(
        measurements,
        propagators,
        initial_phase=initial_phase,
        nonpositive=True,
        support=support_mask,
        step_size=experiment_settings["step"],
        max_iter=experiment_settings["iterations"],
        reduction="mean",
    )
    _stage3_results = {
        "single distance": stage2_constrained,
        "four distances": stage3_multi,
    }
    _images = [result.estimate for result in _stage3_results.values()]
    _labels = list(_stage3_results)
    if phase_truth is not None:
        _images = [phase_truth] + _images
        _labels = ["synthetic truth (reference)"] + _labels
    _rows = [
        result_row(
            "single distance",
            stage2_constrained,
            phase_truth,
            support_mask,
            measurements_single,
            propagators_single,
        ),
        result_row(
            "four distances",
            stage3_multi,
            phase_truth,
            support_mask,
            measurements,
            propagators,
        ),
    ]
    _curves = {
        label: (range(len(result.data_residual)), result.data_residual)
        for label, result in _stage3_results.items()
    }
    mo.vstack(
        [
            mo.md(
                rf"""
                ## 3 · Nonlinear single vs multi-distance

                Both runs use DeepInv PGD, {constraint_text}, the same zero
                initialization, step, and iteration count. Only the number of
                measurements changes. The stacked loss is **averaged over planes**
                so its gradient scale does not grow mechanically with $J$.
                """
            ),
            reconstruction_grid(
                _images, _labels, colorbar_label="phase [rad]", zero_ceiling=True
            ),
            curve_figure(
                _curves,
                xlabel="DeepInv PGD iteration",
                ylabel="relative intensity residual",
                log_y=True,
            ),
            mo.md(markdown_table(_rows, METRIC_COLUMNS)),
        ]
    )
    return (stage3_multi,)


@app.cell(hide_code=True)
def _(
    METRIC_COLUMNS,
    alternating_projections,
    constraint_text,
    curve_figure,
    experiment_settings,
    initial_phase,
    markdown_table,
    measurements,
    mo,
    phase_truth,
    propagators,
    reconstruction_grid,
    result_row,
    stage3_multi,
    support_mask,
):
    stage4_ap = alternating_projections(
        measurements,
        propagators,
        initial_phase=initial_phase,
        nonpositive=True,
        support=support_mask,
        max_iter=experiment_settings["iterations"],
    )
    _stage4_results = {"DeepInv PGD": stage3_multi, "averaged AP": stage4_ap}
    _rows = [
        result_row(
            label,
            result,
            phase_truth,
            support_mask,
            measurements,
            propagators,
        )
        for label, result in _stage4_results.items()
    ]
    _iteration_curves = {
        label: (range(len(result.data_residual)), result.data_residual)
        for label, result in _stage4_results.items()
    }
    _work_curves = {
        label: (result.operator_calls, result.data_residual)
        for label, result in _stage4_results.items()
    }
    mo.vstack(
        [
            mo.md(
                rf"""
                ## 4 · Optimizer comparison: averaged AP vs projected GD

                The data, full Fresnel model, {constraint_text}, zero initialization,
                and iteration count are identical. Both methods are judged with the
                same **intensity residual**, first per iteration and then per estimated
                forward/adjoint propagation budget.
                """
            ),
            mo.callout(
                "AP performs detector-amplitude projections; it is not gradient descent "
                "on the intensity least-squares objective. Thus this is an algorithmic "
                "comparison under shared conditions, not two solvers for an identical "
                "optimization trajectory.",
                kind="warn",
            ),
            reconstruction_grid(
                [result.estimate for result in _stage4_results.values()],
                list(_stage4_results),
                colorbar_label="phase [rad]",
                zero_ceiling=True,
            ),
            curve_figure(
                _iteration_curves,
                xlabel="iteration",
                ylabel="relative intensity residual",
                log_y=True,
            ),
            curve_figure(
                _work_curves,
                xlabel="estimated forward/adjoint-equivalent propagations",
                ylabel="relative intensity residual",
                log_y=True,
            ),
            mo.md(markdown_table(_rows, METRIC_COLUMNS)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    METRIC_COLUMNS,
    angular_frequency_radius,
    constrained_ctf_reconstruct,
    curve_figure,
    experiment_settings,
    fresnel_numbers,
    huhn_nltikh,
    huhn_regularization_filter,
    initial_phase,
    markdown_table,
    measurements,
    mo,
    phase_truth,
    plt,
    projected_gradient_descent,
    propagators,
    radial_average,
    reconstruction_grid,
    result_row,
    stage3_multi,
    support_mask,
):
    _planes = len(propagators)
    huhn_alpha = huhn_regularization_filter(
        tuple(initial_phase.shape[-2:]),
        fresnel_numbers,
        alpha_low=1e-3,
        alpha_high=1e-1,
        alpha_beyond_na=2.0 * _planes,
        device=initial_phase.device,
        dtype=initial_phase.dtype,
    )
    huhn_warm_start = constrained_ctf_reconstruct(
        measurements,
        propagators,
        alpha=huhn_alpha,
        nonpositive=True,
        support=support_mask,
        max_iter=60,
    )
    _sum_loss_step = experiment_settings["step"] / _planes
    stage5_warm = projected_gradient_descent(
        measurements,
        propagators,
        initial_phase=huhn_warm_start,
        nonpositive=True,
        support=support_mask,
        step_size=_sum_loss_step,
        max_iter=experiment_settings["iterations"],
        reduction="sum",
    )
    stage5_regularized = projected_gradient_descent(
        measurements,
        propagators,
        initial_phase=huhn_warm_start,
        alpha=huhn_alpha,
        nonpositive=True,
        support=support_mask,
        step_size=_sum_loss_step,
        max_iter=experiment_settings["iterations"],
        reduction="sum",
    )
    stage5_full = huhn_nltikh(
        measurements,
        propagators,
        fresnel_numbers,
        initial_phase=huhn_warm_start,
        nonpositive=True,
        support=support_mask,
        alpha_low=1e-3,
        alpha_high=1e-1,
        alpha_beyond_na=2.0 * _planes,
        max_iter=experiment_settings["iterations"],
        tolerance=1e-3,
    )
    _stage5_results = {
        "baseline": stage3_multi,
        "+ constrained CTF init": stage5_warm,
        "+ frequency Tikhonov": stage5_regularized,
        "+ BB / nonmonotone search": stage5_full,
    }
    _rows = [
        result_row(
            label,
            result,
            phase_truth,
            support_mask,
            measurements,
            propagators,
        )
        for label, result in _stage5_results.items()
    ]
    _curves = {
        label: (result.operator_calls, result.data_residual)
        for label, result in _stage5_results.items()
    }

    _radius = angular_frequency_radius(
        tuple(initial_phase.shape[-2:]),
        device=initial_phase.device,
        dtype=initial_phase.dtype,
    )
    _profile_radius, _profile_alpha = radial_average(_radius, huhn_alpha)
    _filter_figure, _filter_axis = plt.subplots(figsize=(6.4, 2.9))
    _filter_axis.plot(_profile_radius, _profile_alpha, color="#0072B2", linewidth=1.8)
    _mean_fresnel = sum(fresnel_numbers) / len(fresnel_numbers)
    _first_cutoff = 3.141592653589793 * (2.0 * _mean_fresnel) ** 0.5
    _na_cutoff = 3.141592653589793 * min(initial_phase.shape[-2:]) * _mean_fresnel
    for _cutoff, _text in ((_first_cutoff, "first CTF maximum"), (_na_cutoff, "NA cutoff")):
        _filter_axis.axvline(_cutoff, color="#777777", linewidth=0.8)
        _filter_axis.text(
            _cutoff,
            max(_profile_alpha) * 0.62,
            _text,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
            color="#555555",
        )
    _filter_axis.set_yscale("log")
    _filter_axis.set_xlabel("radial angular frequency [rad px⁻¹]")
    _filter_axis.set_ylabel("α(ξ)")
    _filter_figure.tight_layout()

    mo.vstack(
        [
            mo.md(
                """
                ## 5 · Huhn NLTikh, built as a sequential ablation

                Starting from the stage-3 multi-distance baseline, each column adds
                one ingredient: (1) a constraint-consistent CTF warm start, (2) the
                three-level Fourier Tikhonov filter, then (3) alternating
                Barzilai–Borwein steps, nonmonotone proximal backtracking, and Huhn's relative
                raw-gradient stopping rule. The same warm start is supplied to the
                last three runs so optimizer effects are not confounded with a second
                initialization. Propagation counts exclude that shared one-time CTF
                FFT solve.
                """
            ),
            reconstruction_grid(
                [result.estimate for result in _stage5_results.values()],
                list(_stage5_results),
                colorbar_label="phase [rad]",
                zero_ceiling=True,
            ),
            mo.md(
                "The filter uses α = 10⁻³ below the first CTF maximum, α = 10⁻¹ "
                "above it, and α = 2J beyond the numerical-aperture cutoff. These are "
                "paper-inspired defaults, not tuned hyperparameters; the ablation is "
                "allowed to show that they hurt a mismatched phantom or crop."
            ),
            _filter_figure,
            curve_figure(
                _curves,
                xlabel="estimated forward/adjoint-equivalent propagations",
                ylabel="relative intensity residual",
                log_y=True,
            ),
            mo.md(markdown_table(_rows, METRIC_COLUMNS)),
        ]
    )
    return (stage5_full,)


@app.cell(hide_code=True)
def _(
    DEVICE,
    NOTEBOOK_FULL_FIELD,
    PIXEL_SIZE,
    Path,
    WAVELENGTH,
    full_field_ctf_control,
    full_field_iteration_control,
    full_field_run_control,
    huhn_nltikh,
    make_fresnel_propagators,
    mo,
    np,
    perf_counter,
    source_control,
    torch,
):
    mo.stop(
        source_control.value != "measured",
        mo.md(
            "## Optional native-field reconstruction\n\n"
            "Select **Measured bead data** above to enable this independent run."
        ),
    )
    mo.stop(
        not (full_field_run_control.value or NOTEBOOK_FULL_FIELD),
        mo.vstack(
            [
                mo.md("## Optional native-field reconstruction"),
                mo.callout(
                    "Press **Run optional native-field NLTikh** in the controls to "
                    "reconstruct all 2048 × 1920 pixels. The five-stage comparison "
                    "can remain on a small crop.",
                    kind="info",
                ),
            ]
        ),
    )
    _data_path = Path(__file__).with_name("holograms_beads_updated.npz")
    with np.load(_data_path) as _archive:
        _native_stack = torch.from_numpy(_archive["holograms"]).float()
        _native_fresnel = tuple(
            float(value) for value in _archive["fresnelNumbers"]
        )
    _native_shape = tuple(_native_stack.shape[-2:])
    native_field_measurements = _native_stack[:, None, None].to(DEVICE)
    native_field_propagators = make_fresnel_propagators(
        _native_shape,
        _native_fresnel,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        device=DEVICE,
    )
    native_field_ctf_iterations = int(full_field_ctf_control.value)
    _started = perf_counter()
    native_field_result = huhn_nltikh(
        native_field_measurements,
        native_field_propagators,
        _native_fresnel,
        nonpositive=True,
        support=None,
        alpha_low=1e-3,
        alpha_high=1e-1,
        alpha_beyond_na=2.0 * len(_native_fresnel),
        max_iter=int(full_field_iteration_control.value),
        tolerance=1e-3,
        ctf_init_iterations=native_field_ctf_iterations,
    )
    native_field_total_seconds = perf_counter() - _started
    return (
        native_field_ctf_iterations,
        native_field_measurements,
        native_field_propagators,
        native_field_result,
        native_field_total_seconds,
    )


@app.cell(hide_code=True)
def _(
    full_field_reconstruction_figure,
    native_field_ctf_iterations,
    native_field_result,
    native_field_total_seconds,
    mo,
    size_control,
):
    _figure = full_field_reconstruction_figure(
        native_field_result.estimate, int(size_control.value)
    )
    _last_residual = native_field_result.data_residual[-1]
    _last_calls = native_field_result.operator_calls[-1]
    mo.vstack(
        [
            mo.md("## Optional native-field Huhn NLTikh reconstruction"),
            mo.callout(
                "This phase is reconstructed on the complete measured field. The "
                "orange inset enlarges its center using the comparison crop size; "
                "both views share one phase scale.",
                kind="info",
            ),
            _figure,
            mo.md(
                f"""
                **Shape:** {native_field_result.estimate.shape[-2]} ×
                {native_field_result.estimate.shape[-1]}  ·
                **NLTikh iterations:** {native_field_result.iterations}  ·
                **relative intensity residual:** {_last_residual:.3e}  ·
                **post-init equivalent propagations:** {_last_calls:,}  ·
                **total wall time:** {native_field_total_seconds:.1f} s  ·
                **stop:** {native_field_result.stop_reason}

                The run uses all four stored intensity planes, constrained CTF
                initialization ({native_field_ctf_iterations} iterations), the
                frequency-dependent Tikhonov filter, nonpositive
                phase, and the adaptive BB/nonmonotone policy. No compact support is
                imposed because the measured sample fills the field of view. The
                propagation estimate counts the nonlinear phase only; total wall time
                includes construction and CTF initialization.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, phase_truth, stage5_full):
    _claim = (
        "Because synthetic truth is available, phase NRMSE can be interpreted as "
        "reconstruction accuracy after fixing the constant-phase gauge from the known background."
        if phase_truth is not None
        else "No reference phase is available: residual reduction is evidence of data consistency, not object accuracy."
    )
    mo.vstack(
        [
            mo.md("## Interpretation and provenance"),
            mo.callout(_claim, kind="info"),
            mo.md(
                f"""
                The full run stopped after **{stage5_full.iterations}** iterations:
                **{stage5_full.stop_reason}**. Re-run with another seed, noise level,
                crop size, or phase strength to test whether a ranking is robust; a
                single phantom or crop is not a general benchmark.

                Implementation provenance:

                - Fresnel propagation, phase-retrieval physics, stacked L2 fidelity,
                  automatic VJPs, and fixed-step PGD are DeepInv objects.
                - The CTF closed form, support/sign projection, amplitude-projection
                  loop, Fourier filter, and Huhn BB/nonmonotone policy are thin local
                  adapters for ingredients not exposed by DeepInv.
                - [Huhn et al. (2022)](https://arxiv.org/html/2205.01099v2) defines
                  the CTF/NLTikh formulation and adaptive scheme.
                - [Hagemann et al. (2018)](https://doi.org/10.1063/1.5029927)
                  motivates averaged multi-distance alternating projections.
                - [HoToPy's holography API](https://irp.pages.gwdg.de/hotopy/reference/generated/hotopy.holo.html)
                  was consulted to cross-check the accelerated-ADMM restart and
                  nonmonotone proximal-backtracking conventions; it is not imported
                  and is not a project dependency.
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
