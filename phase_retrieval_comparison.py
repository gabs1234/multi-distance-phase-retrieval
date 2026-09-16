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

    from multi_distance_phase_retrival_huhn.comparison_plots import (
        GROUPS, nonlinear_phase_limits, phase_comparison_figure,
    )
    from multi_distance_phase_retrival_huhn.notebook_plots import (
        image_grid, full_field_reconstruction_figure, tensor_image,
    )
    from multi_distance_phase_retrival_huhn.experiment_io import (
        DEFAULT_CACHE, DEFAULT_DATASET, load_results,
    )
    from multi_distance_phase_retrival_huhn.direct_methods import (
        tie_reconstruct, homogeneous_ctf_ict, polystyrene_8kev, predict_intensity,
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
        GROUPS,
        nonlinear_phase_limits,
        phase_comparison_figure,
        image_grid,
        full_field_reconstruction_figure,
        tensor_image,
        DEFAULT_CACHE,
        DEFAULT_DATASET,
        load_results,
        tie_reconstruct,
        homogeneous_ctf_ict,
        polystyrene_8kev,
        predict_intensity,
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
        Rectangle,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward model and notation

    ### From complex transmission to a fixed material ratio

    In the paper's notation, the two refractive-index components are
    $\delta$ (phase) and $\beta$ (absorption):
    $n=1-\delta+i\beta$. The symbol $\mu$ denotes **projected absorption**,
    not the phase component of the refractive index.
    With $k=2\pi/\lambda$, the transmission relative to vacuum is

    $$
    T(x)=\exp\!\left[ik\int(n(x,z)-1)\,dz\right]
        =\exp\!\left[-k\int\beta(x,z)\,dz
                     -ik\int\delta(x,z)\,dz\right]
        =e^{i\phi(x)-\mu(x)},
    $$
    $$
    \phi(x)=-k\int\delta(x,z)\,dz,\qquad
    \mu(x)=k\int\beta(x,z)\,dz.
    $$

    Thus positive absorption gives $|T|=e^{-\mu}\leq1$.
    This is the attenuating transmission convention behind Eq. (1).

    The **single-material constraint** assumes a known, spatially constant
    ratio $r=\beta/\delta$ at the acquisition energy. Substitution eliminates
    one unknown image:

    $$
    \beta(x,z)=r\,\delta(x,z)
    \quad\Longrightarrow\quad
    \mu(x)=kr\int\delta(x,z)\,dz=-r\,\phi(x)
    \quad\Longrightarrow\quad
    T(x)=e^{(i+r)\phi(x)}.
    $$

    Once $\phi$ is reconstructed, $\mu=-r\phi$ follows from the assumed
    ratio; it is not fitted independently. If instead we define the
    **signed projected ratio** $c=\mu/\phi$, the same elimination reads

    $$
    \mu=c\phi,\qquad T=e^{(i-c)\phi},\qquad c=-r.
    $$

    **Sign convention in the source.** Eq. (3) in the supplied Markdown,
    also reproduced in the
    [arXiv text, section 2](https://arxiv.org/html/2205.01099v2#S2),
    identifies $\mu/\phi$ with $+\beta/\delta$ and then uses
    $e^{(i-c_{\beta/\delta})\phi}$. With its stated definitions
    $\phi=-k\int\delta\,dz$ and $\mu=k\int\beta\,dz$, those ratios have
    opposite signs. The algebra above keeps the positive material ratio
    $r=\beta/\delta$ separate from the signed ratio $c=\mu/\phi$.
    Both conventions reduce to $T=e^{i\phi}$ at zero absorption.

    ### What this notebook actually optimizes

    Every reconstruction below estimates **one real phase image** and fixes
    $\mu=0$: the nonlinear transmission is implemented as
    `torch.exp(1j * phase)`. Neither a nonzero material ratio nor an independent
    absorption image is currently fitted. The CTF baseline is also pure-phase,
    and AP resets the object transmission to unit amplitude.

    In this notebook, **unconstrained** means that phase sign and support
    constraints are disabled; the pure-phase assumption still applies.
    Nonlinear reconstruction removes the weak-object linearization, but does
    not add absorption as a second unknown.

    An independent phase-and-absorption reconstruction would instead optimize
    two real images, for example

    $$
    (\phi_\star,\mu_\star)\in
    \underset{\phi,\ \mu\geq0}{\operatorname{argmin}}
    \left\{
        \frac12\sum_{j=1}^{J}
            \left\||P_j e^{i\phi-\mu}|^2-I_j\right\|_2^2
        +R_\phi(\phi)+R_\mu(\mu)
    \right\},
    $$

    where $R_\phi$ and $R_\mu$ are chosen regularizers and no proportionality
    between $\phi$ and $\mu$ is imposed. This is a possible extension,
    **not an implemented run**. The unknowns would be the projected
    quantities, rather than separate three-dimensional $\delta$ and $\beta$.

    ### Pure-phase forward model used below

    We use the **pure-phase** case of Huhn et al. (2022), Eq. (1):
    absorption is zero ($c_{\beta/\delta}=0$), and the unknown $\phi$ is in radians.
    For normalized intensity $I_j$ at distance $j$,

    $$
    I_j \approx N_j(\phi):=\left|P_j e^{i\phi}\right|^2,\qquad
    P_j u=\mathcal F^{-1}\!\left[
        e^{-i|\xi|^2/(4\pi F_j)}\,\mathcal F u
    \right],\qquad F_j=\frac{\Delta x^2}{\lambda z_j}.
    $$

    Here $J$ is the number of holograms, $\xi$ is angular spatial frequency
    in rad/pixel, and $\mathcal F$ is the orthonormal discrete Fourier
    transform used in the code. Norms below sum over image pixels.
    $P_j^*$ denotes adjoint (backward) Fresnel propagation.

    The constrained runs use the feasible set and pointwise projection
    from the pure-phase setting of Eq. (6):

    $$
    A=\{\phi:\phi(x)\leq 0,\ \phi(x)=0\text{ for }x\notin\Omega\},
    \qquad
    [\Pi_A(v)](x)=
    \begin{cases}
    \min(v(x),0),&x\in\Omega,\\
    0,&x\notin\Omega.
    \end{cases}
    $$

    For synthetic data, $\Omega$ is the known support; for measured data,
    it is the entire image, so only the sign constraint applies.
    In unconstrained runs, $A=\mathbb R^{H\times W}$ and $\Pi_A(v)=v$.
    The equations below state the reconstruction targets and updates;
    iterative results are finite-iteration approximations.
    """)
    return


@app.cell
def _(
    COLORS,
    Rectangle,
    ctf_transfer_functions,
    mo,
    np,
    plt,
    relative_data_residual,
    support_referenced_phase_nrmse,
    tensor_image,
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

    def phase_detail_controls(shape):
        height, width = shape
        sizes = [size for size in (64, 128, 256, 512) if size <= min(shape)]
        if not sizes:
            sizes = [min(shape)]
        default_size = 128 if 128 in sizes else sizes[0]
        detail_size = mo.ui.dropdown(
            options={f"{size} × {size}": size for size in sizes},
            value=f"{default_size} × {default_size}",
            label="close-up size",
        )
        detail_row = mo.ui.slider(
            start=0, stop=height - 1, value=height // 2,
            step=1, show_value=True, label="close-up center row",
        )
        detail_column = mo.ui.slider(
            start=0, stop=width - 1, value=width // 2,
            step=1, show_value=True, label="close-up center column",
        )
        return detail_size, detail_row, detail_column

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
        make_bead_phantom,
        markdown_table,
        phase_detail_controls,
        phase_nrmse,
        radial_average,
        result_row,
    )


@app.cell(hide_code=True)
def _(DEFAULT_CACHE, GROUPS, mo):
    _imported = DEFAULT_CACHE.parent / "presentation"
    saved_cache_control = mo.ui.text(
        value=str(_imported if _imported.exists() else DEFAULT_CACHE),
        label="converged results directory", full_width=True,
    )
    saved_group_control = mo.ui.dropdown(
        options={title.replace(" · tight ROI", ""): i for i, (_, title, _) in enumerate(GROUPS)},
        value=GROUPS[1][1].replace(" · tight ROI", ""), label="comparison",
    )
    saved_run_control = mo.ui.run_button(label="Load converged measured-data results")
    mo.vstack([
        mo.md("## Converged results used in the presentation"),
        mo.md("Load saved native-field results to inspect the same full views and tight sphere ROI "
              "as the slides. This does not rerun the solvers. Every nonlinear panel uses a "
              "single phase scale computed from all full-field estimates; CTF has its own scale. "
              "The interactive experiments below remain short, finite-iteration demonstrations."),
        saved_cache_control, mo.hstack([saved_group_control, saved_run_control]),
    ])
    return saved_cache_control, saved_group_control, saved_run_control


@app.cell
def _(DEFAULT_DATASET, load_results, mo, saved_cache_control, saved_run_control):
    mo.stop(not saved_run_control.value)
    saved_maps, saved_metrics, saved_sources = load_results(saved_cache_control.value, DEFAULT_DATASET)
    return saved_maps, saved_metrics, saved_sources


@app.cell(hide_code=True)
def _(GROUPS, mo, nonlinear_phase_limits, phase_comparison_figure,
      saved_group_control, saved_maps, saved_metrics):
    _, _title, _methods = GROUPS[saved_group_control.value]
    _limits = nonlinear_phase_limits(saved_maps)
    _full, _ = phase_comparison_figure(saved_maps, saved_metrics, _methods,
                                      _title.replace(" · tight ROI", " · full field"), _limits)
    _roi, _ = phase_comparison_figure(saved_maps, saved_metrics, _methods,
                                     _title, _limits, roi=(583, 870, 192))
    mo.vstack([_full, _roi, mo.md(
        "Numerical stopping: projected-gradient ratio ≤ 10⁻³ for PGD/NLTikh; "
        "last 20 relative phase updates ≤ 10⁻⁵ for AP. Both also require the "
        "100-step objective change divided by E(0) ≤ 10⁻⁴. The data have no phase ground truth."
    )])
    return


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
        label="ROI / synthetic image size",
    )
    field_control = mo.ui.dropdown(
        options={
            "Central ROI": "roi",
            "Full projections": "full",
        },
        value="Central ROI",
        label="measured reconstruction field",
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
            mo.hstack([source_control, field_control, size_control], widths="equal"),
            mo.hstack([iteration_control, step_control], widths="equal"),
            mo.hstack([phase_control, noise_control], widths="equal"),
            mo.hstack([ctf_alpha_control, seed_control, run_control], widths="equal"),
            mo.callout(
                "All five stages use the same reconstruction field. For measured data, "
                "choose a central ROI or all 2048 × 1920 pixels at native sampling. "
                "Full projections use about 240× as many pixels as the default ROI, "
                "so need more time and memory. A separate close-up viewer lets you "
                "inspect the reconstructed phase without rerunning the solvers. "
                "For synthetic data, the size sets the simulated field. "
                "Change reconstruction settings, then press Run all five stages.",
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
        field_control,
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
    field_control,
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
        "field": field_control.value,
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
        full_measurements = _full_stack
        if experiment_settings["field"] == "full":
            crop_bounds = None
            _selected_stack = _full_stack
            _selection_note = (
                f"All five stages reconstruct the full {_full_height} × {_full_width} "
                "field at native sampling. The close-up viewer crops only the "
                "displayed phase after reconstruction. "
            )
        else:
            crop_bounds = (
                (_full_height - _size) // 2,
                (_full_width - _size) // 2,
                _size,
                _size,
            )
            _selected_stack = centered_crop(_full_stack, _size)
            _selection_note = (
                f"All five stages reconstruct a central {_size} × {_size} ROI. "
                "Each phase panel includes a full-field inset with this ROI outlined; "
                "phase outside the outline has not been reconstructed. "
                "Cropping plus FFT-periodic boundaries can create edge-model mismatch. "
            )
        measurements = _selected_stack[:, None, None].to(DEVICE)
        clean_measurements = None
        phase_truth = None
        support_mask = None
        propagators = make_fresnel_propagators(
            tuple(measurements.shape[-2:]),
            fresnel_numbers,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            device=DEVICE,
        )
        source_note = (
            _selection_note
            + "Measured holograms come from the local holograms_beads_updated.npz "
            "with their stored intensity values preserved. The accompanying demo documents "
            "dark/flat-field correction, registration, and common magnification; the archive "
            "does not contain the raw calibration fields. No additional correction or "
            "per-crop mean normalization is applied. The sample fills the field of view, "
            "so no support mask is asserted."
        )

    measurements_single = measurements[:1]
    propagators_single = propagators[:1]
    fresnel_numbers_single = fresnel_numbers[:1]
    initial_phase = torch.zeros_like(measurements[0])
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
def _(mo):
    cut_orientation = mo.ui.dropdown(
        options=["horizontal", "vertical"], value="horizontal", label="cut direction"
    )
    cut_band = mo.ui.dropdown(
        options={"1 pixel": 1, "3 pixels": 3, "5 pixels": 5, "9 pixels": 9},
        value="1 pixel", label="averaging band",
    )
    return cut_band, cut_orientation


@app.cell(hide_code=True)
def _(cut_band, cut_orientation, measurements, mo):
    _cross_size = measurements.shape[
        -2 if cut_orientation.value == "horizontal" else -1
    ]
    cut_position = mo.ui.slider(
        start=0, stop=_cross_size - 1,
        value=_cross_size // 2, step=1, show_value=True,
        label="row / column index (zero-based)",
    )
    mo.hstack([cut_orientation, cut_position, cut_band], wrap=True)
    return (cut_position,)


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
            None if crop_bounds is None else full_measurements.numpy()
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
            hologram. With **Central ROI**, the orange rectangle marks the reconstruction
            crop, enlarged in the inset. With **Full projections**, the cut crosses the
            full measured field. For synthetic data, the left panel is the complete
            simulated field. The blue marker locates the cut; a shaded band appears
            when averaging is enabled.
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


@app.cell(hide_code=True)
def _(experiment_settings, fresnel_numbers, homogeneous_ctf_ict, image_grid,
      measurements_single, mo, np, polystyrene_8kev, predict_intensity,
      tensor_image, tie_reconstruct):
    mo.stop(experiment_settings["source"] != "measured")
    _observed = tensor_image(measurements_single[0]).astype(np.float64)
    _number = fresnel_numbers[0]
    _alpha = experiment_settings["ctf_alpha"]
    _gamma = polystyrene_8kev()["gamma"]
    _tie = tie_reconstruct(_observed, _number, alpha=_alpha)
    _ctf, _ict, _contact = homogeneous_ctf_ict(_observed, _number, gamma=_gamma, alpha=_alpha)
    _residuals = [float(np.linalg.norm(predict_intensity(_phase, _number, gamma=_ratio) - _observed)
                       / np.linalg.norm(_observed - 1))
                  for _phase, _ratio in ((_tie, None), (_ctf, _gamma), (_ict, _gamma))]
    mo.vstack([
        mo.md(f"## Direct TIE and matched CTF / ICT on the selected measured field\n\n"
              f"TIE uses a pure-phase model. The homogeneous CTF and ICT pair both use "
              f"the tabulated polystyrene ratio γ = {_gamma:.1f} and the same α = {_alpha:g}. "
              "The remaining five stages retain their pure-phase model. Residuals measure "
              "agreement with intensities, not phase accuracy."),
        image_grid([_tie], [f"TIE · intensity residual {_residuals[0]:.4f}"], colorbar_label="phase [rad]"),
        image_grid([_ctf, _ict], [f"Homogeneous CTF · residual {_residuals[1]:.4f}",
                                 f"ICT · residual {_residuals[2]:.4f}"], colorbar_label="phase [rad]"),
    ])
    return


@app.cell
def _(nonlinear_phase_limits, tensor_image, stage2_free, stage2_constrained,
      stage3_multi, stage4_ap, stage5_warm, stage5_regularized, stage5_full):
    comparison_nonlinear_limits = nonlinear_phase_limits({
        "pgd_free": tensor_image(stage2_free.estimate),
        "pgd_single": tensor_image(stage2_constrained.estimate),
        "pgd_multi": tensor_image(stage3_multi.estimate),
        "ap": tensor_image(stage4_ap.estimate),
        "pgd_warm": tensor_image(stage5_warm.estimate),
        "pgd_tikh": tensor_image(stage5_regularized.estimate),
        "nltikh": tensor_image(stage5_full.estimate),
    })
    return (comparison_nonlinear_limits,)


@app.cell
def _(crop_bounds, full_measurements, image_grid, comparison_nonlinear_limits):
    _context = None if crop_bounds is None else full_measurements[0]

    def reconstruction_grid(
        images, labels, *, colorbar_label, cmap="magma", zero_ceiling=False, separate_indices=()
    ):
        return image_grid(
            images,
            labels,
            colorbar_label=colorbar_label,
            cmap=cmap,
            zero_ceiling=zero_ceiling,
            color_limits=comparison_nonlinear_limits if zero_ceiling else None,
            separate_indices=separate_indices,
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
            mo.md(r"""
                **Phase retrieval — Huhn et al., Eqs. (2) and (4), with
                $c_{\beta/\delta}=0$.** Define the phase CTF
                $h_j(\xi)=2\sin(|\xi|^2/(4\pi F_j))$ and the linear prediction
                $L_j(\phi)=1+\mathcal F^{-1}[h_j\mathcal F\phi]$. Then

                $$
                \phi_{\mathrm{CTF}}
                =\underset{\phi\in\mathbb R^{H\times W}}{\operatorname{argmin}}
                \left\{
                    \frac12\sum_{j=1}^{J}\|L_j(\phi)-I_j\|_2^2
                    +\frac12\|\sqrt{\alpha}\,\mathcal F\phi\|_2^2
                \right\}
                =\mathcal F^{-1}\!\left[
                    \frac{\sum_{j=1}^{J}h_j\,\mathcal F(I_j-1)}
                         {\alpha+\sum_{j=1}^{J}h_j^2}
                \right].
                $$

                The four panels use $J=1$ or $4$ and $\alpha=0$ or the
                selected scalar value. Numerically, the Fourier estimate
                is set to zero wherever the denominator is at most
                $10^{-6}$ times its maximum, including DC when $\alpha=0$.
                The common factor $1/2$ leaves the paper's minimizer unchanged.
                """),
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


@app.cell
def _(
    experiment_settings,
    initial_phase,
    measurements_single,
    projected_gradient_descent,
    propagators_single,
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
    return (stage2_constrained, stage2_free,)


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
    stage2_constrained,
    stage2_free,
):
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
                no regularization, and no sign/support constraints. A nonlinear model
                necessarily replaces the closed-form inverse with an iterative solve,
                so computation is reported rather than silently treated as identical.
                The second contrast keeps the nonlinear run fixed and changes only the
                feasible set.
                """
            ),
            mo.md(r"""
                **Phase retrieval — Eq. (7) with $J=1$, $\alpha=0$,
                solved by Eq. (11).**

                $$
                \phi_\star\in\underset{\phi\in A}{\operatorname{argmin}}
                f_1(\phi),\qquad
                f_1(\phi)=\frac12\|N_1(\phi)-I_1\|_2^2,
                \qquad
                \phi_{k+1}=\Pi_A\!\left(\phi_k-\eta\nabla f_1(\phi_k)\right).
                $$

                With $u_1=P_1e^{i\phi}$, the gradient is the pure-phase
                specialization of Eqs. (9)–(10), scaled for the code's
                half-squared loss:

                $$
                \nabla f_1(\phi)=
                2\operatorname{Re}\!\left[
                    \overline{i e^{i\phi}}\,
                    P_1^*\!\left(u_1\,(|u_1|^2-I_1)\right)
                \right].
                $$

                Both nonlinear runs start at $\phi_0=0$ and use the selected
                fixed step $\eta$. The free run uses the identity projection;
                the constrained run uses $\Pi_A$ defined above.
                """),
            reconstruction_grid(
                _images, _labels, colorbar_label="phase [rad]", zero_ceiling=True,
                separate_indices=(1 if phase_truth is not None else 0,)
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
    return


@app.cell
def _(
    experiment_settings,
    initial_phase,
    measurements,
    projected_gradient_descent,
    propagators,
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
    return (stage3_multi,)


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
    stage3_multi,
):
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
            mo.md(r"""
                **Phase retrieval — the unregularized Eq. (7), averaged
                over distances, with the projected update of Eq. (11).**

                $$
                \phi_\star\in\underset{\phi\in A}{\operatorname{argmin}}
                f_J(\phi),\qquad
                f_J(\phi)=\frac{1}{2J}\sum_{j=1}^{J}\|N_j(\phi)-I_j\|_2^2,
                $$
                $$
                \phi_{k+1}=\Pi_A\!\left[
                    \phi_k-\frac{\eta}{J}\sum_{j=1}^{J}
                    N_j'[\phi_k]^*(N_j(\phi_k)-I_j)
                \right],\qquad \phi_0=0.
                $$

                Here $N_j'[\phi]^*r=
                2\operatorname{Re}[\overline{i e^{i\phi}}\,P_j^*((P_j e^{i\phi})r)]$.
                The two runs use $J=1$ and $J=4$ with the same $\eta$ and $A$.
                The factor $1/(2J)$ rescales the paper's unregularized
                objective without changing its minimizers.
                """),
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
    return


@app.cell
def _(
    alternating_projections,
    experiment_settings,
    initial_phase,
    measurements,
    propagators,
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
    return (stage4_ap,)


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
    stage4_ap,
):
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
            mo.md(r"""
                **Phase updates.** PGD uses
                $\phi_{k+1}=\Pi_A(\phi_k-\eta\nabla f_J(\phi_k))$ from
                section 3. The averaged AP implementation instead uses

                $$
                u_{j,k}=P_j e^{i\phi_k},\qquad
                \widetilde u_{j,k}
                =\sqrt{\max(I_j,0)}\,
                    \frac{u_{j,k}}{\max(|u_{j,k}|,\varepsilon)},
                $$
                $$
                \phi_{k+1}=\Pi_A\!\left[
                    \phi_k+\arg\!\left(e^{-i\phi_k}\frac1J\sum_{j=1}^{J}
                        P_j^*\widetilde u_{j,k}\right)
                \right],\qquad \phi_0=0,\quad\varepsilon=10^{-8}.
                $$

                Each detector amplitude is replaced by the measured amplitude,
                then the back-propagated waves are averaged. Phase increments are
                tracked on the previous branch before applying the constraints,
                avoiding a spurious jump to zero when the phase crosses $-\pi$. These equations describe this
                notebook's AP comparator; Huhn et al. discuss AP in section 4.1
                and reference [25], rather than defining it by their NLTikh
                gradient update.
                """),
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


@app.cell
def _(
    constrained_ctf_reconstruct,
    experiment_settings,
    fresnel_numbers,
    huhn_nltikh,
    huhn_regularization_filter,
    initial_phase,
    measurements,
    projected_gradient_descent,
    propagators,
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
    return (huhn_alpha, huhn_warm_start, stage5_full, stage5_regularized, stage5_warm,)


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
    huhn_alpha,
    huhn_warm_start,
    stage5_full,
    stage5_regularized,
    stage5_warm,
):
    _planes = len(propagators)
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
            mo.md(r"""
                **Phase retrieval — nonlinear Tikhonov, Eqs. (7) and (11).**
                With the code's factor $1/2$ applied to both terms,

                $$
                \phi_\star\in\underset{\phi\in A}{\operatorname{argmin}}
                E_\alpha(\phi),\qquad
                E_\alpha(\phi)=
                    \frac12\sum_{j=1}^{J}\|N_j(\phi)-I_j\|_2^2
                    +\frac12\|\sqrt{\alpha}\,\mathcal F\phi\|_2^2,
                $$
                $$
                g_k=\nabla E_\alpha(\phi_k)=
                    \sum_{j=1}^{J}N_j'[\phi_k]^*(N_j(\phi_k)-I_j)
                    +\mathcal F^{-1}[\alpha\,\mathcal F\phi_k],
                \qquad
                \phi_{k+1}=\Pi_A(\phi_k-\tau_k g_k).
                $$

                The **constrained CTF initialization** solves Eq. (6):

                $$
                \phi_0\approx\phi_{\mathrm{cCTF}}
                =\underset{\phi\in A}{\operatorname{argmin}}
                \left\{
                    \frac12\sum_{j=1}^{J}\|L_j(\phi)-I_j\|_2^2
                    +\frac12\|\sqrt{\alpha}\,\mathcal F\phi\|_2^2
                \right\}.
                $$

                The code computes this warm start with accelerated ADMM
                (paper section 3.1, Eq. (8)). The ablation then uses:

                | Run | Initial phase | Nonlinear objective | Step |
                | --- | --- | --- | --- |
                | baseline | $0$ | $f_J=E_0/J$ | fixed $\eta$ |
                | + constrained CTF init | $\phi_{\mathrm{cCTF}}$ | $E_0$ | fixed $\eta/J$ |
                | + frequency Tikhonov | $\phi_{\mathrm{cCTF}}$ | $E_\alpha$ | fixed $\eta/J$ |
                | full NLTikh | $\phi_{\mathrm{cCTF}}$ | $E_\alpha$ | adaptive $\tau_k$ |

                All three warm-started runs share the same regularized cCTF
                initialization. The $\eta/J$ adjustment preserves the baseline
                data-gradient step when switching from a mean to a summed loss.

                **Frequency weights — Eqs. (5) and (14).** Write
                $\bar F=J^{-1}\sum_j F_j$, $r_1=\pi\sqrt{2\bar F}$ and
                $r_{\mathrm{NA}}=\pi D\bar F$, with $D=\min(H,W)$ in this code.
                The intended three levels are

                $$
                \alpha(\xi)\approx
                \begin{cases}
                10^{-3},&|\xi|<r_1,\\
                10^{-1},&r_1<|\xi|<r_{\mathrm{NA}},\\
                2J,&|\xi|>r_{\mathrm{NA}}.
                \end{cases}
                $$

                The implementation uses smooth error-function transitions
                between these levels.

                **Adaptive steps and stopping — Eqs. (12)–(13).** For
                $s_k=\phi_k-\phi_{k-1}$ and $y_k=g_k-g_{k-1}$,

                $$
                \tau_k^{\mathrm{BB}}=
                \begin{cases}
                \langle s_k,y_k\rangle/\|y_k\|_2^2,&k\text{ odd},\\
                \|s_k\|_2^2/\langle s_k,y_k\rangle,&k\text{ even},
                \end{cases}
                \qquad
                \frac{\|g_k\|_2}{\|\nabla E_\alpha(0)\|_2}\leq10^{-3}.
                $$

                Positive, finite BB proposals are clipped to the allowed
                step range and checked by nonmonotone proximal backtracking.
                The gradient criterion or the iteration limit ends the run;
                a failed line search also stops it.
                """),
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
    return


@app.cell(hide_code=True)
def _(
    crop_bounds,
    experiment_settings,
    measurements,
    mo,
    phase_detail_controls,
    stage1_results,
    stage2_constrained,
    stage2_free,
    stage3_multi,
    stage4_ap,
    stage5_full,
    stage5_regularized,
    stage5_warm,
):
    mo.stop(experiment_settings["source"] != "measured" or crop_bounds is not None)
    comparison_phases = {
        **{f"1 · CTF: {label}": phase for label, phase in stage1_results.items()},
        "2 · Single-distance nonlinear, free": stage2_free.estimate,
        "2 · Single-distance nonlinear, constrained": stage2_constrained.estimate,
        "3 · Multi-distance PGD": stage3_multi.estimate,
        "4 · Averaged AP": stage4_ap.estimate,
        "5 · PGD + CTF initialization": stage5_warm.estimate,
        "5 · PGD + frequency Tikhonov": stage5_regularized.estimate,
        "5 · Full Huhn NLTikh": stage5_full.estimate,
    }
    detail_method = mo.ui.dropdown(
        options=list(comparison_phases),
        value="5 · Full Huhn NLTikh",
        label="reconstruction to inspect",
    )
    detail_size, detail_row, detail_column = phase_detail_controls(
        measurements.shape[-2:]
    )
    mo.vstack(
        [
            mo.md("## Full-field phase close-up"),
            mo.md(
                "Select a reconstruction from any section and move the close-up "
                "using its center row and column (zero-based). "
                "These controls only change the view; they do not rerun any solver. "
                "The orange box locates the enlarged region. "
                "All nonlinear comparisons and close-ups share one full-field phase scale; "
                "CTF has a separately labeled scale."
            ),
            detail_method,
            mo.hstack([detail_size, detail_row, detail_column], wrap=True),
        ]
    )
    return comparison_phases, detail_column, detail_method, detail_row, detail_size


@app.cell(hide_code=True)
def _(
    comparison_phases,
    comparison_nonlinear_limits,
    detail_column,
    detail_method,
    detail_row,
    detail_size,
    full_field_reconstruction_figure,
):
    full_field_reconstruction_figure(
        comparison_phases[detail_method.value],
        int(detail_size.value),
        color_limits=None if detail_method.value.startswith("1 · CTF") else comparison_nonlinear_limits,
        center_row=int(detail_row.value),
        center_column=int(detail_column.value),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Equation for the optional native-field reconstruction

    The optional run applies the same NLTikh objective (Huhn et al.,
    Eq. (7)) to all measured pixels and all four distances:

    $$
    \phi_\star^{\mathrm{native}}\in
    \underset{\phi\leq0}{\operatorname{argmin}}
    \left\{
        \frac12\sum_{j=1}^{4}
            \left\||P_j^{\mathrm{native}}e^{i\phi}|^2-I_j^{\mathrm{native}}\right\|_2^2
        +\frac12\|\sqrt{\alpha_{\mathrm{native}}}\,\mathcal F\phi\|_2^2
    \right\}.
    $$

    It starts from a constrained CTF reconstruction and uses the adaptive
    projected steps in section 5, with $\Pi_A(v)=\min(v,0)$.
    No support mask is supplied. The filter is recomputed on the native
    grid, using $D=\min(2048,1920)$ and $\alpha_{\mathrm{beyond\text{-}NA}}=8$;
    the optional controls set the CTF and nonlinear iteration budgets.
    """)
    return


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
def _(mo, native_field_result, phase_detail_controls):
    native_detail_size, native_detail_row, native_detail_column = phase_detail_controls(
        native_field_result.estimate.shape[-2:]
    )
    mo.vstack(
        [
            mo.md("### Optional NLTikh close-up controls"),
            mo.md(
                "Move the close-up within the reconstructed field. "
                "These display controls do not rerun the optional reconstruction."
            ),
            mo.hstack(
                [native_detail_size, native_detail_row, native_detail_column],
                wrap=True,
            ),
        ]
    )
    return native_detail_column, native_detail_row, native_detail_size


@app.cell(hide_code=True)
def _(
    full_field_reconstruction_figure,
    native_detail_column,
    native_detail_row,
    native_detail_size,
    native_field_ctf_iterations,
    native_field_result,
    native_field_total_seconds,
    mo,
):
    _figure = full_field_reconstruction_figure(
        native_field_result.estimate,
        int(native_detail_size.value),
        center_row=int(native_detail_row.value),
        center_column=int(native_detail_column.value),
    )
    _last_residual = native_field_result.data_residual[-1]
    _last_calls = native_field_result.operator_calls[-1]
    mo.vstack(
        [
            mo.md("## Optional native-field Huhn NLTikh reconstruction"),
            mo.callout(
                "This phase is reconstructed on the complete measured field. The "
                "orange box locates the enlarged region shown alongside it; "
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
