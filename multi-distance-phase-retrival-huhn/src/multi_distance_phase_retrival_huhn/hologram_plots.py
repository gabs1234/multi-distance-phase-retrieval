"""Matched intensity profiles across a stack of propagation distances."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .phase_retrieval import fresnel_numbers_to_distances


def plot_hologram_line_cuts(
    images,
    fresnel_numbers,
    *,
    wavelength,
    pixel_size,
    orientation="horizontal",
    index=None,
    band_width=1,
    context_images=None,
    crop_bounds=None,
):
    """Plot each hologram and its matching cut, ordered by increasing distance.

    ``images`` has shape (planes, height, width). Intensities are used as given.
    ``index`` is a zero-based row (horizontal) or column (vertical). An odd-width
    averaging band is centered on that index and clipped at the image boundary.
    All profiles share spatial and intensity limits; image contrast alone is
    clipped to pooled 1st/99th percentiles. If ``context_images`` is supplied,
    the left panels show those uncropped images with ``crop_bounds`` marked and
    the corresponding reconstruction crop as an inset. Distances use the same
    effective plane-wave convention as the DeepInv forward model.
    """
    arrays = np.asarray(images)
    numbers = tuple(float(value) for value in fresnel_numbers)
    if arrays.ndim != 3 or arrays.shape[0] != len(numbers):
        raise ValueError("images must have shape (planes, height, width), matching F")
    if min(arrays.shape) < 1 or not np.isfinite(arrays).all():
        raise ValueError("images must be nonempty and finite")
    context = None
    if context_images is not None:
        context = np.asarray(context_images)
        if context.ndim != 3 or context.shape[0] != arrays.shape[0]:
            raise ValueError("context_images must match the number of planes")
        if not np.isfinite(context).all():
            raise ValueError("context_images must be finite")
        if crop_bounds is None or len(crop_bounds) != 4:
            raise ValueError("crop_bounds=(top, left, height, width) is required")
        crop_top, crop_left, crop_height, crop_width = map(int, crop_bounds)
        if (
            crop_top < 0
            or crop_left < 0
            or crop_height != arrays.shape[1]
            or crop_width != arrays.shape[2]
            or crop_top + crop_height > context.shape[1]
            or crop_left + crop_width > context.shape[2]
        ):
            raise ValueError("crop_bounds do not locate images inside context_images")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be horizontal or vertical")
    if not isinstance(band_width, int) or band_width < 1 or band_width % 2 != 1:
        raise ValueError("band_width must be a positive odd integer")
    cross_size = arrays.shape[1 if orientation == "horizontal" else 2]
    if index is None:
        index = cross_size // 2
    if not isinstance(index, int) or not 0 <= index < cross_size:
        raise ValueError("line index is outside the image")
    start = max(0, index - band_width // 2)
    stop = min(cross_size, index + band_width // 2 + 1)
    if orientation == "horizontal":
        profiles = arrays[:, start:stop, :].mean(axis=1)
    else:
        profiles = arrays[:, :, start:stop].mean(axis=2)
    distances = np.asarray(
        fresnel_numbers_to_distances(numbers, wavelength, pixel_size)
    )
    order = np.argsort(distances, kind="stable")
    x = (np.arange(profiles.shape[1]) - (profiles.shape[1] - 1) / 2) * pixel_size * 1e6
    ymin = min(float(profiles.min()), 1.0)
    ymax = max(float(profiles.max()), 1.0)
    margin = max(0.08 * (ymax - ymin), 0.01)
    vmin, vmax = np.quantile(arrays, [0.01, 0.99])
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 0.01, vmax + 0.01
    if context is not None:
        context_vmin, context_vmax = np.quantile(context, [0.01, 0.99])
        if np.isclose(context_vmin, context_vmax):
            context_vmin, context_vmax = context_vmin - 0.01, context_vmax + 0.01
    style = {
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
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            len(order), 2,
            figsize=(9.2, 1.8 * len(order) + 1.2),
            squeeze=False,
            gridspec_kw={"width_ratios": [1, 3.5]},
        )
        for row, plane in enumerate(order):
            image_axis, profile_axis = axes[row]
            color = "#666666" if row == 0 else "#0072B2"
            if context is None:
                image_axis.imshow(
                    arrays[plane], cmap="gray", vmin=vmin, vmax=vmax,
                    origin="upper", interpolation="nearest",
                )
                cut_axis = image_axis
            else:
                image_axis.imshow(
                    context[plane], cmap="gray", vmin=context_vmin, vmax=context_vmax,
                    origin="upper", interpolation="nearest",
                )
                image_axis.add_patch(
                    Rectangle(
                        (crop_left - 0.5, crop_top - 0.5), crop_width, crop_height,
                        fill=False, edgecolor="#D55E00", linewidth=1.2,
                    )
                )
                cut_axis = image_axis.inset_axes([0.53, 0.03, 0.45, 0.45])
                cut_axis.imshow(
                    arrays[plane], cmap="gray", vmin=vmin, vmax=vmax,
                    origin="upper", interpolation="nearest",
                )
                cut_axis.set_title("crop", fontsize=7, color="#D55E00", pad=1)
                cut_axis.set_xticks([])
                cut_axis.set_yticks([])
                for spine in cut_axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#D55E00")
                    spine.set_linewidth(1.0)
                if row == 0:
                    image_axis.set_title("full field + crop inset", fontsize=8, pad=2)
            if orientation == "horizontal":
                cut_axis.axhline(index, color="#0072B2", linewidth=1.2)
                if stop - start > 1:
                    cut_axis.axhspan(start - 0.5, stop - 0.5, color="#0072B2", alpha=0.22)
            else:
                cut_axis.axvline(index, color="#0072B2", linewidth=1.2)
                if stop - start > 1:
                    cut_axis.axvspan(start - 0.5, stop - 0.5, color="#0072B2", alpha=0.22)
            image_axis.set_axis_off()
            if row:
                profile_axis.sharex(axes[0, 1])
                profile_axis.sharey(axes[0, 1])
                profile_axis.plot(x, profiles[order[0]], color="#666666", linestyle=":", linewidth=1.0)
            profile_axis.axhline(1.0, color="#888888", linestyle="--", linewidth=0.65)
            profile_axis.plot(x, profiles[plane], color=color, linewidth=1.4)
            profile_axis.set_title(
                f"Plane {plane + 1}:  F = {numbers[plane]:.4e}  ·  "
                f"effective z = {distances[plane] * 1e3:.2f} mm",
                fontsize=10, color=color,
            )
            profile_axis.set_xlim(float(x[0]), float(x[-1]))
            profile_axis.set_ylim(ymin - margin, ymax + margin)
            profile_axis.spines["bottom"].set_bounds(float(x[0]), float(x[-1]))
            profile_axis.spines["left"].set_bounds(ymin, ymax)
            profile_axis.tick_params(labelbottom=row == len(order) - 1)
            profile_axis.set_ylabel("Intensity I / I₀")
        coordinate = "x" if orientation == "horizontal" else "y"
        axes[-1, 1].set_xlabel(f"{coordinate} relative to crop center [µm]")
        line_name = "row" if orientation == "horizontal" else "column"
        selection = f"{line_name} {index}" if stop - start == 1 else f"{line_name}s {start}–{stop - 1} averaged"
        figure.suptitle("Registered fringes evolve as effective distance grows", x=0.06, ha="left", fontsize=14)
        figure.text(
            0.06, 0.932,
            f"{selection} (zero-based) · shared scales · dotted: first plane · dashed: I / I₀ = 1",
            fontsize=9, color="#555555",
        )
        figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, hspace=0.5, wspace=0.32)
    return figure
