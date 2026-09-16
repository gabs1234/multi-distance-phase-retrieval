"""Notebook views sharing the comparison's phase normalization policy."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .comparison_plots import phase_limits, colorbar_extension


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
    color_limits=None,
    separate_indices=(),
):
    arrays = [tensor_image(image) for image in images]
    if color_limits is None:
        vmin, vmax = phase_limits(arrays)
        if np.isclose(vmin, vmax):
            vmin, vmax = float(vmin) - 1, float(vmax) + 1
    else:
        vmin, vmax = color_limits
    separate_indices = set(separate_indices)
    if separate_indices and (min(separate_indices) < 0 or max(separate_indices) >= len(arrays)):
        raise ValueError("separate_indices must refer to displayed panels")
    separate_limits = phase_limits([arrays[i] for i in separate_indices]) if separate_indices else None
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
    figure = plt.figure(
        figsize=(3.15 * len(arrays) + 0.75, 3.5), layout="constrained"
    )
    grid = figure.add_gridspec(2, len(arrays), height_ratios=[1, 0.065])
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(arrays))]
    pictures = []
    for index, (axis, array, label) in enumerate(zip(axes, arrays, labels, strict=True)):
        low, high = separate_limits if index in separate_indices else (vmin, vmax)
        last_image = axis.imshow(array, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        pictures.append(last_image)
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
    # Mixed CTF/nonlinear rows need distinct bars; every nonlinear bar uses
    # the same limits supplied from the complete set of full-field estimates.
    for separate in (False, True):
        indices = [i for i in range(len(arrays)) if (i in separate_indices) == separate]
        if not indices:
            continue
        if separate_indices:
            cax = figure.add_subplot(grid[1, indices[0]])
        else:
            cax = figure.add_subplot(grid[1, :])
        group_limits = separate_limits if separate else (vmin, vmax)
        figure.colorbar(pictures[indices[0]], cax=cax, orientation="horizontal",
                       label=("CTF phase [rad] · separate scale" if separate else colorbar_label),
                       extend=colorbar_extension([arrays[i] for i in indices], group_limits))
    return figure

def full_field_reconstruction_figure(
    phase, detail_size, *, center_row=None, center_column=None, color_limits=None
):
    array = tensor_image(phase)
    height, width = array.shape
    detail_height = min(int(detail_size), height)
    detail_width = min(int(detail_size), width)
    if center_row is None:
        center_row = height // 2
    if center_column is None:
        center_column = width // 2
    top = max(0, min(int(center_row) - detail_height // 2, height - detail_height))
    left = max(0, min(int(center_column) - detail_width // 2, width - detail_width))
    if color_limits is None:
        vmin, vmax = phase_limits([array])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(vmin) - 1, float(vmax) + 1
    else:
        vmin, vmax = color_limits
    figure = plt.figure(figsize=(10.0, 4.8), layout="constrained")
    grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    axis = figure.add_subplot(grid[0, 0])
    detail_axis = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[0, 2])
    phase_image = axis.imshow(array, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
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
        f"Full reconstruction · {height} × {width}", fontsize=12
    )
    axis.set_axis_off()
    detail_axis.imshow(
        array[top : top + detail_height, left : left + detail_width],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    detail_axis.set_title(
        f"Close-up · {detail_height} × {detail_width}\n"
        f"rows {top}–{top + detail_height - 1}, "
        f"columns {left}–{left + detail_width - 1}",
        fontsize=10,
        color="#9D3D00",
    )
    detail_axis.set_axis_off()
    figure.colorbar(
        phase_image, cax=colorbar_axis, label="phase [rad]",
        extend=colorbar_extension([array], (vmin, vmax)),
    )
    return figure

