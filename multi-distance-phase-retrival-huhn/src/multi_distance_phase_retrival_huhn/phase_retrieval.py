r"""Reference implementations for controlled Fresnel phase-retrieval studies.

The routines in this module deliberately expose the individual ingredients of
the reconstruction.  They are small enough to audit against the equations in
Huhn et al., while using :class:`deepinv.physics.FresnelPropagation` for the
actual wave propagation.

The implemented object model is the pure-phase convention used for the bead
data in the paper,

.. math::

    \psi = \exp(i\phi), \qquad \phi \leq 0.

All FFTs are orthonormal.  Consequently the spatial and Fourier-domain squared
norms have identical scaling, which makes the Tikhonov weights unambiguous.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time

import torch
from torch import Tensor

import deepinv as dinv
from deepinv.physics import FresnelPropagation


# With the Gaussian-CDF parameterization in ``_smooth_step``, this gives the
# same default span from the first CTF maximum toward its first nonzero root as
# the error-function construction used in HoToPy.
DEFAULT_TRANSITION_FRACTION = 0.5 * (math.sqrt(2.0) - 1.0)


@dataclass
class ReconstructionResult:
    """An estimate and the diagnostics recorded during an iterative method."""

    estimate: Tensor
    objective: list[float]
    data_term: list[float]
    regularization: list[float]
    data_residual: list[float]
    relative_gradient: list[float]
    projected_gradient: list[float]
    iterate_residual: list[float]
    step_size: list[float]
    operator_calls: list[int]
    elapsed_seconds: float
    iterations: int
    stop_reason: str
    initial_estimate: Tensor | None = None


def fresnel_numbers_to_distances(
    fresnel_numbers: Sequence[float], wavelength: float, pixel_size: float
) -> tuple[float, ...]:
    """Convert pixel-size Fresnel numbers to plane-wave distances.

    The convention is ``F = pixel_size**2 / (wavelength * distance)``.
    """

    if wavelength <= 0 or pixel_size <= 0:
        raise ValueError("wavelength and pixel_size must be positive")
    if not fresnel_numbers:
        raise ValueError("at least one Fresnel number is required")
    if any(float(number) <= 0 for number in fresnel_numbers):
        raise ValueError("all Fresnel numbers must be positive")
    return tuple(
        pixel_size**2 / (wavelength * float(number))
        for number in fresnel_numbers
    )


def make_fresnel_propagators(
    shape: tuple[int, int],
    fresnel_numbers: Sequence[float],
    *,
    wavelength: float,
    pixel_size: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.cfloat,
) -> tuple[FresnelPropagation, ...]:
    """Construct one DeepInv Fresnel propagator per measurement distance."""

    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {shape}")
    distances = fresnel_numbers_to_distances(
        fresnel_numbers, wavelength=wavelength, pixel_size=pixel_size
    )
    return tuple(
        FresnelPropagation(
            img_size=(1, height, width),
            wavelength=wavelength,
            distance=distance,
            pixel_size=pixel_size,
            dtype=dtype,
            device=device,
        )
        for distance in distances
    )


def _measurement_stack(measurements: Tensor | Sequence[Tensor]) -> Tensor:
    if isinstance(measurements, Tensor):
        stack = measurements
    else:
        if not measurements:
            raise ValueError("at least one measurement is required")
        stack = torch.stack(tuple(measurements), dim=0)
    if stack.ndim == 3:
        stack = stack[:, None, None]
    elif stack.ndim == 4:
        stack = stack[:, None]
    if stack.ndim != 5:
        raise ValueError(
            "measurements must have shape (J,H,W), (J,C,H,W), or (J,B,C,H,W); "
            f"got {tuple(stack.shape)}"
        )
    return stack


def _check_inputs(
    phase: Tensor,
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
) -> Tensor:
    stack = _measurement_stack(measurements)
    if phase.ndim != 4:
        raise ValueError(f"phase must have shape (B,C,H,W), got {tuple(phase.shape)}")
    if phase.is_complex():
        raise ValueError("phase must be real-valued")
    if len(propagators) != stack.shape[0]:
        raise ValueError(
            f"got {len(propagators)} propagators for {stack.shape[0]} measurements"
        )
    if tuple(stack.shape[1:]) != tuple(phase.shape):
        raise ValueError(
            f"measurement images {tuple(stack.shape[1:])} and phase "
            f"{tuple(phase.shape)} must match"
        )
    return stack.to(device=phase.device, dtype=phase.dtype)


def simulate_intensities(
    phase: Tensor, propagators: Sequence[FresnelPropagation]
) -> Tensor:
    """Evaluate ``|P_j exp(i phase)|**2`` for every distance."""

    if phase.ndim != 4 or phase.is_complex():
        raise ValueError("phase must be a real tensor with shape (B,C,H,W)")
    if not propagators:
        raise ValueError("at least one propagator is required")
    transmission = torch.exp(1j * phase)
    return torch.stack(
        tuple(propagator.A(transmission).abs().square() for propagator in propagators)
    )


def ctf_transfer_functions(
    propagators: Sequence[FresnelPropagation], *, dtype: torch.dtype | None = None
) -> Tensor:
    """Return the pure-phase linear CTF multipliers ``2 sin(theta_j)``.

    DeepInv stores the Fresnel transfer function as
    ``H_j = exp(-i theta_j)``.  Linearizing ``|P_j exp(i phi)|**2`` at zero
    therefore gives ``I_j - 1 = -2 Im(H_j) F(phi)``.
    """

    if not propagators:
        raise ValueError("at least one propagator is required")
    if dtype is None:
        dtype = propagators[0].H.real.dtype
    return torch.stack(tuple(-2.0 * prop.H.imag.to(dtype=dtype) for prop in propagators))


def _regularization_tensor(
    alpha: float | Tensor | None, reference: Tensor
) -> Tensor:
    if alpha is None:
        return torch.zeros(reference.shape[-2:], dtype=reference.dtype, device=reference.device)
    value = torch.as_tensor(alpha, dtype=reference.dtype, device=reference.device)
    if value.ndim > 2 or (
        value.ndim == 2 and tuple(value.shape) != tuple(reference.shape[-2:])
    ):
        raise ValueError(
            "alpha must be scalar or match the two spatial dimensions; "
            f"got {tuple(value.shape)}"
        )
    if torch.any(value < 0):
        raise ValueError("Tikhonov weights must be non-negative")
    return value


def ctf_reconstruct(
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    alpha: float | Tensor | None = None,
    rcond: float = 1e-6,
) -> Tensor:
    """Closed-form pure-phase CTF reconstruction.

    With ``h_j = 2 sin(theta_j)`` this computes

    ``F(phi) = sum_j h_j F(I_j - 1) / (alpha + sum_j h_j**2)``.

    If ``alpha`` is zero, modes below ``rcond`` times the largest denominator
    are set to zero.  This includes the unobservable constant phase mode and is
    therefore a Moore--Penrose inverse rather than division by an arbitrary
    epsilon.
    """

    stack = _measurement_stack(measurements)
    if len(propagators) != stack.shape[0]:
        raise ValueError("the number of measurements and propagators must match")
    if rcond < 0:
        raise ValueError("rcond must be non-negative")
    ctf = ctf_transfer_functions(propagators, dtype=stack.dtype).to(stack.device)
    contrast_spectrum = torch.fft.fft2(stack - 1.0, norm="ortho")
    numerator = (ctf[:, None, None] * contrast_spectrum).sum(dim=0)
    denominator = ctf.square().sum(dim=0)
    alpha_tensor = _regularization_tensor(alpha, stack)
    denominator = denominator + alpha_tensor

    cutoff = rcond * denominator.amax()
    resolved = denominator > cutoff
    safe_denominator = torch.where(resolved, denominator, torch.ones_like(denominator))
    spectrum = torch.where(
        resolved[None, None], numerator / safe_denominator[None, None], 0.0
    )
    return torch.fft.ifft2(spectrum, norm="ortho").real


def angular_frequency_radius(
    shape: tuple[int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the radial angular frequency in radians per pixel."""

    fy = 2.0 * math.pi * torch.fft.fftfreq(
        shape[0], d=1.0, device=device, dtype=dtype
    )
    fx = 2.0 * math.pi * torch.fft.fftfreq(
        shape[1], d=1.0, device=device, dtype=dtype
    )
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(grid_x.square() + grid_y.square())


def _smooth_step(radius: Tensor, cutoff: float, width: float) -> Tensor:
    if width <= 0:
        return (radius >= cutoff).to(radius.dtype)
    scale = math.sqrt(2.0) * width
    return 0.5 * (1.0 + torch.erf((radius - cutoff) / scale))


def huhn_regularization_filter(
    shape: tuple[int, int],
    fresnel_numbers: Sequence[float],
    *,
    alpha_low: float = 1e-3,
    alpha_high: float = 1e-1,
    alpha_beyond_na: float | None = None,
    transition_fraction: float = DEFAULT_TRANSITION_FRACTION,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Construct Huhn's frequency-dependent Tikhonov weights.

    The first transition is located at the first pure-phase CTF maximum,
    ``|xi| = pi sqrt(2 mean(F))`` (paper Eq. 5).  If
    ``alpha_beyond_na`` is supplied, a third level begins at
    ``|xi| = pi D mean(F)`` with ``D=min(shape)`` (paper Eq. 14).  Error-function
    transitions implement the paper's smooth, approximate steps.  The default
    first-transition width spans the first CTF maximum toward its next root,
    following the HoToPy implementation convention.
    """

    if not fresnel_numbers or any(float(value) <= 0 for value in fresnel_numbers):
        raise ValueError("fresnel_numbers must be a non-empty positive sequence")
    if min(alpha_low, alpha_high) < 0:
        raise ValueError("regularization weights must be non-negative")
    if alpha_beyond_na is not None and alpha_beyond_na < 0:
        raise ValueError("alpha_beyond_na must be non-negative")
    if transition_fraction < 0:
        raise ValueError("transition_fraction must be non-negative")

    radius = angular_frequency_radius(shape, device=device, dtype=dtype)
    mean_fresnel = sum(float(value) for value in fresnel_numbers) / len(
        fresnel_numbers
    )
    first_maximum = math.pi * math.sqrt(2.0 * mean_fresnel)
    width = transition_fraction * first_maximum
    alpha = alpha_low + (alpha_high - alpha_low) * _smooth_step(
        radius, first_maximum, width
    )

    if alpha_beyond_na is not None:
        na_cutoff = math.pi * min(shape) * mean_fresnel
        na_width = transition_fraction * max(na_cutoff, first_maximum)
        alpha = alpha + (alpha_beyond_na - alpha_high) * _smooth_step(
            radius, na_cutoff, na_width
        )
    return alpha


def project_phase(
    phase: Tensor,
    *,
    nonpositive: bool = False,
    support: Tensor | None = None,
) -> Tensor:
    """Project a phase image onto sign and support constraints."""

    projected = phase
    if support is not None:
        mask = support.to(device=phase.device, dtype=torch.bool)
        if tuple(mask.shape[-2:]) != tuple(phase.shape[-2:]):
            raise ValueError("support must match the phase's spatial shape")
        while mask.ndim < phase.ndim:
            mask = mask.unsqueeze(0)
        projected = torch.where(mask, projected, torch.zeros_like(projected))
    if nonpositive:
        projected = projected.clamp_max(0.0)
    return projected


class PhaseConstraintPrior(dinv.optim.Prior):
    """DeepInv prior whose proximal map is a phase-constraint projection.

    The function value is zero on the feasible iterates generated by its
    proximal map.  Keeping this as an explicit DeepInv prior lets
    :class:`deepinv.optim.PGD` report the data-fidelity objective.
    """

    def __init__(
        self, *, nonpositive: bool = False, support: Tensor | None = None
    ) -> None:
        super().__init__()
        self.nonpositive = nonpositive
        if support is None:
            self.register_buffer("support", None)
        else:
            self.register_buffer("support", support.to(dtype=torch.bool))
        self.explicit_prior = True

    def fn(self, x: Tensor, *args, **kwargs) -> Tensor:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def prox(
        self, x: Tensor, *args, gamma: float = 1.0, **kwargs
    ) -> Tensor:
        del gamma
        return project_phase(
            x, nonpositive=self.nonpositive, support=self.support
        )


class FourierRegularizedDataFidelity(dinv.optim.DataFidelity):
    """DeepInv stacked L2 fidelity plus Fourier-diagonal Tikhonov penalty.

    A factor of one half is applied to both the squared data residual and the
    Fourier penalty.  This scales Huhn's complete objective by one half and
    therefore leaves its minimizer unchanged.
    """

    def __init__(
        self,
        base: dinv.optim.StackedPhysicsDataFidelity,
        alpha: float | Tensor | None,
        *,
        data_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if data_scale <= 0:
            raise ValueError("data_scale must be positive")
        self.base = base
        self.data_scale = float(data_scale)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha_tensor = torch.as_tensor(alpha)
            if torch.any(alpha_tensor < 0):
                raise ValueError("Tikhonov weights must be non-negative")
            self.register_buffer("alpha", alpha_tensor)

    def regularization(self, x: Tensor) -> Tensor:
        if self.alpha is None:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        spectrum = torch.fft.fft2(x, norm="ortho")
        return 0.5 * (
            alpha * spectrum.abs().square()
        ).sum(dim=tuple(range(1, x.ndim)))

    def fn(self, x: Tensor, y, physics, *args, **kwargs) -> Tensor:
        return self.data_term(x, y, physics, *args, **kwargs) + self.regularization(x)

    def data_term(self, x: Tensor, y, physics, *args, **kwargs) -> Tensor:
        """Return the scaled stacked DeepInv L2 data term."""

        return self.data_scale * self.base.fn(x, y, physics, *args, **kwargs)

    def grad(self, x: Tensor, y, physics, *args, **kwargs) -> Tensor:
        gradient = self.data_scale * self.base.grad(
            x, y, physics, *args, **kwargs
        )
        if self.alpha is None:
            return gradient
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        spectrum = torch.fft.fft2(x, norm="ortho")
        return gradient + torch.fft.ifft2(alpha * spectrum, norm="ortho").real


def make_deepinv_phase_problem(
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    alpha: float | Tensor | None = None,
    reduction: str = "sum",
):
    """Build the nonlinear model from DeepInv physics and fidelity objects."""

    stack = _measurement_stack(measurements)
    if len(propagators) != stack.shape[0]:
        raise ValueError("the number of measurements and propagators must match")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")
    plane_physics = []
    for propagator in propagators:
        transmission = dinv.physics.Physics(
            A=lambda phase, **_: torch.exp(1j * phase)
        )
        intensity = dinv.physics.PhaseRetrieval(B=propagator)
        plane_physics.append(dinv.physics.compose(transmission, intensity))
    physics = dinv.physics.stack(*plane_physics)
    observations = dinv.utils.TensorList(
        [measurement for measurement in stack.unbind(dim=0)]
    )
    base_fidelity = dinv.optim.StackedPhysicsDataFidelity(
        [dinv.optim.L2() for _ in propagators]
    )
    data_scale = 1.0 if reduction == "sum" else 1.0 / len(propagators)
    fidelity = FourierRegularizedDataFidelity(
        base_fidelity, alpha, data_scale=data_scale
    )
    return observations, physics, fidelity


def constrained_ctf_reconstruct(
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    alpha: float | Tensor | None,
    nonpositive: bool = False,
    support: Tensor | None = None,
    rho: float = 1e-2,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    acceleration: bool = True,
    restart_eta: float = 0.999,
) -> Tensor:
    """Constrained CTF reconstruction by accelerated scaled ADMM.

    This routine is used for the constraint-consistent CTF warm start of the
    nonlinear method.  Without constraints it dispatches to the closed-form
    CTF inverse.  The default Nesterov acceleration with residual restart is
    the fast-ADMM variant used for paper Eq. 8; setting ``acceleration=False``
    recovers ordinary scaled ADMM.
    """

    if not nonpositive and support is None:
        return ctf_reconstruct(measurements, propagators, alpha=alpha)
    if rho <= 0:
        raise ValueError("rho must be positive")
    if max_iter < 0 or tolerance < 0:
        raise ValueError("max_iter and tolerance must be non-negative")
    if not 0 < restart_eta < 1:
        raise ValueError("restart_eta must lie in (0, 1)")
    stack = _measurement_stack(measurements)
    if len(propagators) != stack.shape[0]:
        raise ValueError("the number of measurements and propagators must match")
    ctf = ctf_transfer_functions(propagators, dtype=stack.dtype).to(stack.device)
    contrast_spectrum = torch.fft.fft2(stack - 1.0, norm="ortho")
    data_numerator = (ctf[:, None, None] * contrast_spectrum).sum(dim=0)
    denominator = ctf.square().sum(dim=0) + _regularization_tensor(alpha, stack)

    shape = tuple(stack.shape[1:])
    primal = torch.zeros(shape, dtype=stack.dtype, device=stack.device)
    accepted_auxiliary = primal.clone()
    accepted_dual = primal.clone()
    accelerated_auxiliary = primal.clone()
    accelerated_dual = primal.clone()
    acceleration_factor = 1.0
    restart_measure: Tensor | None = None
    epsilon = torch.finfo(stack.dtype).eps

    for iteration in range(max_iter):
        rhs = data_numerator + rho * torch.fft.fft2(
            accelerated_auxiliary - accelerated_dual, norm="ortho"
        )
        primal = torch.fft.ifft2(
            rhs / (denominator[None, None] + rho), norm="ortho"
        ).real
        auxiliary = project_phase(
            primal + accelerated_dual,
            nonpositive=nonpositive,
            support=support,
        )
        dual = accelerated_dual + primal - auxiliary

        raw_primal_residual = torch.linalg.vector_norm(primal - auxiliary)
        raw_dual_residual = torch.linalg.vector_norm(
            auxiliary - accelerated_auxiliary
        )
        primal_scale = torch.maximum(
            torch.linalg.vector_norm(primal),
            torch.linalg.vector_norm(auxiliary),
        ).clamp_min(epsilon)
        dual_scale = torch.maximum(
            torch.linalg.vector_norm(accelerated_auxiliary),
            torch.linalg.vector_norm(auxiliary),
        ).clamp_min(epsilon)
        primal_residual = raw_primal_residual / primal_scale
        dual_residual = raw_dual_residual / dual_scale
        if max(float(primal_residual), float(dual_residual)) <= tolerance:
            accepted_auxiliary = auxiliary
            break

        if not acceleration:
            accepted_auxiliary = auxiliary
            accepted_dual = dual
            accelerated_auxiliary = auxiliary
            accelerated_dual = dual
            continue

        current_measure = rho * (
            raw_primal_residual.square() + raw_dual_residual.square()
        )
        if iteration == 0 or current_measure < restart_eta * restart_measure:
            next_factor = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * acceleration_factor**2)
            )
            momentum = (acceleration_factor - 1.0) / next_factor
            previous_auxiliary = accepted_auxiliary
            previous_dual = accepted_dual
            accepted_auxiliary = auxiliary
            accepted_dual = dual
            accelerated_auxiliary = auxiliary + momentum * (
                auxiliary - previous_auxiliary
            )
            accelerated_dual = dual + momentum * (dual - previous_dual)
            acceleration_factor = next_factor
            restart_measure = current_measure
        else:
            acceleration_factor = 1.0
            restart_measure = restart_measure / restart_eta
            accelerated_auxiliary = accepted_auxiliary
            accelerated_dual = accepted_dual
    return accepted_auxiliary


def _regularization_value_gradient(
    phase: Tensor, alpha: float | Tensor | None
) -> tuple[Tensor, Tensor]:
    alpha_tensor = _regularization_tensor(alpha, phase)
    if not torch.any(alpha_tensor):
        zero = phase.new_zeros(())
        return zero, torch.zeros_like(phase)
    spectrum = torch.fft.fft2(phase, norm="ortho")
    value = (alpha_tensor * spectrum.abs().square()).sum()
    gradient = 2.0 * torch.fft.ifft2(
        alpha_tensor * spectrum, norm="ortho"
    ).real
    return value, gradient


def nonlinear_value_and_gradient(
    phase: Tensor,
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    alpha: float | Tensor | None = None,
    reduction: str = "sum",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate Huhn's nonlinear objective and its analytic gradient.

    Returns ``(objective, data_term, regularization, gradient)``.  ``reduction``
    may be ``"sum"`` (the paper's convention) or ``"mean"`` over distances.
    """

    stack = _check_inputs(phase, measurements, propagators)
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

    transmission = torch.exp(1j * phase)
    data_term = phase.new_zeros(())
    gradient = torch.zeros_like(phase)
    gamma_transmission_conjugate = torch.conj(1j * transmission)

    for measurement, propagator in zip(stack, propagators, strict=True):
        detector_field = propagator.A(transmission)
        residual = detector_field.abs().square() - measurement
        data_term = data_term + residual.square().sum()
        backpropagated = propagator.A_adjoint(detector_field * residual)
        gradient = gradient + 4.0 * torch.real(
            gamma_transmission_conjugate * backpropagated
        )

    if reduction == "mean":
        data_term = data_term / len(propagators)
        gradient = gradient / len(propagators)
    regularization, regularization_gradient = _regularization_value_gradient(
        phase, alpha
    )
    gradient = gradient + regularization_gradient
    return data_term + regularization, data_term, regularization, gradient


def _projected_gradient_norm(
    phase: Tensor,
    gradient: Tensor,
    *,
    nonpositive: bool,
    support: Tensor | None,
) -> Tensor:
    mapping = phase - project_phase(
        phase - gradient, nonpositive=nonpositive, support=support
    )
    return torch.linalg.vector_norm(mapping)


def projected_gradient_descent(
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    initial_phase: Tensor,
    alpha: float | Tensor | None = None,
    nonpositive: bool = False,
    support: Tensor | None = None,
    step_size: float = 0.05,
    max_iter: int = 50,
    tolerance: float = 0.0,
    reduction: str = "sum",
) -> ReconstructionResult:
    """Run fixed-step projected gradient descent with DeepInv's PGD solver.

    The nonlinear forward maps, stacked L2 fidelity, automatic VJPs, iteration
    loop, and convergence check are provided by DeepInv.  The only local prior
    is the proximal projection onto the requested support/sign constraints.
    ``reduction="mean"`` normalizes the data term by the number of planes,
    which is useful when measurement diversity is the experimental variable.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

    stack = _check_inputs(initial_phase, measurements, propagators)
    started = time.perf_counter()
    phase = project_phase(
        initial_phase.detach().clone(), nonpositive=nonpositive, support=support
    )
    observations, physics, fidelity = make_deepinv_phase_problem(
        stack, propagators, alpha=alpha, reduction=reduction
    )
    prior = PhaseConstraintPrior(
        nonpositive=nonpositive, support=support
    ).to(phase.device)

    with torch.no_grad():
        initial_objective = fidelity.fn(phase, observations, physics).sum()
        initial_regularization = fidelity.regularization(phase).sum()
    contrast_norm = torch.linalg.vector_norm(stack - 1.0).clamp_min(
        torch.finfo(phase.dtype).eps
    )

    def regularization_metric(_history, _previous: Tensor, current: Tensor) -> float:
        with torch.no_grad():
            value = fidelity.regularization(current.unsqueeze(0))[0]
        return float(value)

    solver = dinv.optim.PGD(
        data_fidelity=fidelity,
        prior=prior,
        lambda_reg=1.0,
        stepsize=step_size,
        max_iter=max_iter,
        crit_conv="residual",
        thres_conv=tolerance,
        early_stop=tolerance > 0,
        custom_metrics={"regularization": regularization_metric},
        verbose=False,
        show_progress_bar=False,
    )
    phase, metrics = solver(
        observations, physics, init=phase, compute_metrics=True
    )

    batch_size = phase.shape[0]
    iterations = len(metrics["residual"][0]) if batch_size else 0

    def sum_metric(name: str, index: int) -> float:
        return float(sum(metrics[name][batch][index] for batch in range(batch_size)))

    objectives = [float(initial_objective)] + [
        sum_metric("cost", index) for index in range(iterations)
    ]
    regularizers = [float(initial_regularization)] + [
        sum_metric("regularization", index) for index in range(iterations)
    ]
    data_terms = [
        objective - regularizer
        for objective, regularizer in zip(objectives, regularizers, strict=True)
    ]
    data_residuals = [
        math.sqrt(max(0.0, 2.0 * value / fidelity.data_scale))
        / float(contrast_norm)
        for value in data_terms
    ]
    iterate_residuals = [float("nan")] + [
        float(
            sum(metrics["residual"][batch][index] for batch in range(batch_size))
            / batch_size
        )
        for index in range(iterations)
    ]

    # DeepInv evaluates a cost after every PGD step.  A nonlinear gradient
    # evaluates the propagation twice plus one reverse propagation per plane;
    # the following is therefore a forward/adjoint-equivalent work count.
    planes = len(propagators)
    operator_calls = [planes + 4 * planes * index for index in range(iterations + 1)]
    stop_reason = (
        "DeepInv iterate-residual tolerance reached"
        if solver.has_converged
        else "maximum iterations reached"
    )

    elapsed = time.perf_counter() - started
    return ReconstructionResult(
        estimate=phase.detach(),
        objective=objectives,
        data_term=data_terms,
        regularization=regularizers,
        data_residual=data_residuals,
        relative_gradient=[],
        projected_gradient=[],
        iterate_residual=iterate_residuals,
        step_size=[step_size] * len(objectives),
        operator_calls=operator_calls,
        elapsed_seconds=elapsed,
        iterations=iterations,
        stop_reason=stop_reason,
        initial_estimate=project_phase(
            initial_phase.detach().clone(),
            nonpositive=nonpositive,
            support=support,
        ),
    )


def alternating_projections(
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    initial_phase: Tensor,
    nonpositive: bool = False,
    support: Tensor | None = None,
    max_iter: int = 50,
    tolerance: float = 0.0,
    epsilon: float = 1e-8,
) -> ReconstructionResult:
    """Averaged multi-distance alternating projections.

    Each measurement projection replaces the propagated amplitude by
    ``sqrt(I_j)`` while retaining the current detector-plane phase.  The
    back-propagated waves are averaged, as in Hagemann et al. (2018), before
    the pure-phase, sign, and optional support constraints are imposed.

    The recorded objective is the common *intensity* least-squares diagnostic;
    AP itself is an amplitude-projection method and does not minimize that
    objective by gradient descent.
    """

    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    stack = _check_inputs(initial_phase, measurements, propagators)
    started = time.perf_counter()
    phase = project_phase(
        initial_phase.detach().clone(), nonpositive=nonpositive, support=support
    )
    wave = torch.exp(1j * phase)

    def diagnostic(current_phase: Tensor) -> tuple[float, float]:
        predictions = simulate_intensities(current_phase, propagators)
        residual = predictions - stack
        value = residual.square().sum()
        contrast_norm = torch.linalg.vector_norm(stack - 1.0).clamp_min(epsilon)
        relative = torch.linalg.vector_norm(residual) / contrast_norm
        return float(value), float(relative)

    initial_objective, initial_residual = diagnostic(phase)
    calls = len(propagators)
    objectives = [initial_objective]
    residuals = [initial_residual]
    iterate_residuals = [float("nan")]
    operator_calls = [calls]
    stop_reason = "maximum iterations reached"

    for _ in range(max_iter):
        previous_phase = phase
        backpropagated = []
        for measurement, propagator in zip(stack, propagators, strict=True):
            detector_field = propagator.A(wave)
            detector_phase = detector_field / detector_field.abs().clamp_min(epsilon)
            projected_field = measurement.clamp_min(0.0).sqrt() * detector_phase
            backpropagated.append(propagator.A_adjoint(projected_field))
        calls += 2 * len(propagators)
        averaged_wave = torch.stack(backpropagated).mean(dim=0)
        phase = torch.angle(averaged_wave)
        phase = project_phase(phase, nonpositive=nonpositive, support=support)
        wave = torch.exp(1j * phase)
        objective, relative_residual = diagnostic(phase)
        calls += len(propagators)
        objectives.append(objective)
        residuals.append(relative_residual)
        iterate_residuals.append(
            float(
                torch.linalg.vector_norm(phase - previous_phase)
                / torch.linalg.vector_norm(phase).clamp_min(epsilon)
            )
        )
        operator_calls.append(calls)
        if tolerance > 0 and relative_residual <= tolerance:
            stop_reason = "data-residual tolerance reached"
            break

    elapsed = time.perf_counter() - started
    return ReconstructionResult(
        estimate=phase.detach(),
        objective=objectives,
        data_term=objectives.copy(),
        regularization=[0.0] * len(objectives),
        data_residual=residuals,
        relative_gradient=[],
        projected_gradient=[],
        iterate_residual=iterate_residuals,
        step_size=[],
        operator_calls=operator_calls,
        elapsed_seconds=elapsed,
        iterations=len(objectives) - 1,
        stop_reason=stop_reason,
        initial_estimate=initial_phase.detach().clone(),
    )


def huhn_nltikh(
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    fresnel_numbers: Sequence[float],
    *,
    initial_phase: Tensor | None = None,
    nonpositive: bool = True,
    support: Tensor | None = None,
    alpha_low: float = 1e-3,
    alpha_high: float = 1e-1,
    alpha_beyond_na: float | None = None,
    max_iter: int = 100,
    tolerance: float = 1e-3,
    initial_step: float | None = None,
    min_step: float = 1e-8,
    max_step: float = 1.0,
    line_search_memory: int = 2,
    line_search_contraction: float = 0.25,
    max_line_search_steps: int = 5,
    ctf_init_iterations: int = 100,
) -> ReconstructionResult:
    """Huhn-style nonlinear Tikhonov reconstruction.

    The method combines the ingredients in Huhn et al. (2022): multi-distance
    nonlinear intensity fitting, a constraint-consistent CTF warm start,
    frequency-dependent Tikhonov weights, projected gradient steps, alternating
    Barzilai--Borwein step lengths (paper Eq. 12), a non-monotone proximal
    backtracking test, and the relative raw-gradient stopping rule from paper
    Eq. 13.

    The default third regularization level is ``2 * J`` as suggested in Eq. 14.
    Use :func:`huhn_regularization_filter` directly when a two-level filter is
    desired instead.
    """

    stack = _measurement_stack(measurements)
    if len(propagators) != stack.shape[0] or len(fresnel_numbers) != stack.shape[0]:
        raise ValueError("measurements, propagators, and Fresnel numbers must match")
    if max_iter < 0 or ctf_init_iterations < 0:
        raise ValueError("iteration counts must be non-negative")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if not 0 < line_search_contraction < 1:
        raise ValueError("line_search_contraction must lie in (0, 1)")
    if max_line_search_steps < 1:
        raise ValueError("max_line_search_steps must be at least one")
    if line_search_memory < 1:
        raise ValueError("line_search_memory must be at least one")
    if min_step <= 0 or max_step < min_step:
        raise ValueError("step bounds are invalid")

    if alpha_beyond_na is None:
        alpha_beyond_na = 2.0 * len(propagators)
    alpha = huhn_regularization_filter(
        tuple(stack.shape[-2:]),
        fresnel_numbers,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        alpha_beyond_na=alpha_beyond_na,
        device=stack.device,
        dtype=stack.dtype,
    )
    if initial_phase is None:
        warm_start = constrained_ctf_reconstruct(
            stack,
            propagators,
            alpha=alpha,
            nonpositive=nonpositive,
            support=support,
            max_iter=ctf_init_iterations,
        )
    else:
        warm_start = project_phase(
            initial_phase.detach().clone(),
            nonpositive=nonpositive,
            support=support,
        )

    started = time.perf_counter()
    phase = warm_start.detach().clone()
    observations, physics, fidelity = make_deepinv_phase_problem(
        stack, propagators, alpha=alpha, reduction="sum"
    )
    constraint_prior = PhaseConstraintPrior(
        nonpositive=nonpositive, support=support
    ).to(phase.device)

    def value_components(current: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        objective_value = fidelity.fn(current, observations, physics).sum()
        regularization_value = fidelity.regularization(current).sum()
        return (
            objective_value,
            objective_value - regularization_value,
            regularization_value,
        )

    with torch.no_grad():
        objective, data, regularization = value_components(phase)
        gradient = fidelity.grad(phase, observations, physics)
        zero_gradient = fidelity.grad(
            torch.zeros_like(phase), observations, physics
        )

    # One value evaluation costs J forward propagations.  DeepInv's automatic
    # nonlinear VJP evaluates the forward map twice and performs one reverse
    # propagation, giving three forward/adjoint-equivalent calls per gradient.
    calls = 7 * len(propagators)
    reference_gradient_norm = torch.linalg.vector_norm(zero_gradient).clamp_min(
        torch.finfo(phase.dtype).eps
    )
    if initial_step is None:
        initial_step = 1.0 / (
            4.0 * len(propagators) + float(alpha.amax())
        )
    elif initial_step <= 0:
        raise ValueError("initial_step must be positive")
    step = float(min(max(initial_step, min_step), max_step))
    contrast_norm = torch.linalg.vector_norm(stack - 1.0).clamp_min(
        torch.finfo(phase.dtype).eps
    )

    objectives = [float(objective)]
    data_terms = [float(data)]
    regularizers = [float(regularization)]
    data_residuals = [
        math.sqrt(max(0.0, 2.0 * float(data))) / float(contrast_norm)
    ]
    relative_gradients = [
        float(torch.linalg.vector_norm(gradient) / reference_gradient_norm)
    ]
    projected_gradients = [
        float(
            _projected_gradient_norm(
                phase,
                gradient,
                nonpositive=nonpositive,
                support=support,
            )
            / reference_gradient_norm
        )
    ]
    iterate_residuals = [float("nan")]
    steps = [step]
    operator_calls = [calls]
    stop_reason = "maximum iterations reached"

    for iteration in range(1, max_iter + 1):
        reference_value = max(objectives[-line_search_memory:])
        accepted = False
        trial_phase = phase
        trial_objective = objective
        trial_data = data
        trial_regularization = regularization

        for _ in range(max_line_search_steps):
            candidate = constraint_prior.prox(phase - step * gradient)
            with torch.no_grad():
                candidate_objective, candidate_data, candidate_regularization = (
                    value_components(candidate)
                )
            calls += len(propagators)
            direction = candidate - phase
            sufficient_decrease = (
                reference_value
                + float(torch.sum(gradient * direction))
                + 0.5 / step * float(torch.sum(direction.square()))
            )
            trial_phase = candidate
            trial_objective = candidate_objective
            trial_data = candidate_data
            trial_regularization = candidate_regularization
            if float(candidate_objective) <= sufficient_decrease:
                accepted = True
                break
            step = max(step * line_search_contraction, min_step)

        if not accepted:
            stop_reason = "line search failed"
            break

        previous_phase = phase
        previous_gradient = gradient
        accepted_step = step
        phase = trial_phase
        objective = trial_objective
        data = trial_data
        regularization = trial_regularization
        with torch.no_grad():
            gradient = fidelity.grad(phase, observations, physics)
        calls += 3 * len(propagators)

        displacement = phase - previous_phase
        gradient_change = gradient - previous_gradient
        sy = float(torch.sum(displacement * gradient_change))
        ss = float(torch.sum(displacement.square()))
        yy = float(torch.sum(gradient_change.square()))
        if sy > 0 and ss > 0 and yy > 0:
            if iteration % 2 == 1:
                proposed_step = sy / yy
            else:
                proposed_step = ss / sy
            if math.isfinite(proposed_step):
                step = min(max(proposed_step, min_step), max_step)

        relative_gradient = torch.linalg.vector_norm(gradient) / reference_gradient_norm
        projected_gradient = _projected_gradient_norm(
            phase, gradient, nonpositive=nonpositive, support=support
        ) / reference_gradient_norm
        objectives.append(float(objective))
        data_terms.append(float(data))
        regularizers.append(float(regularization))
        data_residuals.append(
            math.sqrt(max(0.0, 2.0 * float(data))) / float(contrast_norm)
        )
        relative_gradients.append(float(relative_gradient))
        projected_gradients.append(float(projected_gradient))
        iterate_residuals.append(
            float(
                torch.linalg.vector_norm(displacement)
                / torch.linalg.vector_norm(phase).clamp_min(
                    torch.finfo(phase.dtype).eps
                )
            )
        )
        steps.append(accepted_step)
        operator_calls.append(calls)
        if float(relative_gradient) <= tolerance:
            stop_reason = "relative-gradient tolerance reached"
            break

    elapsed = time.perf_counter() - started
    return ReconstructionResult(
        estimate=phase.detach(),
        objective=objectives,
        data_term=data_terms,
        regularization=regularizers,
        data_residual=data_residuals,
        relative_gradient=relative_gradients,
        projected_gradient=projected_gradients,
        iterate_residual=iterate_residuals,
        step_size=steps,
        operator_calls=operator_calls,
        elapsed_seconds=elapsed,
        iterations=len(objectives) - 1,
        stop_reason=stop_reason,
        initial_estimate=warm_start.detach(),
    )


def relative_data_residual(
    phase: Tensor,
    measurements: Tensor | Sequence[Tensor],
    propagators: Sequence[FresnelPropagation],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return ``||A(phi)-I|| / ||I-1||`` for comparable data consistency."""

    stack = _check_inputs(phase, measurements, propagators)
    prediction = simulate_intensities(phase, propagators)
    numerator = torch.linalg.vector_norm(prediction - stack)
    denominator = torch.linalg.vector_norm(stack - 1.0).clamp_min(epsilon)
    return float(numerator / denominator)


def support_referenced_phase_nrmse(
    estimate: Tensor,
    truth: Tensor,
    support: Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Synthetic-reference NRMSE after fixing the constant-phase gauge.

    The estimate is background-referenced using the known complement of the
    support, then the relative error is evaluated inside the support.
    """

    if tuple(estimate.shape) != tuple(truth.shape):
        raise ValueError("estimate and truth must have the same shape")
    mask = support.to(device=estimate.device, dtype=torch.bool)
    while mask.ndim < estimate.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(estimate)
    background = ~mask
    if not torch.any(mask) or not torch.any(background):
        raise ValueError("support must contain both foreground and background")
    referenced = estimate - estimate[background].mean()
    numerator = torch.linalg.vector_norm(referenced[mask] - truth[mask])
    denominator = torch.linalg.vector_norm(truth[mask]).clamp_min(epsilon)
    return float(numerator / denominator)
