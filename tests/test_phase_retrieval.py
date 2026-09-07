"""Numerical checks for the phase-retrieval comparison building blocks."""

from __future__ import annotations

import math
import unittest

import torch

import deepinv as dinv
from multi_distance_phase_retrival_huhn.phase_retrieval import (
    PhaseConstraintPrior,
    alternating_projections,
    angular_frequency_radius,
    constrained_ctf_reconstruct,
    ctf_reconstruct,
    ctf_transfer_functions,
    huhn_nltikh,
    huhn_regularization_filter,
    make_deepinv_phase_problem,
    make_fresnel_propagators,
    nonlinear_value_and_gradient,
    project_phase,
    projected_gradient_descent,
    simulate_intensities,
)


WAVELENGTH = 1.5498e-10
PIXEL_SIZE = 196e-9
FRESNEL_NUMBERS = (1.59e-3, 1.49e-3)


def make_problem(
    shape: tuple[int, int] = (20, 18), *, dtype: torch.dtype = torch.float32
):
    complex_dtype = torch.cdouble if dtype == torch.float64 else torch.cfloat
    propagators = make_fresnel_propagators(
        shape,
        FRESNEL_NUMBERS,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        dtype=complex_dtype,
    )
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, shape[0], dtype=dtype),
        torch.linspace(-1.0, 1.0, shape[1], dtype=dtype),
        indexing="ij",
    )
    support = (x.square() + y.square()) < 0.72**2
    sphere = torch.sqrt(
        (1.0 - ((x + 0.12) / 0.43).square() - ((y - 0.08) / 0.38).square())
        .clamp_min(0.0)
    )
    truth = (-0.35 * sphere * support)[None, None]
    measurements = simulate_intensities(truth, propagators)
    return truth, support, measurements, propagators


class TestCTF(unittest.TestCase):
    def test_ctf_is_the_derivative_of_the_deepinv_forward_model(self):
        shape = (18, 16)
        truth, _, _, propagators = make_problem(shape, dtype=torch.float64)
        direction = truth - truth.mean()
        epsilon = 1e-5
        finite_difference = (
            simulate_intensities(epsilon * direction, propagators) - 1.0
        ) / epsilon
        transfer = ctf_transfer_functions(propagators, dtype=direction.dtype)
        predicted = torch.fft.ifft2(
            transfer[:, None, None]
            * torch.fft.fft2(direction, norm="ortho"),
            norm="ortho",
        ).real
        relative_error = torch.linalg.vector_norm(finite_difference - predicted) / (
            torch.linalg.vector_norm(predicted).clamp_min(1e-15)
        )
        self.assertLess(float(relative_error), 2e-5)

    def test_ctf_inverse_fits_linear_multi_distance_data(self):
        shape = (24, 22)
        truth, _, _, propagators = make_problem(shape, dtype=torch.float64)
        truth = truth - truth.mean()
        transfer = ctf_transfer_functions(propagators, dtype=truth.dtype)
        truth_spectrum = torch.fft.fft2(truth, norm="ortho")
        measurements = 1.0 + torch.fft.ifft2(
            transfer[:, None, None] * truth_spectrum, norm="ortho"
        ).real
        estimate = ctf_reconstruct(
            measurements, propagators, alpha=0.0, rcond=1e-10
        )
        fitted = torch.fft.ifft2(
            transfer[:, None, None]
            * torch.fft.fft2(estimate, norm="ortho"),
            norm="ortho",
        ).real
        residual = torch.linalg.vector_norm(fitted - (measurements - 1.0))
        scale = torch.linalg.vector_norm(measurements - 1.0)
        self.assertLess(float(residual / scale), 1e-8)

    def test_accelerated_constrained_ctf_is_feasible_and_improves_fit(self):
        truth, support, _, propagators = make_problem((20, 20))
        transfer = ctf_transfer_functions(propagators, dtype=truth.dtype)
        measurements = 1.0 + torch.fft.ifft2(
            transfer[:, None, None]
            * torch.fft.fft2(truth, norm="ortho"),
            norm="ortho",
        ).real
        estimate = constrained_ctf_reconstruct(
            measurements,
            propagators,
            alpha=1e-3,
            nonpositive=True,
            support=support,
            max_iter=40,
            tolerance=1e-5,
        )
        fitted = torch.fft.ifft2(
            transfer[:, None, None]
            * torch.fft.fft2(estimate, norm="ortho"),
            norm="ortho",
        ).real
        self.assertLessEqual(float(estimate.max()), 0.0)
        self.assertEqual(float(estimate[..., ~support].abs().max()), 0.0)
        self.assertLess(
            float(torch.linalg.vector_norm(fitted - (measurements - 1.0))),
            float(torch.linalg.vector_norm(measurements - 1.0)),
        )


class TestDeepInvIntegration(unittest.TestCase):
    def test_deepinv_objective_and_vjp_match_reference_gradient(self):
        truth, _, measurements, propagators = make_problem(
            (14, 12), dtype=torch.float64
        )
        torch.manual_seed(2)
        phase = truth + 0.03 * torch.randn_like(truth)
        alpha = 0.025
        objective, _, _, gradient = nonlinear_value_and_gradient(
            phase, measurements, propagators, alpha=alpha
        )
        observations, physics, fidelity = make_deepinv_phase_problem(
            measurements, propagators, alpha=alpha
        )
        deepinv_objective = fidelity.fn(phase, observations, physics).sum()
        deepinv_gradient = fidelity.grad(phase, observations, physics)
        self.assertTrue(
            torch.allclose(deepinv_objective, 0.5 * objective, rtol=1e-9, atol=1e-9)
        )
        self.assertTrue(
            torch.allclose(deepinv_gradient, 0.5 * gradient, rtol=1e-8, atol=1e-9)
        )
        self.assertIsInstance(physics, dinv.physics.StackedPhysics)

    def test_constraint_prior_is_a_deepinv_prior(self):
        x = torch.tensor([[[[-1.0, 2.0], [-3.0, 4.0]]]])
        support = torch.tensor([[True, True], [False, False]])
        prior = PhaseConstraintPrior(nonpositive=True, support=support)
        expected = torch.tensor([[[[-1.0, 0.0], [0.0, 0.0]]]])
        self.assertIsInstance(prior, dinv.optim.Prior)
        self.assertTrue(torch.equal(prior.prox(x), expected))
        self.assertTrue(torch.equal(project_phase(x, nonpositive=True, support=support), expected))

    def test_deepinv_pgd_decreases_objective_and_enforces_constraints(self):
        truth, support, measurements, propagators = make_problem()
        result = projected_gradient_descent(
            measurements,
            propagators,
            initial_phase=torch.zeros_like(truth),
            nonpositive=True,
            support=support,
            step_size=0.12,
            max_iter=6,
            reduction="mean",
        )
        self.assertEqual(result.iterations, 6)
        self.assertLess(result.objective[-1], result.objective[0])
        self.assertLess(result.data_residual[-1], result.data_residual[0])
        self.assertLessEqual(float(result.estimate.max()), 0.0)
        self.assertEqual(float(result.estimate[..., ~support].abs().max()), 0.0)


class TestHuhnComponents(unittest.TestCase):
    def test_three_level_regularization_filter(self):
        shape = (64, 64)
        alpha = huhn_regularization_filter(
            shape,
            FRESNEL_NUMBERS,
            alpha_low=1e-3,
            alpha_high=1e-1,
            alpha_beyond_na=4.0,
            transition_fraction=0.02,
            dtype=torch.float64,
        )
        radius = angular_frequency_radius(shape, dtype=torch.float64)
        mean_fresnel = sum(FRESNEL_NUMBERS) / len(FRESNEL_NUMBERS)
        first_cutoff = math.pi * math.sqrt(2.0 * mean_fresnel)
        na_cutoff = math.pi * min(shape) * mean_fresnel
        middle_radius = 0.5 * (first_cutoff + na_cutoff)
        middle_index = torch.argmin((radius - middle_radius).abs())
        self.assertAlmostEqual(float(alpha[0, 0]), 1e-3, places=6)
        self.assertAlmostEqual(float(alpha.flatten()[middle_index]), 1e-1, places=2)
        self.assertAlmostEqual(float(alpha.max()), 4.0, places=5)

    def test_ap_and_huhn_smoke(self):
        truth, support, measurements, propagators = make_problem((16, 16))
        zero = torch.zeros_like(truth)
        ap = alternating_projections(
            measurements,
            propagators,
            initial_phase=zero,
            nonpositive=True,
            support=support,
            max_iter=3,
        )
        self.assertLess(ap.data_residual[-1], ap.data_residual[0])
        full = huhn_nltikh(
            measurements,
            propagators,
            FRESNEL_NUMBERS,
            nonpositive=True,
            support=support,
            max_iter=3,
            ctf_init_iterations=4,
        )
        self.assertEqual(full.iterations, 3)
        self.assertTrue(math.isfinite(full.objective[-1]))
        self.assertLessEqual(float(full.estimate.max()), 0.0)
        self.assertEqual(len(full.objective), len(full.operator_calls))
        self.assertEqual(len(full.objective), len(full.data_residual))


if __name__ == "__main__":
    unittest.main()
