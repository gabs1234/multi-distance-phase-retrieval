"""Regressions for phase branches, direct inverses, scales, and provenance."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from multi_distance_phase_retrival_huhn.comparison_plots import (
    GROUPS, NONLINEAR_METHODS, nonlinear_phase_limits, phase_comparison_figure,
)
from multi_distance_phase_retrival_huhn.direct_methods import (
    fresnel_phase, homogeneous_ctf_ict, polystyrene_8kev, tie_reconstruct,
)
from multi_distance_phase_retrival_huhn.experiment_io import (
    RUNS, import_checkpoints, load_results, sha256, validate_checkpoint,
)
from multi_distance_phase_retrival_huhn.phase_retrieval import (
    alternating_projections, constrained_ctf_reconstruct, make_fresnel_propagators,
    simulate_intensities,
)


class TestDirectMethods(unittest.TestCase):
    def test_tie_closes_linear_model_and_has_zero_dc(self):
        shape = (24, 22)
        chi = fresnel_phase(shape, 0.5)
        observed = 1 + 0.05 * np.cos(2*np.pi*np.arange(shape[1])/shape[1])[None, :]
        observed = np.broadcast_to(observed, shape)
        phase = tie_reconstruct(observed, 0.5, alpha=0)
        fitted = np.fft.ifft2(2*chi*np.fft.fft2(phase)).real
        np.testing.assert_allclose(fitted, observed - 1, atol=1e-13)
        self.assertAlmostEqual(float(phase.mean()), 0, places=13)

    def test_empty_beam_and_uniform_absorber(self):
        empty = np.ones((12, 10))
        for alpha in (0, 0.01):
            ctf, ict, contact = homogeneous_ctf_ict(empty, 0.01, gamma=700, alpha=alpha)
            np.testing.assert_array_equal(ctf, 0)
            np.testing.assert_array_equal(ict, 0)
            np.testing.assert_array_equal(contact, 1)
        observed = empty * 0.9
        ctf, ict, contact = homogeneous_ctf_ict(observed, 0.01, gamma=700, alpha=0)
        np.testing.assert_allclose(ctf, 350*(observed - 1))
        np.testing.assert_allclose(ict, 350*np.log(observed))
        np.testing.assert_allclose(np.exp(2*ict/700), contact, atol=1e-14)

    def test_invalid_ict_contact_is_rejected_without_clipping(self):
        observed = np.broadcast_to(1 + 0.5*np.cos(2*np.pi*np.arange(8)/8), (8, 8))
        with self.assertRaisesRegex(ValueError, "nonpositive contact"):
            homogeneous_ctf_ict(observed, 1/50, gamma=1, alpha=0)

    def test_tabulated_material_ratio(self):
        self.assertAlmostEqual(polystyrene_8kev()["gamma"], 721.3088309, places=5)


class TestAPBranch(unittest.TestCase):
    def test_negative_phase_beyond_pi_is_not_reset_to_zero(self):
        props = make_fresnel_propagators((12, 10), (0.01, 0.02),
            wavelength=1.5498e-10, pixel_size=196e-9, dtype=torch.cdouble)
        phase = torch.full((1, 1, 12, 10), -3.5, dtype=torch.float64)
        data = simulate_intensities(phase, props)
        result = alternating_projections(data, props, initial_phase=phase,
                                        nonpositive=True, max_iter=3)
        torch.testing.assert_close(result.estimate, phase, atol=1e-12, rtol=1e-12)
        self.assertLess(result.iterate_residual[-1], 1e-12)

    def test_admm_reports_failure_to_reach_tolerance(self):
        props = make_fresnel_propagators((12, 10), (0.01,),
            wavelength=1.5498e-10, pixel_size=196e-9)
        diagnostics = {}
        constrained_ctf_reconstruct(torch.ones(1, 1, 1, 12, 10), props,
            alpha=0.01, nonpositive=True, max_iter=0, diagnostics=diagnostics)
        self.assertFalse(diagnostics["converged"])
        self.assertEqual(diagnostics["iterations"], 0)


class TestSharedScale(unittest.TestCase):
    def test_same_phase_maps_keep_their_colors_across_comparisons_and_rois(self):
        maps = {name: np.linspace(-4.548, 0, 20*18).reshape(20, 18)
                for name in NONLINEAR_METHODS}
        maps["pgd_free"] = np.linspace(-3.7, 3.565, 20*18).reshape(20, 18)
        maps["ctf_single"] = 10*maps["pgd_free"]
        limits = nonlinear_phase_limits(maps)
        self.assertEqual(limits, (-5, 4))
        metrics = {name: {"iterations": 100, "residual": 0.1} for name in maps}
        for group in GROUPS[1:]:
            for roi in (None, (3, 4, 8)):
                figure, asset = phase_comparison_figure(maps, metrics, group[2], group[1], limits, roi=roi)
                for name, _ in group[2]:
                    if name in NONLINEAR_METHODS:
                        self.assertEqual(asset["color_limits_rad_by_method"][name], [-5, 4])
                for axis in figure.axes:
                    for image in axis.images:
                        if image.norm.vmin == -5:
                            self.assertEqual(image.norm.vmax, 4)
                plt.close(figure)


class TestImportedProvenance(unittest.TestCase):
    def test_import_preserves_bytes_and_refuses_tampering_or_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.npz"
            np.savez(dataset, holograms=np.ones((4, 8, 6)))
            original = root / "original"
            original.mkdir()
            metadata = {"source_hashes": {"old/dataset": sha256(dataset)},
                        "converged": True, "histories": {"step_size": [0.05]}}
            for name in RUNS:
                np.savez(original / f"{name}.npz", phase=np.zeros((8, 6)), metadata=json.dumps(metadata))
            imported = root / "imported"
            import_checkpoints(original, imported, dataset)
            maps, _, hashes = load_results(imported, dataset)
            self.assertEqual(len(maps), len(RUNS))
            self.assertEqual(hashes, metadata["source_hashes"])
            for name in RUNS:
                self.assertEqual(sha256(original / f"{name}.npz"), sha256(imported / f"{name}.npz"))
            path = imported / "pgd_free.npz"
            with self.assertRaisesRegex(ValueError, "immutable historical"):
                validate_checkpoint(path, metadata, dataset, current_only=True)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                load_results(imported, dataset)


if __name__ == "__main__":
    unittest.main()
