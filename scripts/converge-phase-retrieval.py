"""Rerun the notebook's full-field comparisons with explicit convergence checks.

Uses the same solvers as the notebook. Checkpoints retain native phase maps
and complete histories, so plots can be regenerated without solving again.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phase-retrieval-mpl")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import torch
from multi_distance_phase_retrival_huhn.phase_retrieval import (
    alternating_projections, constrained_ctf_reconstruct, ctf_reconstruct,
    huhn_nltikh, huhn_regularization_filter,
    make_deepinv_phase_problem, make_fresnel_propagators, project_phase,
    nonlinear_value_and_gradient,
    projected_gradient_descent, relative_data_residual,
)

from multi_distance_phase_retrival_huhn.experiment_io import (
    DEFAULT_CACHE, DEFAULT_DATASET, fingerprint, validate_checkpoint,
)

CACHE = DEFAULT_CACHE
DEVICE = "cuda"
DATASET = DEFAULT_DATASET
PG_TOL = 1e-3
AP_TOL = 1e-5
BLOCK = 100
STABLE_OBJECTIVE = 1e-4
MAX_ITER = 20000
SPECS = {
    "pgd_free": (1, False, False, False),
    "pgd_single": (1, True, False, False),
    "pgd_multi": (4, True, False, False),
    "ap_continuous": (4, True, False, False),
    "pgd_warm": (4, True, True, False),
    "pgd_tikh": (4, True, True, True),
    "nltikh": (4, True, True, True),
}
HISTORIES = ("objective", "data_term", "regularization", "data_residual",
             "relative_gradient", "projected_gradient", "iterate_residual",
             "step_size", "operator_calls")


def sources():
    return fingerprint(DATASET)


def save(name, phase, metadata):
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / f"{name}.npz"
    temporary = CACHE / f"{name}.pending.npz"
    np.savez(temporary, phase=phase.detach().squeeze().cpu().numpy(),
             metadata=json.dumps(metadata))
    temporary.replace(destination)


def load(name, source_hashes, *, current_only=True):
    path = CACHE / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata"]))
        validate_checkpoint(path, metadata, DATASET, current_only=current_only, hashes=source_hashes)
        phase = torch.from_numpy(archive["phase"].copy()).to(DEVICE)[None, None]
    return phase, metadata


def stationarity(phase, observations, physics, fidelity, nonpositive):
    with torch.no_grad():
        gradient = fidelity.grad(phase, observations, physics)
        # Unit-step projected-gradient mapping; the same definition as the
        # notebook's projected_gradient diagnostic, normalized at zero.
        projected = phase - project_phase(phase - gradient, nonpositive=nonpositive)
        zero_gradient = fidelity.grad(torch.zeros_like(phase), observations, physics)
        scale = torch.linalg.vector_norm(zero_gradient).clamp_min(1e-12)
        return float(torch.linalg.vector_norm(projected) / scale), float(
            torch.linalg.vector_norm(gradient) / scale)


def warm_start(measurements, props, alpha, source_hashes):
    cached = load("ctf_warm", source_hashes)
    if cached is not None:
        return cached[0]
    state = {}
    started = time.perf_counter()
    with torch.no_grad():
        phase = constrained_ctf_reconstruct(measurements, props, alpha=alpha,
            nonpositive=True, max_iter=10000, tolerance=1e-3, diagnostics=state)
    if max(state["primal_residual"], state["dual_residual"]) > 1e-3:
        raise RuntimeError(f"Constrained CTF did not converge: {state}")
    metadata = {"source_hashes": source_hashes, "iterations": state.pop("iterations"),
                "converged": True, "stop": "ADMM primal and dual residuals <= 1e-3",
                "elapsed_seconds": time.perf_counter() - started, **state}
    save("ctf_warm", phase, metadata)
    print("ctf_warm", json.dumps(metadata), flush=True)
    return phase


def run(name, measurements, props, numbers, alpha, warm, source_hashes, blocks):
    count, constrained, warm_started, regularized = SPECS[name]
    data, operators = measurements[:count], props[:count]
    penalty = alpha if regularized else None
    reduction = "sum" if warm_started else "mean"
    observations, physics, fidelity = make_deepinv_phase_problem(
        data, operators, alpha=penalty, reduction=reduction)
    cached = load(name, source_hashes)
    if cached is None:
        phase = warm.clone() if warm_started else torch.zeros_like(data[0])
        metadata = {"source_hashes": source_hashes, "device": DEVICE, "iterations": 0,
            "converged": False, "elapsed_seconds": 0.0, "checks": [],
            "histories": {key: [] for key in HISTORIES}, "planes": count,
            "step": None if name == "nltikh" else (0.05 if warm_started else 0.2), "reduction": reduction,
            "constraint": "nonpositive" if constrained else "free",
            "initialization": "converged constrained CTF" if warm_started else "zero",
            "regularized": regularized,
            "algorithm": "notebook solver in blocks of 100; NLTikh BB and line-search memory restart at each block"}
    else:
        phase, metadata = cached
    metadata["criterion"] = (
        "last 20 relative phase updates <= 1e-5" if name.startswith("ap")
        else "relative projected gradient <= 1e-3") + (
        " and abs(E_k - E_(k-100)) / E(0) <= 1e-4")
    with torch.no_grad():
        reference_objective = float(fidelity.fn(torch.zeros_like(phase), observations, physics).sum())
    if name.startswith("ap"):
        # AP's recorded diagnostic is the full sum of intensity squares.
        reference_objective *= 2 * count
    metadata["reference_objective_at_zero"] = reference_objective
    for check in metadata["checks"]:
        index = check["iteration"]
        values = metadata["histories"]["objective"]
        check["relative_objective_change_100"] = abs(values[index] - values[index - BLOCK]) / reference_objective
    if metadata["converged"]:
        print(f"{name}: loaded converged run ({metadata['iterations']} iterations)", flush=True)
        return
    for _ in range(blocks):
        if metadata["iterations"] >= MAX_ITER:
            raise RuntimeError(f"{name} reached {MAX_ITER} without convergence")
        before = phase.clone()
        with torch.no_grad():
            if name == "ap_continuous":
                result = alternating_projections(data, operators, initial_phase=phase,
                    nonpositive=constrained, max_iter=BLOCK, tolerance=0)
                metadata["algorithm"] = "Averaged AP with phase increments tracked continuously before nonpositive projection"
            elif name == "nltikh":
                result = huhn_nltikh(data, operators, numbers, initial_phase=phase,
                    nonpositive=constrained, max_iter=BLOCK, tolerance=0,
                    alpha_low=1e-3, alpha_high=1e-1, alpha_beyond_na=8.0,
                    max_line_search_steps=20)
            else:
                result = projected_gradient_descent(data, operators, initial_phase=phase,
                    nonpositive=constrained, max_iter=BLOCK, tolerance=0,
                    step_size=metadata["step"], alpha=penalty, reduction=reduction)
        phase = result.estimate
        metadata["elapsed_seconds"] += result.elapsed_seconds
        first = metadata["iterations"] == 0
        history = metadata["histories"]
        previous_calls = history["operator_calls"][-1] if history["operator_calls"] else 0
        for key in HISTORIES:
            values = list(getattr(result, key))
            if key == "operator_calls":
                values = [previous_calls + value for value in values]
            history[key].extend(values if first else values[1:])
        metadata["iterations"] += result.iterations
        pg, raw = stationarity(phase, observations, physics, fidelity, constrained)
        objective_change = abs(result.objective[-1] - result.objective[0]) / max(
            reference_objective, 1e-12)
        phase_change = float(torch.linalg.vector_norm(phase - before) /
                             torch.linalg.vector_norm(phase).clamp_min(1e-12))
        update = max(result.iterate_residual[-20:])
        converged = (update <= AP_TOL if name.startswith("ap") else pg <= PG_TOL) and (
            objective_change <= STABLE_OBJECTIVE and result.iterations == BLOCK)
        metadata["converged"] = converged
        metadata["residual"] = relative_data_residual(phase, data, operators)
        check = {"iteration": metadata["iterations"], "relative_projected_gradient": pg,
            "relative_raw_gradient": raw, "relative_phase_change_100": phase_change,
            "relative_objective_change_100": objective_change, "max_last_20_updates": update,
            "residual": metadata["residual"]}
        metadata["checks"].append(check)
        metadata["stop"] = "convergence criterion reached" if converged else result.stop_reason
        save(name, phase, metadata)
        print(name, json.dumps({**check, "converged": converged,
                               "seconds": round(metadata["elapsed_seconds"], 1)}), flush=True)
        if result.iterations < BLOCK:
            raise RuntimeError(f"{name}: {result.stop_reason}; convergence not certified")
        if converged:
            return


def main():
    global CACHE, DATASET, DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--blocks", type=int, default=MAX_ITER // BLOCK)
    parser.add_argument("--verify", action="store_true", help="Independently recompute saved stopping measures")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    CACHE, DATASET, DEVICE = args.cache.resolve(), args.dataset.resolve(), args.device
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native-resolution runs")
    torch.set_num_threads(4)
    source_hashes = sources()
    with np.load(DATASET) as archive:
        measurements = torch.from_numpy(archive["holograms"]).to(DEVICE)[:, None, None]
        numbers = tuple(float(value) for value in archive["fresnelNumbers"])
    props = make_fresnel_propagators(tuple(measurements.shape[-2:]), numbers,
        wavelength=1.5498e-10, pixel_size=196e-9, device=DEVICE)
    alpha = huhn_regularization_filter(tuple(measurements.shape[-2:]), numbers,
        alpha_low=1e-3, alpha_high=1e-1, alpha_beyond_na=8.0, device=DEVICE)
    if args.verify:
        verify(measurements, props, alpha, source_hashes)
        return
    with torch.no_grad():
        for name, count, scalar in (("ctf_single", 1, 0.0), ("ctf_single_tikh", 1, 1e-2),
                ("ctf_multi", 4, 0.0), ("ctf_multi_tikh", 4, 1e-2)):
            if load(name, source_hashes) is None:
                phase = ctf_reconstruct(measurements[:count], props[:count], alpha=scalar)
                save(name, phase, {"source_hashes": source_hashes, "converged": True,
                    "iterations": 0, "stop": "closed-form inverse", "planes": count,
                    "residual": relative_data_residual(phase, measurements[:count], props[:count])})
    warm = warm_start(measurements, props, alpha, source_hashes)
    for name in args.methods:
        run(name, measurements, props, numbers, alpha, warm, source_hashes, args.blocks)
    incomplete = [name for name in args.methods if not load(name, source_hashes)[1]["converged"]]
    if incomplete:
        raise RuntimeError(f"Saved partial checkpoints; convergence not reached: {incomplete}")


def verify(measurements, props, alpha, hashes):
    report = []
    with torch.no_grad():
        for name, (count, constrained, warm_started, regularized) in SPECS.items():
            phase, result = load(name, hashes, current_only=False)
            assert result["converged"], name
            assert torch.isfinite(phase).all(), name
            assert not constrained or float(phase.max()) <= 0, name
            assert phase.shape == measurements.shape[1:], name
            data, operators = measurements[:count], props[:count]
            residual = relative_data_residual(phase, data, operators)
            assert math.isclose(residual, result["residual"], rel_tol=1e-5), (
                name, "residual differs; verify with the original compute backend",
                DEVICE, residual, result["residual"])
            check = result["checks"][-1]
            assert check["relative_objective_change_100"] <= STABLE_OBJECTIVE, name
            if name == "ap_continuous":
                next_result = alternating_projections(data, operators, initial_phase=phase,
                    nonpositive=True, max_iter=1)
                stopping = next_result.iterate_residual[-1]
                assert stopping <= AP_TOL, (name, stopping)
                assert max(result["histories"]["iterate_residual"][-20:]) <= AP_TOL
            else:
                # Independent analytic derivative, rather than the automatic
                # derivative used in DeepInv's solver and checkpoint checks.
                kw = {"alpha": alpha if regularized else None,
                      "reduction": "sum" if warm_started else "mean"}
                objective, _, _, twice_gradient = nonlinear_value_and_gradient(phase, data, operators, **kw)
                _, _, _, twice_reference = nonlinear_value_and_gradient(torch.zeros_like(phase), data, operators, **kw)
                mapping = phase - project_phase(phase - twice_gradient / 2, nonpositive=constrained)
                stopping = float(torch.linalg.vector_norm(mapping) / torch.linalg.vector_norm(twice_reference / 2))
                assert stopping <= PG_TOL, (name, stopping)
                assert math.isclose(stopping, check["relative_projected_gradient"], rel_tol=2e-3, abs_tol=1e-7), name
                assert math.isclose(float(objective) / 2, result["histories"]["objective"][-1], rel_tol=1e-5), name
            assert len(result["histories"]["data_residual"]) == result["iterations"] + 1, name
            report.append({"method": name, "iterations": result["iterations"], "residual": residual,
                           "independent_stopping_measure": stopping, "passed": True})
    (CACHE / "verification.json").write_text(json.dumps({"verification_source_hashes": hashes, "device": DEVICE, "checks": report}, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
