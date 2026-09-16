"""Single-distance TIE and matched homogeneous CTF/ICT Fourier inverses.

All frequencies are cycles per pixel, F = pixel_size**2 / (wavelength*z),
and the Fresnel convention is exp(-i*pi*|q|**2/F). Intensities are normalized
to the empty beam (1). These routines do not normalize or crop the input.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.fft import fft2, fftfreq, ifft2


def fresnel_phase(shape, fresnel_number):
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must contain two positive dimensions")
    if not np.isfinite(fresnel_number) or fresnel_number <= 0:
        raise ValueError("fresnel_number must be finite and positive")
    return np.pi * (fftfreq(shape[0])[:, None]**2
                    + fftfreq(shape[1])[None, :]**2) / fresnel_number


def _intensity(observed):
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim != 2 or not np.isfinite(observed).all():
        raise ValueError("observed must be a finite 2D intensity image")
    if np.min(observed) < 0:
        raise ValueError("intensity must be nonnegative")
    return observed


def _inverse(contrast, transfer, alpha):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    denominator = transfer**2 + alpha
    spectrum = np.zeros(contrast.shape, dtype=np.complex128)
    np.divide(transfer * fft2(contrast, norm="ortho"), denominator,
              out=spectrum, where=denominator > 0)
    return ifft2(spectrum, norm="ortho").real


def tie_reconstruct(observed, fresnel_number, *, alpha=0.01):
    """Pure-phase, finite-distance TIE inverse with Tikhonov regularization.

    phi_hat = h * FFT(I-1)/(h**2 + alpha), h = 2*pi*|q|**2/F.
    The DC phase is zero. This linearization requires small Fresnel phase
    over the spatial frequencies of interest; it is not the general TIE PDE.
    """
    observed = _intensity(observed)
    return _inverse(observed - 1, 2 * fresnel_phase(observed.shape, fresnel_number), alpha)


def homogeneous_ctf_ict(observed, fresnel_number, *, gamma, alpha=0.01):
    """Return matched CTF phase, ICT phase, and filtered contact intensity.

    gamma = delta/beta > 0, h = 2*(sin(chi)+cos(chi)/gamma).
    Both inverses regularize s = gamma/2*(I0-1) with the same alpha.
    CTF takes phi=s; ICT takes phi=gamma/2*log(1+2*s/gamma).
    A nonpositive contact intensity is rejected, never silently clipped.
    """
    observed = _intensity(observed)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    chi = fresnel_phase(observed.shape, fresnel_number)
    transfer = 2 * (np.sin(chi) + np.cos(chi) / gamma)
    ctf = _inverse(observed - 1, transfer, alpha)
    contact = 1 + 2 * ctf / gamma
    if np.min(contact) <= 0:
        raise ValueError("ICT has nonpositive contact intensity; cannot take its logarithm")
    ict = gamma / 2 * np.log1p(2 * ctf / gamma)
    return ctf, ict, contact


def predict_intensity(phase, fresnel_number, *, gamma=None):
    """Nonlinear Fresnel prediction, optionally using a homogeneous object."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError("phase must be a finite 2D array")
    if gamma is not None and (not np.isfinite(gamma) or gamma <= 0):
        raise ValueError("gamma must be finite and positive")
    wave = np.exp(1j * phase if gamma is None else phase / gamma + 1j * phase)
    chi = fresnel_phase(phase.shape, fresnel_number)
    return np.abs(ifft2(np.exp(-1j * chi) * fft2(wave, norm="ortho"), norm="ortho"))**2


def polystyrene_8kev():
    """CXRO interpolation retained with the source rows for reproducibility."""
    path = Path(__file__).with_name("data") / "polystyrene-8kev.json"
    table = json.loads(path.read_text())
    factors = np.zeros(2)
    for atom in table["elements"].values():
        rows = np.asarray(atom["rows"])
        factors += atom["count"] * np.array([
            np.interp(table["energy_eV"], rows[:, 0], rows[:, col]) for col in (1, 2)])
    wavelength = 1.5498e-10
    density = table["density_g_cm3"] * 1e6 / table["molar_mass_g_mol"] * 6.02214076e23
    delta, beta = 2.8179403262e-15 * wavelength**2 / (2*np.pi) * density * factors
    return {"delta": float(delta), "beta": float(beta), "gamma": float(delta / beta),
            "source_file": "data/polystyrene-8kev.json",
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sphere_15um_transmission": float(np.exp(-4*np.pi*beta*15e-6/wavelength))}
