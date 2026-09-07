# Controlled multi-distance Fresnel phase retrieval

This project contains a reactive [Marimo](https://marimo.io/) notebook for a
controlled scientific comparison of linear and nonlinear near-field phase
retrieval. It follows the pure-phase convention

\[
T = \exp(i\phi), \qquad \phi \leq 0,
\]

and the four Fresnel numbers of the polystyrene-bead data used by Huhn et al.

## Run the notebook

The project is already configured to use the editable DeepInv checkout declared
in `pyproject.toml`.

```bash
uv sync
uv run marimo edit phase_retrieval_comparison.py
```

For a read-only application:

```bash
uv run marimo run phase_retrieval_comparison.py
```

Choose the data and numerical settings, then press **Run all five stages**.
The default 128 × 128 synthetic comparison takes only a few seconds on Apple
Silicon. The measured-data branch can take longer. If an MPS operation is not
available in a particular PyTorch release, launch with
`PYTORCH_ENABLE_MPS_FALLBACK=1` or select CPU in the local DeepInv setup.

For measured data, the preview shows every uncropped 2048 × 1920 hologram with
the reconstruction crop outlined and enlarged as an inset. The inset carries a
horizontal or vertical intensity line cut, sorted by increasing equivalent
propagation distance (decreasing Fresnel number). Move the row/column slider to
follow the same feature through all four planes; optionally average across a
3/5/9-pixel band. A dotted copy of the first-distance profile highlights fringe
shifts. All profiles share the same spatial and intensity scales, and these
controls do not rerun the solvers.
An example using the center row of the 128 × 128 measured crop is saved as
[PNG](figures/measured_hologram_line_cuts.png) and
[SVG](figures/measured_hologram_line_cuts.svg).

Every crop-based measured reconstruction includes a small first-distance
full-FOV inset with the reconstructed region outlined in orange. The phase
outside that outline has not been computed. For an actual full-field phase,
select measured data and use **Run optional native-field NLTikh**. This separate
run applies the final Huhn-style method to all 2048 × 1920 pixels while leaving
the controlled five-stage comparison at the selected crop size. Its CTF and
nonlinear iteration counts have separate controls because the native field has
240 times as many pixels as the default 128 × 128 crop.
The conservative iteration defaults are intended as a feasibility check;
increase both counts and confirm residual/phase stability before interpreting
the native-field reconstruction scientifically.

## Experimental sequence

The notebook is organized so every direct contrast changes one axis:

1. A 2 × 2 linear CTF baseline compares one/four distances and zero/scalar
   Tikhonov regularization.
2. Single-distance nonlinear DeepInv PGD compares an unconstrained phase with
   the same run under physical constraints.
3. The constrained nonlinear solver changes from one to four distances. Its
   stacked loss is averaged over planes to keep the PGD gradient scale fixed.
4. Averaged alternating projections and DeepInv projected GD share data,
   physics, initialization, constraints, and iteration count. They are compared
   using a common intensity residual, both per iteration and per approximate
   propagation budget.
5. A sequential ablation adds constrained CTF initialization,
   frequency-dependent Tikhonov regularization, and finally Huhn's alternating
   Barzilai–Borwein steps, nonmonotone line search, and gradient stopping rule.

The synthetic source is the default because it provides a known phase and
support. The included measured holograms have no reference phase and contain an
extended bead field, so the notebook reports data consistency but makes no
accuracy claim and does not invent a compact support mask. Center cropping also
introduces an explicitly noted FFT-boundary limitation.

The measured branch reads `holograms` and `fresnelNumbers` directly from the
local `holograms_beads_updated.npz`. The accompanying demo describes the data as
dark/flat-field corrected, registered, and rescaled to common magnification;
the archive itself contains a publication reference but no raw calibration
fields with which to independently verify the corrections. Center cropping
preserves the stored intensity values: no further flat/dark correction or
per-crop mean normalization is applied. Distances displayed in the preview
are equivalent plane-wave distances, consistent with the DeepInv model.

## DeepInv implementation boundary

The executable nonlinear implementation stays with DeepInv:

- `FresnelPropagation` performs wave propagation.
- `PhaseRetrieval` and composed/stacked `Physics` objects implement
  \(\phi \mapsto |P_z e^{i\phi}|^2\).
- `StackedPhysicsDataFidelity` and `L2` define the data term.
- DeepInv automatic VJPs provide nonlinear gradients.
- `deepinv.optim.PGD` runs every fixed-step projected-gradient comparison.

Thin local code supplies only methods not currently exposed as the needed
DeepInv abstractions: the CTF inverse, constraint projection, averaged
amplitude-projection loop, Fourier-diagonal filter, and Huhn-specific
BB/nonmonotone update policy. HoToPy was consulted as an implementation
cross-check for the accelerated-ADMM restart and nonmonotone proximal
backtracking conventions; it is neither imported nor listed as a dependency.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run marimo check --strict phase_retrieval_comparison.py
```

For a noninteractive full-cell smoke test:

```bash
PHASE_RETRIEVAL_AUTORUN=1 \
  uv run marimo export html phase_retrieval_comparison.py \
  -o /tmp/phase-retrieval-comparison.html --force
```

Set `PHASE_RETRIEVAL_SOURCE=measured` as well to exercise the measured branch.

The numerical tests independently check the CTF sign/derivative, the linear
inverse, the DeepInv objective and automatic VJP against an analytic gradient,
constraint enforcement, PGD descent, the Huhn filter levels, and AP/Huhn smoke
runs.

## Files

- `phase_retrieval_comparison.py` — interactive scientific notebook.
- `src/multi_distance_phase_retrival_huhn/phase_retrieval.py` — tested solver
  and diagnostic building blocks.
- `tests/test_phase_retrieval.py` — numerical regression tests.
- `src/multi_distance_phase_retrival_huhn/hologram_plots.py` — matched hologram
  line cuts, also usable for standalone figure exports.
- `holograms_beads_updated.npz` — corrected and registered four-distance bead
  holograms.
- `demo_fresnel_phase_retrieval.py` — compact original DeepInv example.

## References

- Huhn, Lohse, Lucht, and Salditt, “Nonlinear Tikhonov regularization for
  x-ray phase-contrast tomography,” 2022:
  <https://arxiv.org/html/2205.01099v2>
- Hagemann et al., “Single-pulse phase-contrast imaging at free-electron
  lasers in the hard x-ray regime,” 2018:
  <https://doi.org/10.1063/1.5029927>
- HoToPy holography API, reference-only cross-check:
  <https://irp.pages.gwdg.de/hotopy/reference/generated/hotopy.holo.html>
