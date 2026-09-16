# Multi-distance phase retrieval

A [marimo](https://marimo.io/) notebook comparing CTF, alternating projections,
projected gradient descent, and nonlinear Tikhonov phase retrieval using DeepInv.

## Run

Requires Python 3.12+ and uv. DeepInv is pinned to a public development
commit with Fresnel propagation support.

```bash
uv sync
uv run marimo edit phase_retrieval_comparison.py
```

The notebook starts with synthetic data. Choose the settings and press
**Run all five stages**. For measured data, place
`holograms_beads_updated.npz` beside the notebook; it is excluded from Git.

For measured data, **measured reconstruction field** selects a **Central ROI**
or **Full projections** for all five sections. Full projections reconstruct
all 2048 × 1920 pixels at native sampling. After the full-field comparison,
the **Full-field phase close-up** viewer lets you select any method and change
the displayed region's size and position without rerunning the solvers.
The independent optional NLTikh run also has its own close-up controls.

## Data and reference

The plots in [figures/](figures/) were generated here from the experimental
polystyrene-sphere data used by Huhn et al.; the measurements are their group's
work. The dataset was previously studied by
[Hagemann, Töpperwien, and Salditt (2018)](https://doi.org/10.1063/1.5029927).

The nonlinear Tikhonov method and measured-data example follow:

Simon Huhn, Leon Merten Lohse, Jens Lucht, and Tim Salditt.
[Fast algorithms for nonlinear and constrained phase retrieval in near-field X-ray holography based on Tikhonov regularization](https://arxiv.org/html/2205.01099v2).
*Optics Express* 30(18), 32871–32886 (2022).
[doi:10.1364/OE.462368](https://doi.org/10.1364/OE.462368).

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

## Reproduce the presentation comparisons

The scientific source now lives in this repository. The [separate presentation repository](https://github.com/gabs1234/deepinv-hackathon-day-1)
contains the narrative, Typst equations, and copies of the generated assets;
its Python entry points delegate to the scripts here.

The notebook has a **Converged results used in the presentation** viewer. Press
**Load converged measured-data results** to inspect saved full fields and the
same tight sphere ROI. The five interactive stages below remain inexpensive
finite-iteration experiments, and are labeled as such. Their nonlinear maps
also share one display range across comparisons and close-ups.

For a fresh native-resolution run (four 2048 × 1920 holograms):

```bash
uv run python scripts/converge-phase-retrieval.py
uv run python scripts/converge-phase-retrieval.py --verify
uv run python scripts/render-converged-phase-retrieval.py
uv run python scripts/render-tie-introduction.py
```

These commands write checkpoints to `.cache/phase-retrieval/converged/` and
figures to `figures/comparison/`, both ignored by Git. `--dataset`, `--cache`,
and the renderers' `--output` options accept explicit paths. The direct-method
renderer uses `--converged-cache` for CTF inputs and `--cache` for its own direct
inverses. Renderers need only CPU; the native solver defaults to CUDA. A CPU
solver is available with `--device cpu`. Verify on the same backend used to
compute a run: float32 CPU/GPU propagators can differ enough to fail the strict
residual equality check, even when their displayed reconstructions agree.

Completed runs are reused, and unfinished runs resume. `--methods pgd_single
pgd_multi` selects a subset; `--blocks 1` runs at most one 100-step block per
method and returns an error if convergence remains unconfirmed. NLTikh restarts
its BB and line-search memory at every block, matching the presentation run.
Changes to solver code, runner settings, data, or dependency versions invalidate
a normal cache. Changes to notebook prose or plotting do not.

### Existing presentation results

The imported presentation checkpoints are in
`.cache/phase-retrieval/presentation/`. Their arrays and original metadata are
preserved byte-for-byte. An `imported-provenance.json` records their hashes;
these historical results can be verified and rendered, but cannot be resumed
with modified source. They are local data, not bundled in Git or in `docs/`.

```bash
uv run python scripts/converge-phase-retrieval.py --verify \
  --cache .cache/phase-retrieval/presentation
uv run python scripts/render-converged-phase-retrieval.py \
  --cache .cache/phase-retrieval/presentation
uv run python scripts/render-tie-introduction.py \
  --converged-cache .cache/phase-retrieval/presentation
```

For an existing external checkpoint directory, import it once into a new path:

```bash
uv run python scripts/import-phase-retrieval-checkpoints.py \
  --source /path/to/old/checkpoints --cache .cache/phase-retrieval/presentation
```

A clean checkout without those local checkpoints must compute fresh results
with the first set of commands. No reconstruction is inferred from PNG files.

### Scientific and plotting conventions

- PGD/NLTikh: projected-gradient mapping divided by the zero-phase gradient
  norm ≤ 10⁻³. AP: all of its last 20 relative phase updates ≤ 10⁻⁵.
  Both also require `abs(E[k] - E[k-100]) / E[0] ≤ 10⁻⁴`, checked every 100
  iterations. The constrained CTF warm start requires both relative ADMM
  residuals ≤ 10⁻³. Hitting an iteration limit is not declared convergence.
- AP tracks phase increments on the previous iterate's branch before applying
  sign/support constraints. This fixes the reset to zero at phases below −π;
  it is not spatial phase unwrapping. The checkpoint name `ap_continuous`
  distinguishes the corrected historical run.
- Every nonlinear full-field comparison and ROI uses `magma` with extrema
  pooled across all seven nonlinear methods and rounded outward to whole
  radians. The existing measured runs use **−5 to +4 rad**, including positive
  free-PGD values. Mixed CTF/nonlinear panels have separate labeled CTF bars.
  CTF uses pooled full-field 1–99% limits with saturation markers. No phase
  offsets, rescaling, or ROI-specific renormalization are applied.
- The 192 × 192 ROI starts at zero-based `(row, column) = (583, 870)` and spans
  37.632 µm at 196 nm per pixel. It approximately matches the red box in Huhn
  Figure 1; the paper does not provide exact pixel coordinates. Cropping is
  performed after reconstruction. Renderers accept `--roi TOP LEFT SIZE`.
- TIE is the pure-phase finite-distance inverse with transfer `h = 2χ` and
  zero DC, where `χ = π|q|²/F`. Its regularized inverse uses `h/(h² + α)`.
  This is the small-propagation linearization, not a solver for the general
  variable-intensity TIE PDE.
- The additional homogeneous CTF/ICT pair uses the same tabulated polystyrene
  ratio `γ = δ/β ≈ 721.31` and `α = 0.01`. Both first invert with
  `hγ = 2(sin χ + cos χ/γ)`. CTF takes the filtered contrast `s` as phase;
  ICT takes `φ = γ/2 log(1 + 2s/γ)`. The common regularization acts on
  `s = γ/2 (I₀−1)`. Nonpositive contact intensities are rejected before taking
  the logarithm. Source rows and interpolation are in the package's
  `data/polystyrene-8kev.json`. See [Faragó et al.](https://doi.org/10.1364/OL.530330).
- Residuals use each method's measured planes and nonlinear forward model;
  the homogeneous CTF/ICT pair includes absorption in both predictions. The
  other comparisons remain pure phase. These are intensity consistency
  measures, not phase-accuracy scores; this experimental dataset has no phase
  ground truth. Numerical convergence does not establish a global optimum.

`manifest.json`, `details-manifest.json`, `convergence-summary.json`, and
`tie-introduction-manifest.json` accompany the figures with data/source hashes,
original histories, stopping checks, ROI coordinates, and display limits.
