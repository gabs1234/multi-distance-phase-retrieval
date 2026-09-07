# Multi-distance phase retrieval

A [marimo](https://marimo.io/) notebook comparing CTF, alternating projections,
projected gradient descent, and nonlinear Tikhonov phase retrieval using DeepInv.

## Run

Requires Python 3.12+ and uv. Set the DeepInv checkout path in
`pyproject.toml` to your local checkout; the current configuration uses a
development version with Fresnel propagation support.

```bash
uv sync
uv run marimo edit phase_retrieval_comparison.py
```

The notebook starts with synthetic data. Choose the settings and press
**Run all five stages**. For measured data, place
`holograms_beads_updated.npz` beside the notebook; it is excluded from Git.

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
