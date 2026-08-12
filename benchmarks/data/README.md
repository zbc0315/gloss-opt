# Benchmark data

All data needed to reproduce the paper benchmarks ships with this repository:

- `qm9.csv`: QM9 dataset (DeepChem mirror of Ramakrishnan et al., 2014).
- `buchwald_hartwig.xlsx`: Buchwald-Hartwig reaction yields (Dreher and
  Doyle input data, from the rxn_yields repository).
- `qm9_strat/*.npz`: stratified QM9 candidate pools used by the scaling
  study. These are derived from `qm9.csv` and can be rebuilt with
  `python -m benchmarks.build_qm9_strat`.

The Arrhenius-2D landscape is analytic (`benchmarks/virtual_functions.py`)
and needs no data files. If a file is deleted, `benchmarks/datasets.py`
re-downloads it from the original public source on next use.
