# Precursor-Reconfig-for-MAS-2026
Data and code for the paper "Precursor-Triggered Reconfiguration of Communication, Sensing and Actuation Structures for Safe Recovery in Multi-Agent Systems"

# masrecon — Precursor-triggered reconfiguration of communication, sensing
# and actuation structures (code for the IJRNC submission)

Code, results and figures for the manuscript

> "Precursor-Triggered Reconfiguration of Communication, Sensing and
> Actuation Structures for Safe Recovery in Multi-Agent Systems"

Pure Python (NumPy + matplotlib only). One-dimensional formation-keeping
benchmark, split-conformal precursor detection, Kruskal structure selection,
Jacobi-type DMPC with discrete-time CBF constraints.

## Environment (as used for the reported results)

| item | version |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core i7-12700 (2.10 GHz), single thread, 32 GB RAM |
| Python | 3.14.7 |
| NumPy | 2.5.3 |
| matplotlib | 3.11.2 |

```
pip install -r requirements.txt
```

Run every script from the repository root, with UTF-8 forced
(`python -X utf8 ...` or `PYTHONUTF8=1`); on Windows the default code page
(CP932) breaks the log output.

## Repository layout

```
masrecon/      core library (config, structure, plant, risk, qp, select,
               control, runner, metrics)
experiments/   experiment and plotting scripts
results/       result files used in the paper + SHA256SUMS.txt
figs/          final PDF figures
tools/         sha256sums.py (write / verify the hash list)
```

## Reproducing the paper

File names of the plotting scripts predate the final figure numbering;
the table gives the mapping that matters.

| Paper item | Produced by | Result file |
|---|---|---|
| Fig. 1 (architecture) | `experiments/fig1_concept.py` | none (conceptual) |
| Fig. 2 (single trial) | `experiments/exp0_smoke.py`, `figures.py` | `results/smoke_*.npz` |
| Table 1, Fig. 3 | `experiments/exp1_main.py`, `figures.py` | `results/exp1.npz` |
| Table 1 / Sec. 6.2 (w_s = 0 comparison) | `experiments/exp1_ws0.py` | `results/exp1_ws0.npz`, `results/exp1_ws2.npz` |
| Fig. 4 (weight sweep), Sec. 6.3 | `experiments/exp8_wsafety.py`, `fig8_wsafety.py` | `results/exp8_wsafety.csv`, `.npy` |
| Table 2, Fig. 5 (scaling with N) | `experiments/exp4_campaign2.py`, `fig5_nscale.py` | `results/exp4_campaign2.npz` |
| Fig. 6 (selection time) | `experiments/exp2b_seltime.py`, `fig4_scaling.py` | `results/exp2b_seltime.npz` |
| Table 3, Fig. 7 (degradation mechanisms) | `experiments/exp5_mechanism.py`, `figures.py` | `results/exp5_mechanism.npy` |
| Fig. 8 (conformal validity) | `experiments/exp3b_conformal.py`, `fig7_conformal.py` | `results/exp3b_conformal.npz` |

### Suggested order

```
python -X utf8 experiments/exp0_smoke.py
python -X utf8 experiments/exp1_main.py           # ~4 min, 600 runs
python -X utf8 experiments/exp1_ws0.py            # w_safety = 0 comparison
python -X utf8 experiments/exp8_wsafety.py        # weight sweep (210 runs)
python -X utf8 experiments/exp4_campaign2.py      # N = 4, 8, 16
python -X utf8 experiments/exp2b_seltime.py       # selection-time scaling
python -X utf8 experiments/exp3b_conformal.py     # conformal validity sweep
python -X utf8 experiments/exp5_mechanism.py      # degradation mechanisms
python -X utf8 experiments/fig8_wsafety.py
python -X utf8 experiments/fig5_nscale.py
python -X utf8 experiments/fig4_scaling.py
python -X utf8 experiments/fig7_conformal.py
python -X utf8 experiments/figures.py             # Figs. 1, 2, 3, 7
```

Run times other than `exp1_main.py` depend on the machine; the figure
scripts take seconds.

## Built-in checks

* `exp4_campaign2.py` asserts, before saving, that its N = 4 column
  reproduces the w_s = 2 column of the weight sweep (`exp8`) run for run
  in violation, violation area and recovery time. A mismatch aborts the run.
* `exp1_main.py` draws the initial condition once and passes it to all three
  methods, so the paired tests are valid.
* Figures read their data from a single result file; the script prints that
  file's SHA-256 prefix.

## Result files and hashes

`results/SHA256SUMS.txt` lists the full SHA-256 of every result file behind a
data figure or table.

```
python -X utf8 tools/sha256sums.py verify
```

The hash identifies the files used for the manuscript. It does **not**
guarantee that a re-run reproduces the same bytes: NumPy's `Generator` is not
bit-reproducible across versions, and `.npz` archives embed timestamps.
Aggregate statistics are insensitive to this; individual run values may differ
slightly between environments.

## Scope and known limitations

* One-dimensional benchmark, N <= 16 (selection-time test up to |C| = 2144).
* Remedy 2 of Section 4.2 (hard constraint by matroid contraction,
  `FORCE = True`) is implemented in `masrecon/select.py` but is **not**
  evaluated in the paper.
* The nominal-drift fallback (used when a critical edge is lost) is kept
  deliberately: it is the configuration the paper analyses as violating the
  tube condition (G2).
* Barrier constraints are softened with a quadratic penalty; the paper makes
  no claim about the magnitude of the slack (Remark 6, Lemma 4).

## License

Code: MIT (see `LICENSE`). Result files in `results/`: CC BY 4.0.

## Citation

Please cite the manuscript (reference to be added on acceptance) and, for the
code, the archived release (DOI to be added).
