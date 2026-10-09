# Precursor-Reconfig-for-MAS-2026

Data and code for the paper "Precursor-Triggered Reconfiguration of Safety
Contracts and Communication Structures for Safe Recovery in Multi-Agent
Systems" (submitted to the International Journal of Robust and Nonlinear
Control).

Pure Python (NumPy + matplotlib only). One-dimensional formation-keeping
benchmark, split-conformal precursor detection, exact structure selection on a
contracted graphic matroid, Jacobi-type DMPC with discrete-time CBF
constraints, latched stop contracts and a one-step shield.

## Environment (as used for the reported results)

| item | version |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core i7-12700 (2.10 GHz), single thread, 32 GB RAM |
| Python | 3.14.7 |
| NumPy | 2.5.3 |
| matplotlib | 3.11.2 |

    pip install -r requirements.txt

Run every script from the repository root with UTF-8 forced
(`python -X utf8 ...` or `PYTHONUTF8=1`).

## Repository layout

    masrecon/      core library (config, structure, plant, risk, qp, select,
                   control, runner, metrics)
    experiments/   experiment and plotting scripts
    results/       result files used in the paper + SHA256SUMS.txt
    figs/          final PDF figures
    tools/         sha256sums.py, check_impl.py

## Reproducing the paper

| Paper item | Produced by | Result file |
|---|---|---|
| Fig. 1 (architecture) | `experiments/fig_v12.py fig1` | none |
| Table 1, Table 3 (Campaign I) | `experiments/exp9_remedies.py --set c1` | `results/exp9_c1.npz` |
| Fig. 2 (single trial) | `experiments/fig_v12.py fig2` | recomputed |
| Table 2 (oracle lead, static spacing, N = 4) | `experiments/exp11_lead.py` | `results/exp11_lead.npz` |
| Sec. 6.3 (state-based check of Theorem 1) | `experiments/exp11b_check.py` | log only |
| Sec. 6.3 (run (u,2,2)/7) | `experiments/exp11c_run7.py` | log only |
| Fig. 3 (ablation, weight sweep) | `experiments/fig_v12.py fig8` | `results/fig8_v12.npz`, `results/exp8_wsafety.csv` |
| Table 4, Fig. 6 (degradation mechanisms) | `experiments/exp5_mechanism.py` | `results/exp5_mechanism.npy` |
| Table 5, Fig. 4 (scaling with N) | `experiments/exp11_lead.py --N 8/16`, `experiments/exp4_campaign2.py`, `experiments/fig_v12.py fig5` | `results/exp11_lead_N8.npz`, `results/exp11_lead_N16.npz`, `results/exp4_campaign2.npz` |
| Fig. 5 (conformal validity) | `experiments/exp3b_conformal.py`, `fig7_conformal.py` | `results/exp3b_conformal.npz` |
| Fig. 7 (selection time) | `experiments/exp2b_seltime.py`, `fig4_scaling.py` | `results/exp2b_seltime.npz` |

Script and figure file names predate the final numbering; the table gives the
mapping that matters.

## Configurations used in the paper

| Paper name | `exp9` / `exp11` condition |
|---|---|
| post-oracle | `B1_post` |
| always-stop | `B2_always_stop` |
| tighten-only | `C5_tighten_only` |
| latch-held | `T_latch_stop` |
| proposed | `T_latch_rel10` |
| latch-reconf | `T_latch_reconf` |
| static | `S_static` |
| ablation (weighted, drift) | `C0_weight` |

## Result files and hashes

`results/SHA256SUMS.txt` lists the SHA-256 of every result file behind a table
or data figure:

    python -X utf8 tools/sha256sums.py verify

The hash identifies the files used for the manuscript; it does not guarantee
that a re-run reproduces the same bytes (NumPy's `Generator` is not
bit-reproducible across versions). Aggregate statistics are insensitive to
this.

## Scope and known limitations

* One-dimensional benchmark, N <= 16 (selection-time test up to |C| = 2144).
* No process disturbance; randomness enters through initial positions and
  diagnostic signals only.
* The shield has no recovery mode: after a missed precursor a follower inside
  d_min of a stopped predecessor is held at rest.
* The nominal-drift fallback is kept only for the ablation, as the
  configuration that violates tube condition (G2).
* A risk-dependent tube back-off is permitted by the theory but not
  implemented.

## License

Code: MIT (see `LICENSE`). Result files in `results/`: CC BY 4.0.

## Citation

Please cite the manuscript (reference to be added on acceptance) and the
archived release: https://doi.org/10.5281/zenodo.XXXXXXXX
