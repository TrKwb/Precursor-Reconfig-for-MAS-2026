# -*- coding: utf-8 -*-
"""Configuration dataclasses for the precursor-triggered reconfiguration study."""
from dataclasses import dataclass, field


@dataclass
class PlantCfg:
    N: int = 4
    dt: float = 0.05
    T: float = 26.0
    spacing: float = 1.9            # nominal inter-agent spacing delta
    d_min: float = 1.2              # minimum safe gap
    u_max: float = 3.6
    v_ref: float = 3.0              # moving formation reference
    drag: float = 1.0               # viscous drag coefficient c
    p0_jitter: float = 0.10
    seed: int = 0
    # The nominal margin delta - d_min = 0.70 m is deliberately smaller than the
    # braking distance from v_ref (0.818 m).  A follower that begins braking at
    # the instant the agent ahead of it stops therefore cannot avoid an
    # incursion; only a follower that has already opened the gap can.  This is
    # the regime in which anticipation is worth having.


@dataclass
class DegradeCfg:
    t_deg: float = 6.0              # degradation onset [s]
    T_deg: float = 6.0              # duration; failure at t_deg + T_deg
    ar_lo: float = 0.30             # AR(1) coefficient, healthy
    ar_hi: float = 0.97             # ... at failure (critical slowing down)
    sig_lo: float = 1.0
    sig_hi: float = 2.8
    drop_hi: float = 0.18           # intermittent dropout probability


@dataclass
class RiskCfg:
    W: int = 64                     # sliding-window length [samples]
    stride: int = 4
    pe_order: int = 4
    pe_delay: int = 1
    emb_m: int = 3                  # RQA embedding dimension
    emb_tau: int = 2
    rr_target: float = 0.05         # recurrence rate
    l_min: int = 2
    n_cal: int = 1200               # conformal calibration windows
    alpha: float = 0.20             # family-wise level (union bound over |C|)
    persistence: int = 2            # consecutive flags required
    # alpha_e = alpha / |C| must satisfy alpha_e >= 1/(n_cal+1), otherwise the
    # detector cannot fire at all; build_calibration() checks this.


@dataclass
class CtrlCfg:
    Nh: int = 10                    # MPC horizon
    q_p: float = 4.0
    q_v: float = 1.0
    r_u: float = 0.20
    gamma: float = 0.25             # discrete-CBF decay
    slack_w: float = 3e3
    drift_rate: float = 1.2         # worst-case bound growth when a link is absent
    est_drift: float = 0.6          # dead-reckoning error growth when unsensed


@dataclass
class Cfg:
    plant: PlantCfg = field(default_factory=PlantCfg)
    deg: DegradeCfg = field(default_factory=DegradeCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    ctrl: CtrlCfg = field(default_factory=CtrlCfg)

    # ---- structure-selection objective -------------------------------------
    w_risk: float = 1.0             # penalise risky elements
    w_change: float = 0.35          # penalise switching
    w_perf: float = 0.5             # generic per-element cost

    w_safety: float = 2.0           # REWARD barrier-critical edges
    # An edge (i,j) is safety-critical when a barrier function couples x_i and
    # x_j; for the pairwise barriers h_i = p_{i+1} - p_i - d_min these are the
    # consecutive pairs.  Without this term Kruskal satisfies the connectivity
    # requirement (R2) while discarding the edge on which the CBF tightening
    # depends, and the tightening silently stops working: over the 30
    # collision-capable runs of the weight sweep (N = 4), violations fall from
    # 43 % (w_safety = 0) to a floor of 10 % (w_safety >= 1) and the replayed
    # critical-edge retention rises from 57 % to 90 %, where both saturate.
    # The floor is 2 missed precursors plus 1 run in which the critical edge was
    # itself flagged; a weight buys priority, not a guarantee.
    # Proposition 4 is unaffected: the greedy algorithm returns a minimum-weight
    # basis of a matroid for ANY weight function.

    F_max: int = 2                  # tolerated simultaneous element losses

    # ---- remedy switches (exp9_remedies.py) --------------------------------
    # Every default reproduces the configuration used for Tables 1-2 and
    # Figs. 2-5, so existing results are unchanged.
    force_critical: bool = False    # Remedy 2: critical edges forced by matroid
                                    # contraction (Prop. 5); see select.py
    fallback_tube: str = "drift"    # tube when the critical edge is absent:
                                    # "drift" (invalid, as in the paper) |
                                    # "stop"  (immediate-stop tube, valid)
    always_stop: bool = False       # 'prop' only: every AHEAD neighbour is
                                    # predicted with the immediate-stop tube
                                    # (unconditional Remark 5(b))
    hard_barrier: bool = False      # two-tier constraints (Sec. 5.6(b)):
                                    # h >= 0 hard, decay soft; falls back to
                                    # soft when infeasible (counted)
    freeze_structure: bool = False  # 'prop' only: flagged links/sensors do not
                                    # trigger reselection; structure changes at
                                    # real failures only
    # ---- remedies / baselines (exp9); all default OFF ----
    force_critical: bool = False     # Remedy 2: contract the critical edges (Prop. 5)
    fallback_tube: str = "drift"     # "drift" = nominal-drift (invalid); "stop" = immediate-stop tube
    always_stop: bool = False        # baseline: immediate-stop tube for every forward neighbour
    hard_barrier: bool = False       # two-tier: hard h >= 0 in addition to the softened decay
    freeze_structure: bool = False   # tighten from detection, but reconfigure only at failure