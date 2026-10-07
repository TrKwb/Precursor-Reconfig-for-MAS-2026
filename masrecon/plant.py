# -*- coding: utf-8 -*-
"""1-D point-mass MAS with viscous drag + exogenous element health.

Plant
-----
    p_i^+ = p_i + dt v_i + dt^2/2 u_i ,      v_i^+ = (1 - c dt) v_i + dt u_i

A healthy agent holds v = v_ref with u = c v_ref.  There is NO process
disturbance: the only randomness is the initial-position jitter (x0) and the
diagnostic signals.  Before failure a degrading actuator keeps its full control
authority (health enters the diagnostic signal only), which is Assumption 3 of
Theorem 1.

Failure modes of a dead actuator (`fail_mode`, default 'freeze')
------------------------------------------------------------------
  'freeze' (default, legacy)  the agent stops ABRUPTLY: from the failure step on,
           its position is held and its velocity is zero.  The deceleration is
           unbounded, i.e. OUTSIDE the braking envelope of Lemma 5; this is the
           failure for which a precursor is required.
  'coast'  the input is lost (u = 0) and the agent decelerates under drag only,
           |dv/dt| = c v <= c v_ref < u_max.  INSIDE the braking envelope.
  'brake'  the agent applies maximal admissible braking (v kept >= 0): exactly
           ON the boundary of the braking envelope.
'coast' and 'brake' exist to test the first half of the division of roles:
failures inside the envelope are handled without any precursor.

Diagnostic signals and degradation mechanisms
---------------------------------------------
Each candidate element e carries a health variable eta_e(k) in [0,1] and emits a
scalar diagnostic signal y_e(k).  The health schedule is EXOGENOUS: diagnostic
signals do not depend on the control method, so the comparison is paired and the
risk trace need be computed only once per trial.

Four mechanisms are provided (see `set_mechanism`):

  'ar'    (default) critical slowing down: the AR(1) coefficient rises towards
          unity and the innovation scale grows, with intermittent dropouts.
          NOTE: this drives the AC1 and log-sd features DIRECTLY, so detection
          on this mechanism alone would be close to circular.
  'bias'  a slow bias ramp.  Autocorrelation and variance are UNCHANGED.
  'osc'   an emerging periodic component, rescaled so that the variance is
          UNCHANGED.
  'quant' progressive quantisation / stiction, rescaled so that the variance is
          UNCHANGED.

Severity (`severity`, default 1.0) scales the DEPTH of the 'ar' degradation
(ar_hi - ar_lo, sig_hi - sig_lo, drop_hi) without changing the healthy signal,
so a detector calibrated on healthy data needs no recalibration.  It is the
knob of the SNR sweep (exp10, E4) together with cfg.deg.T_deg.  severity = 1.0
reproduces the legacy signals bit for bit (the random stream is unchanged for
every severity, so different severities are paired).
"""
import numpy as np

FAIL_MODES = ("freeze", "coast", "brake")


class Plant:
    # ------------------------------------------------------------------
    def __init__(self, cfg, elements, failing=(), rng=None, mechanism="ar",
                 mech_amp=1.0, fail_mode="freeze", severity=1.0):
        assert fail_mode in FAIL_MODES, "unknown fail_mode %r" % (fail_mode,)
        assert severity >= 0.0, "severity must be non-negative"
        self.c = cfg
        self.el = elements
        self.N = cfg.plant.N
        self.dt = cfg.plant.dt
        self.K = int(round(cfg.plant.T / self.dt))
        c = cfg.plant.drag
        self.A = np.array([[1.0, self.dt], [0.0, 1.0 - c * self.dt]])
        self.B = np.array([[0.5 * self.dt ** 2], [self.dt]])
        self.rng = rng if rng is not None else np.random.default_rng(cfg.plant.seed)
        self.failing = list(failing)
        self.fail_mode = fail_mode
        self.severity = float(severity)
        self.mech = "ar"
        self.mech_amp = mech_amp
        self._health()
        self._signals()
        if mechanism != "ar":
            self.set_mechanism(mechanism, mech_amp)

    # ---------------- element health ----------------------------------
    def _health(self):
        d = self.c.deg
        self.k_deg = int(round(d.t_deg / self.dt))
        self.k_fail = int(round((d.t_deg + d.T_deg) / self.dt))
        H = np.ones((self.K + 1, len(self.el)))
        for e in self.failing:
            j = self.el.index[e]
            ramp = np.clip((np.arange(self.K + 1) - self.k_deg)
                           / max(self.k_fail - self.k_deg, 1), 0.0, 1.0)
            H[:, j] = 1.0 - ramp
        self.health = H

    # ---------------- baseline diagnostic signals ---------------------
    def _signals(self):
        """Mechanism 'ar': critical slowing down + growing variance + dropouts."""
        d = self.c.deg
        K, M = self.K + 1, len(self.el)
        s = self.severity
        y = np.zeros((K, M))
        z = np.zeros(M)
        for k in range(K):
            hh = self.health[k]
            if s == 1.0:                                     # legacy, bit-identical
                a = d.ar_lo + (d.ar_hi - d.ar_lo) * (1.0 - hh)
                sg = d.sig_lo + (d.sig_hi - d.sig_lo) * (1.0 - hh)
                pdrop = d.drop_hi * (1.0 - hh)
            else:
                a = np.minimum(d.ar_lo + s * (d.ar_hi - d.ar_lo) * (1.0 - hh), 0.999)
                sg = d.sig_lo + s * (d.sig_hi - d.sig_lo) * (1.0 - hh)
                pdrop = np.minimum(s * d.drop_hi * (1.0 - hh), 1.0)
            z = a * z + sg * np.sqrt(np.maximum(1.0 - a ** 2, 1e-3)) \
                * self.rng.standard_normal(M)
            drop = self.rng.random(M) < pdrop
            y[k] = np.where(drop, 0.0, z)
        self.y = y

    # ---------------- alternative degradation mechanisms --------------
    def set_mechanism(self, mech="ar", amp=1.0):
        """Regenerate the diagnostic signals under an alternative mechanism.

        For 'bias', 'osc' and 'quant' the AR(1) coefficient and the innovation
        scale are held at their HEALTHY values, and the post-degradation signal
        is rescaled to the healthy standard deviation where necessary, so that
        neither AC1 nor log-sd carries the degradation.  The detector must then
        rely on PE / DET / LAM.
        """
        self.mech = mech
        self.mech_amp = amp
        if mech == "ar":
            self._signals()
            return

        K, M = self.K + 1, len(self.el)
        d = self.c.deg
        a, sg = d.ar_lo, d.sig_lo              # FIXED at the healthy values
        z = np.zeros(M)
        y = np.zeros((K, M))
        for k in range(K):
            z = a * z + sg * np.sqrt(1.0 - a ** 2) * self.rng.standard_normal(M)
            y[k] = z
        dg = 1.0 - self.health                 # degradation depth in [0, 1]

        if mech == "bias":
            # Slow bias ramp: mean shifts, autocorrelation and variance unchanged.
            y = y + 2.2 * amp * dg

        elif mech == "osc":
            # Emerging periodic component; variance restored by rescaling.
            t = np.arange(K)[:, None] * self.dt
            y = y + 2.0 * amp * dg * np.sin(2.0 * np.pi * 1.7 * t)
            self._rescale_to_healthy_sd(y)

        elif mech == "quant":
            # Progressive quantisation / stiction; variance restored by rescaling.
            for k in range(K):
                q = 0.05 + 1.6 * amp * dg[k]
                y[k] = np.round(y[k] / np.maximum(q, 1e-6)) * q
            self._rescale_to_healthy_sd(y)

        else:
            raise ValueError("unknown mechanism: %r "
                             "(expected 'ar', 'bias', 'osc' or 'quant')" % mech)
        self.y = y

    def _rescale_to_healthy_sd(self, y):
        """In-place: rescale the post-onset segment of each degrading channel so
        that its standard deviation matches the pre-onset (healthy) value."""
        for j in self.failing:
            jj = self.el.index[j] if not isinstance(j, (int, np.integer)) else j
            s0 = y[:self.k_deg, jj].std()
            s1 = y[self.k_deg:, jj].std()
            if s1 > 1e-9 and s0 > 1e-9:
                y[self.k_deg:, jj] *= (s0 / s1)

    # ---------------- interface ---------------------------------------
    def is_dead(self, e, k):
        return self.health[k, self.el.index[e]] <= 1e-9

    def alive_elements(self, k):
        return [e for e in self.el.all if not self.is_dead(e, k)]

    def fail_info(self):
        """Ground truth for analysis scripts (never used by a controller)."""
        return dict(failing=list(self.failing), k_deg=self.k_deg,
                    k_fail=self.k_fail, fail_mode=self.fail_mode,
                    severity=self.severity, mechanism=self.mech)

    def x0(self):
        p = np.arange(self.N) * self.c.plant.spacing
        p = p + self.rng.uniform(-self.c.plant.p0_jitter,
                                 self.c.plant.p0_jitter, self.N)
        return np.stack([p, np.full(self.N, self.c.plant.v_ref)], axis=1)

    def p_ref(self, k):
        return (np.arange(self.N) * self.c.plant.spacing
                + self.c.plant.v_ref * k * self.dt)

    def _u_brake(self, v):
        """Maximal admissible braking that keeps v >= 0 (used by 'brake')."""
        a = 1.0 - self.c.plant.drag * self.dt
        umax = None
        for sec, nm in (("plant", "u_max"), ("ctrl", "u_max"), ("plant", "umax"), ("ctrl", "umax")):
            umax = getattr(getattr(self.c, sec, None), nm, None)
            if umax is not None:
                break
        if umax is None:
            raise AttributeError("u_max not found in cfg.plant / cfg.ctrl")
        v = max(float(v), 0.0)
        return -umax if a * v - self.dt * umax >= 0 else -a * v / self.dt

    def step(self, x, u, k=None):
        if k is not None and self.fail_mode != "freeze":
            u = np.array(u, dtype=float)
            for i in range(self.N):
                if self.is_dead(("u", i, i), k):
                    u[i] = 0.0 if self.fail_mode == "coast" else self._u_brake(x[i, 1])
        xn = x @ self.A.T + u[:, None] * self.B.T
        if k is not None and self.fail_mode == "freeze":
            for i in range(self.N):
                if self.is_dead(("u", i, i), k):
                    xn[i, 0] = x[i, 0]
                    xn[i, 1] = 0.0                  # abrupt stop
        return xn