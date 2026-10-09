# Benchmark of Sec. 6.1. Failure model: freeze (v := 0 at k_f), i.e. outside the braking envelope.
dt, c, umax, vref, delta, dmin, gam, Nh = 0.05, 1.0, 3.6, 3.0, 1.9, 1.2, 0.25, 10
a = 1 - c * dt

def step(p, v, u):
    return p + dt * v + 0.5 * dt * dt * u, a * v + dt * u

def brake(v):                      # max braking, clipped so that v_next >= 0 (Assumption 2)
    u = -umax
    return -a * v / dt if a * v + dt * u < 0 else u

def max_brake_traj(p, v, n):
    out = [p]
    for _ in range(n):
        p, v = step(p, v, brake(v)); out.append(p)
    return out

def stop_feasible(g, v):
    """Leader frozen at gap g; follower at speed v. Max braking minimises the follower position at every
    future step, hence is the best plan; check the DCBF decay h(t+1) >= (1-gam) h(t) until rest."""
    h, p = g - dmin, 0.0
    if h < 0: return False
    while v > 1e-12:
        p, v = step(p, v, brake(v)); hn = g - p - dmin
        if hn < (1 - gam) * h - 1e-12: return False
        h = hn
    return True

def env_plan_feasible(g, vf, vl, u0, n=60):
    """Envelope tube of the leader; follower applies u0 then max-brakes."""
    L = max_brake_traj(g, vl, n); p, v = 0.0, vf; h = L[0] - dmin
    for t in range(n):
        p, v = step(p, v, u0 if t == 0 else brake(v)); hn = L[t + 1] - p - dmin
        if hn < (1 - gam) * h - 1e-12: return False
        h = hn
    return True

def lstar(d0):
    g, v = d0, vref
    for k in range(400):
        if stop_feasible(g, v): return k
        pf, v = step(0.0, v, brake(v)); g += dt * vref - pf
    return None

if __name__ == "__main__":
    p, v = 0.0, vref
    while v > 1e-12: p, v = step(p, v, brake(v))
    print("braking distance %.4f m" % p)
    print("L* (delta=1.9) = %d steps" % lstar(delta))
    print("nominal cruise feasible under envelope tube:", env_plan_feasible(delta, vref, vref, c * vref))
    print("nominal feasible under stop tube (max brake first):", stop_feasible(delta, vref))