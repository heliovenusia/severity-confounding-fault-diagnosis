"""
Quadruple-tank plant implementation for Layer-A.

Implemented as a standalone module in this source tree, unchanged in its
numerics, and documented with explicit parameter provenance (see
parameters.py).

Responsibilities:
  - nonlinear equilibrium (steady_state)
  - continuous-time linearization about that equilibrium (linearize)
  - zero-order-hold discretization (discretize)
  - full discrete closed-loop model assembly, partial (ny=2) and full
    (ny=4) sensing configurations, decentralized static output-feedback
    controller (build_model, check_stability)
  - fault-direction utilities and the closed-loop "difference system"
    used by the structural and statistical engines (Ep_dir,
    sensor_config, diff_system_matrices)
  - finite-horizon step-response utilities, with and without onset delay
    (step_response_shape, delayed_step_response_shape)

Nothing in this module performs a structural discernibility test or a
statistical Bayes computation -- those live in src/structural and
src/statistical respectively, and both import this module rather than
duplicating the plant equations.
"""
import numpy as np
import control

from .parameters import PARAMETER_MANIFEST, Provenance

np.set_printoptions(precision=5, suppress=True)

_P = {p.name: p.value for p in PARAMETER_MANIFEST}

# Physical constants (see parameters.py for provenance of every value below)
g = _P["g"]
A1 = A3 = _P["A1"]
A2 = A4 = _P["A2"]
a1 = a3 = _P["a1"]
a2 = a4 = _P["a2"]
kc = _P["kc"]

# Two literature-style reference operating points, used only as sanity-check
# targets (see steady_state() docstring) -- not the source of any reported
# structural or statistical result, which always recomputes h0 self-consistently.
REF_MIN_PHASE = dict(gamma1=_P["gamma1_minphase"], gamma2=_P["gamma2_minphase"],
                      k1=_P["k1_minphase"], k2=_P["k2_minphase"], v0=_P["v0_minphase"])
REF_NONMIN_PHASE = dict(gamma1=_P["gamma1_nonminphase"], gamma2=_P["gamma2_nonminphase"],
                         k1=_P["k1_nonminphase"], k2=_P["k2_nonminphase"], v0=_P["v0_nonminphase"])

DEFAULT_K1 = _P["k1_swept"]
DEFAULT_K2 = _P["k2_swept"]
DEFAULT_V0 = _P["v0_swept"]
DEFAULT_TS = _P["Ts"]
DEFAULT_KGAIN = _P["Kgain"]


def steady_state(gamma1, gamma2, k1, k2, v1, v2):
    """Closed-form nonlinear steady state (h1..h4) for a given valve split
    and pump voltage pair, from the quadruple-tank balance equations."""
    h3 = ((1 - gamma2) * k2 * v2 / a3) ** 2 / (2 * g)
    h4 = ((1 - gamma1) * k1 * v1 / a4) ** 2 / (2 * g)
    h1 = ((gamma1 * k1 * v1 + (1 - gamma2) * k2 * v2) / a1) ** 2 / (2 * g)
    h2 = ((gamma2 * k2 * v2 + (1 - gamma1) * k1 * v1) / a2) ** 2 / (2 * g)
    return np.array([h1, h2, h3, h4])


def linearize(gamma1, gamma2, k1, k2, h0):
    """Continuous-time linearization about the equilibrium h0 = (h1,h2,h3,h4)."""
    h1, h2, h3, h4 = h0
    T1 = (A1 / a1) * np.sqrt(2 * h1 / g)
    T2 = (A2 / a2) * np.sqrt(2 * h2 / g)
    T3 = (A3 / a3) * np.sqrt(2 * h3 / g)
    T4 = (A4 / a4) * np.sqrt(2 * h4 / g)
    Ac = np.array([
        [-1 / T1, 0, A3 / (A1 * T3), 0],
        [0, -1 / T2, 0, A4 / (A2 * T4)],
        [0, 0, -1 / T3, 0],
        [0, 0, 0, -1 / T4],
    ])
    Bc = np.array([
        [gamma1 * k1 / A1, 0],
        [0, gamma2 * k2 / A2],
        [0, (1 - gamma2) * k2 / A3],
        [(1 - gamma1) * k1 / A4, 0],
    ])
    return Ac, Bc, (T1, T2, T3, T4)


def discretize(Ac, Baug_c, Ts):
    """Zero-order-hold discretization via python-control."""
    n = Ac.shape[0]
    m = Baug_c.shape[1]
    sysc = control.ss(Ac, Baug_c, np.eye(n), np.zeros((n, m)))
    sysd = control.c2d(sysc, Ts, method="zoh")
    return np.asarray(sysd.A), np.asarray(sysd.B)


def build_model(gamma, k1=DEFAULT_K1, k2=DEFAULT_K2, v0=DEFAULT_V0,
                 Ts=DEFAULT_TS, Kgain=DEFAULT_KGAIN):
    """Assemble the full discrete, closed-loop quadruple-tank model at a
    given valve-split gamma (gamma1=gamma2=gamma, the swept-model
    convention used throughout this codebase)."""
    h0 = steady_state(gamma, gamma, k1, k2, v0, v0)
    Ac, Bc, Ts_consts = linearize(gamma, gamma, k1, k2, h0)
    n = 4
    # unit-norm pump fault directions, used by structural/statistical engines
    b1 = Bc[:, 0] / np.linalg.norm(Bc[:, 0])
    b2 = Bc[:, 1] / np.linalg.norm(Bc[:, 1])
    Baug_c = np.column_stack([Bc, b1, b2])
    Ad, Baug_d = discretize(Ac, Baug_c, Ts)
    Bd = Baug_d[:, :2]
    b1d = Baug_d[:, 2]
    b2d = Baug_d[:, 3]
    C2 = np.array([[kc, 0, 0, 0], [0, kc, 0, 0]])  # partial sensing: h1,h2
    C4 = kc * np.eye(4)                             # full sensing: h1,h2,h3,h4
    K = Kgain * np.eye(2)  # decentralized static proportional output feedback, u = K y + r
    return dict(h0=h0, Ac=Ac, Bc=Bc, Ad=Ad, Bd=Bd, b1d=b1d, b2d=b2d,
                C2=C2, C4=C4, K=K, Ts=Ts, Ts_consts=Ts_consts,
                gamma=gamma, k1=k1, k2=k2, v0=v0, Kgain=Kgain)


def check_stability(model):
    """Closed-loop eigenvalues under the CONTROL-relevant 2-sensor (h1,h2)
    pairing. The controller always uses this pairing regardless of how many
    sensors the diagnostic/discernibility layer is allowed to observe --
    the "sensor configuration" axis varies what the structural/statistical
    test can see, never what stabilizes the plant."""
    Acl = model["Ad"] + model["Bd"] @ model["K"] @ model["C2"]
    eig = np.linalg.eigvals(Acl)
    return eig, float(np.max(np.abs(eig)))


def Ep_dir(model, theta):
    return np.cos(theta) * model["b1d"] + np.sin(theta) * model["b2d"]


def sensor_config(model, ny):
    if ny not in (2, 4):
        raise ValueError("ny must be 2 (partial: h1,h2) or 4 (full: h1..h4)")
    return model["C2"] if ny == 2 else model["C4"]


def diff_system_matrices(model, theta, sensor_idx, ny):
    """Build the closed-loop 'difference system' (Acl, Edelta, C, Ddelta)
    used by every downstream structural/statistical computation. Edelta's
    two columns are: [process-fault direction, sensor-fault direction on
    channel sensor_idx]; Ddelta's two columns are the matching direct
    feedthrough (zero for the process column, -e_{sensor_idx} for the
    sensor column). C is the DIAGNOSTIC output map (ny sensors, which may
    exceed the 2 sensors the controller itself uses)."""
    C = sensor_config(model, ny)
    Acl = model["Ad"] + model["Bd"] @ model["K"] @ model["C2"]
    Ep = Ep_dir(model, theta)
    Es = np.zeros(ny)
    Es[sensor_idx] = 1.0
    # A sensor fault only perturbs the controller if the faulty sensor is
    # one of the two control-relevant sensors (sensor_idx < 2); a fault on
    # h3/h4 under full (ny=4) sensing does not feed back into u at all.
    if sensor_idx < 2:
        Es_ctrl = np.zeros(2)
        Es_ctrl[sensor_idx] = 1.0
    else:
        Es_ctrl = np.zeros(2)
    Edelta = np.column_stack([Ep, -model["Bd"] @ model["K"] @ Es_ctrl])
    Ddelta = np.column_stack([np.zeros(ny), -Es])
    return Acl, Edelta, C, Ddelta


def invariant_zeros(Acl, Edelta, C, Ddelta):
    sys = control.ss(Acl, Edelta, C, Ddelta, dt=1)
    try:
        z = control.zeros(sys)
    except Exception:
        z = np.array([])
    return z


def step_response_shape(Acl, Bcol, Ccol, Dcol, L):
    """Finite-horizon (length L) stacked output response to a unit step in
    a single scalar fault channel, starting from zero state."""
    n = Acl.shape[0]
    x = np.zeros(n)
    ys = []
    for _ in range(L):
        y = Ccol @ x + Dcol * 1.0
        ys.append(y)
        x = Acl @ x + Bcol * 1.0
    return np.concatenate(ys)


def delayed_step_response_shape(Acl, Bcol, Ccol, Dcol, L, delay):
    """Same as step_response_shape, but the fault channel switches on only
    at sample t=delay (0 before). Output is still stacked over the SAME
    fixed observation window t=0..L-1."""
    n = Acl.shape[0]
    x = np.zeros(n)
    ys = []
    for t in range(L):
        active = 1.0 if t >= delay else 0.0
        y = Ccol @ x + Dcol * active
        ys.append(y)
        x = Acl @ x + Bcol * active
    return np.concatenate(ys)


def misalignment(model, theta, sensor_idx, ny, L):
    """Cosine-based geometric shape proxy only. NOT a statistical-difficulty
    measure -- retained purely as a diagnostic/geometric quantity. The
    Bayes-relevant statistical quantities are Delta/A* in
    src/statistical/point_astar.py, never this cosine."""
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, theta, sensor_idx, ny)
    up = step_response_shape(Acl, Edelta[:, 0], C, Ddelta[:, 0], L)
    us = step_response_shape(Acl, Edelta[:, 1], C, Ddelta[:, 1], L)
    nu, ns = np.linalg.norm(up), np.linalg.norm(us)
    if nu < 1e-12 or ns < 1e-12:
        return 1.0, np.nan
    cosang = np.clip(np.dot(up, us) / (nu * ns), -1, 1)
    return 1 - abs(cosang), cosang
