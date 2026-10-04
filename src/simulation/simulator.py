"""
Reproducible trajectory simulator for Layer-A dataset generation.

Every simulated run carries the full metadata block required by the
implementation spec (run_id, random_seed, fault_class, fault_channel,
fault_family, fault_severity, fault_onset, duration, gamma,
sensor_configuration, operating_point, noise configuration, structural
status, d_cap, ensemble condition ID) so downstream ML work never has to
reverse-engineer what generated a trajectory, and so no simulator-only
variable can silently leak into a model's actual input features. The
metadata block is exported separately from the observed trajectory, so
callers can be explicit about which columns are model-visible inputs and
which are ground-truth/audit-only metadata.

Fault temporal family: STEP_WITH_ONSET, persistent to the end of the fixed
L-sample observation window once triggered. Duration is not separately
varied in this release -- this is the standing case for leak/degradation-
type faults; a variable-duration extension is a known future direction,
not something already implemented here.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from ..plant import build_model, diff_system_matrices, DEFAULT_KGAIN
from ..faults import PROCESS_FAULTS
from ..structural import structural_pair
from ..statistical.point_astar import _severity_anchor
from ..statistical.ensemble_astar import delayed_step_response


@dataclass
class RunMetadata:
    run_id: str
    condition_id: str
    random_seed: int
    fault_class: str            # "process" | "sensor" | "healthy"
    fault_channel: str          # process_fault_id, "sensor_h{idx}", or "none"
    fault_family: str           # "step_with_onset" | "none"
    fault_severity_pct: Optional[float]
    fault_onset: Optional[int]
    duration: str                # "persistent_to_window_end" (fixed in this release) | "none"
    gamma: float
    sensor_configuration: int    # ny: 2 (partial) or 4 (full)
    operating_point: str         # "swept_gamma_model" (see plant/quadtank.py build_model docstring)
    sigma_h: float
    sigma_v: float
    structural_verdict: str
    structural_mechanism: str
    d_cap: int
    ensemble_condition_id: str

    def as_dict(self):
        return asdict(self)


def simulate_run(condition, run_id: str, seed: int, fault_class: str,
                  severity_pct: Optional[float] = None, onset: Optional[int] = None,
                  sigma_h: float = 0.0, sigma_v: float = 0.3, L: int = 8,
                  Kgain: float = DEFAULT_KGAIN):
    """Simulate one labeled trajectory for the given ExperimentCondition.

    fault_class: "process" (the condition's process_fault_id fires),
                 "sensor"  (a fault on the condition's sensor_idx fires),
                 "healthy" (no fault; severity_pct/onset ignored).

    Returns (Y, meta): Y is the (L, ny) observed-output trajectory (the
    only thing a downstream diagnostic model may see); meta is a
    RunMetadata instance (ground truth / audit-only, exported separately).
    """
    if fault_class not in ("process", "sensor", "healthy"):
        raise ValueError("fault_class must be 'process', 'sensor', or 'healthy'")
    rng = np.random.default_rng(seed)

    model = build_model(gamma=condition.gamma, Kgain=Kgain)
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, 0.0, condition.sensor_idx, condition.ny)
    ny = C.shape[0]

    struct = structural_pair(condition.process_fault_id, condition.sensor_idx,
                              condition.ny, condition.gamma, L=8, Kgain=Kgain)

    if fault_class == "healthy":
        mu = np.zeros(L * ny)
        fault_channel, fault_family = "none", "none"
        severity_pct_out, onset_out, duration = None, None, "none"
    else:
        if severity_pct is None:
            a, b = np.log(1.0), np.log(30.0)  # declared prior support, config.SEVERITY_PRIOR
            severity_pct = float(np.exp(rng.uniform(a, b)))
        if onset is None:
            onset = int(rng.choice([0, 1, 2, 3]))  # declared prior support, config.ONSET_PRIOR
        if fault_class == "process":
            rec = PROCESS_FAULTS[condition.process_fault_id]
            anchor = _severity_anchor(condition.process_fault_id, model, condition.sensor_idx)[0]
            Bcol, Dcol = rec.direction_fn(model), np.zeros(ny)
            fault_channel, fault_family = condition.process_fault_id, "step_with_onset"
        else:
            anchor = _severity_anchor(condition.process_fault_id, model, condition.sensor_idx)[1]
            Bcol, Dcol = Edelta[:, 1], Ddelta[:, 1]
            fault_channel, fault_family = f"sensor_h{condition.sensor_idx + 1}", "step_with_onset"
        alpha = (severity_pct / 100.0) * anchor
        mu = alpha * delayed_step_response(Acl, Bcol, C, Dcol, L, onset)
        severity_pct_out, onset_out, duration = severity_pct, onset, "persistent_to_window_end"

    from ..statistical.point_astar import state_noise_toeplitz
    G = state_noise_toeplitz(Acl, C, L)
    Sigma = (sigma_h ** 2) * (G @ G.T) + (sigma_v ** 2) * np.eye(L * ny)
    y_flat = rng.multivariate_normal(mu, Sigma)
    Y = y_flat.reshape(L, ny)

    meta = RunMetadata(
        run_id=run_id, condition_id=condition.condition_id, random_seed=seed,
        fault_class=fault_class, fault_channel=fault_channel, fault_family=fault_family,
        fault_severity_pct=severity_pct_out, fault_onset=onset_out, duration=duration,
        gamma=condition.gamma, sensor_configuration=condition.ny,
        operating_point="swept_gamma_model", sigma_h=sigma_h, sigma_v=sigma_v,
        structural_verdict=struct.verdict.value, structural_mechanism=struct.mechanism.value,
        d_cap=struct.d_cap, ensemble_condition_id=condition.condition_id,
    )
    return Y, meta
