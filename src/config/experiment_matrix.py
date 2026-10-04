"""
Layer-A experiment matrix: the definitive list of which
(process-fault, sensor-fault, sensing-configuration) pairs constitute the
paper's contrasting cases. Every later stage (structural table,
statistical tables, dataset generator) reads this list rather than
re-deriving it.

L_PRIMARY = 8 is the single explicit primary horizon for every reported
structural/statistical quantity; L_ROBUST is shown separately, never
substituted into a primary row.

gamma is treated as a robustness / operating-condition variable, not the
central theoretical axis: each pair's gamma is either the worst-case-for-
separation gamma* found by the structural sweep
(for AMBIGUOUS pairs, this is where d_cap is maximized) or a representative
mid-range value (for SEPARATED controls, where d_cap=0 across the whole
sweep and gamma choice is not adversarial).
"""
from dataclasses import dataclass, field

L_PRIMARY = 8
L_ROBUST = (6, 12, 24)
GAMMA_GRID_19PT = tuple(0.05 + 0.90 * i / 18 for i in range(19))  # matches np.linspace(0.05,0.95,19)

# Declared statistical priors -- severity is CONTINUOUS log-uniform, approximated by
# Gauss-Legendre quadrature; onset is DISCRETE uniform, exact.
SEVERITY_PRIOR = dict(distribution="continuous_log_uniform", pct_lo=1.0, pct_hi=30.0,
                       quadrature_rule="gauss_legendre", n_severity_frozen=31,
                       convergence_note="the AGGREGATE ensemble-A* value converges (stays "
                                        "within the declared 1e-3 tolerance of the n=127 "
                                        "value) from n_severity=7 onward for all 9 "
                                        "conditions tested; this is NOT the same as the "
                                        "individual Bayes decisions being unchanged -- between "
                                        "0 and 4 of 6,000 fixed test-point decisions actually "
                                        "flip going from n=7 to n=15 (observed per-condition "
                                        "range), and these largely cancel in the aggregate "
                                        "rather than genuinely vanishing (a prior write-up "
                                        "overclaimed 'zero decisions flip'; corrected here). "
                                        "n_severity=31 is therefore "
                                        "used as the conservative PRODUCTION DEFAULT everywhere "
                                        "in this codebase (statistical.DEFAULT_N_SEVERITY), not "
                                        "n=7, even though n=7's aggregate accuracy was already "
                                        "adequate -- see experiments/run_ensemble_astar.py "
                                        "output for the full per-resolution flip-count table")
ONSET_PRIOR = dict(distribution="discrete_uniform", support=(0, 1, 2, 3), note="exact, not a "
                    "quadrature approximation of anything continuous")

# Declared noise operating point used for dataset/table generation
# (also swept separately in the parameter map).
NOISE_REFERENCE = dict(sigma_h=0.0, sigma_v=0.3, units="sigma_h: cm (state), sigma_v: V (output)")


@dataclass(frozen=True)
class ExperimentCondition:
    condition_id: str
    process_fault_id: str
    sensor_idx: int
    ny: int                      # sensing configuration: 2=partial (h1,h2), 4=full (h1..h4)
    gamma: float
    gamma_role: str              # "worst_case_for_ambiguity" | "representative_midrange"
    expected_verdict: str        # AMBIGUOUS | SEPARATED -- re-verified by tests
    expected_mechanism: str      # DIMENSION_FORCED | EXCESS_FAULT_SPECIFIC | NOT_APPLICABLE
    role: str                    # "ambiguous_case" | "separated_control"
    notes: str = ""


# The contrasting set used throughout this codebase's structural
# and statistical analysis.
EXPERIMENT_MATRIX = [
    ExperimentCondition(
        condition_id="pump1_h1_partial",
        process_fault_id="pump1", sensor_idx=0, ny=2, gamma=0.05,
        gamma_role="worst_case_for_ambiguity",
        expected_verdict="AMBIGUOUS", expected_mechanism="DIMENSION_FORCED",
        role="ambiguous_case",
        notes="d_cap=1=floor at L=8, robust across L in {6,12,24}. Eliminable by adding sensors.",
    ),
    ExperimentCondition(
        condition_id="pump1_h1_full",
        process_fault_id="pump1", sensor_idx=0, ny=4, gamma=0.50,
        gamma_role="representative_midrange",
        expected_verdict="SEPARATED", expected_mechanism="NOT_APPLICABLE",
        role="separated_control",
        notes="d_cap=0 at every tested gamma and horizon. Direct sensor-count negative control for pump1_h1_partial.",
    ),
    ExperimentCondition(
        condition_id="pump2_h1_partial",
        process_fault_id="pump2", sensor_idx=0, ny=2, gamma=0.05,
        gamma_role="worst_case_for_ambiguity",
        expected_verdict="AMBIGUOUS", expected_mechanism="DIMENSION_FORCED",
        role="ambiguous_case",
        notes="Mirrors pump1_h1_partial under the actuator symmetry.",
    ),
    ExperimentCondition(
        condition_id="pump2_h1_full",
        process_fault_id="pump2", sensor_idx=0, ny=4, gamma=0.50,
        gamma_role="representative_midrange",
        expected_verdict="SEPARATED", expected_mechanism="NOT_APPLICABLE",
        role="separated_control",
        notes="Sensor-count negative control for pump2_h1_partial.",
    ),
    ExperimentCondition(
        condition_id="leak_tank1_h1_partial",
        process_fault_id="leak_tank1", sensor_idx=0, ny=2, gamma=0.05,
        gamma_role="worst_case_for_ambiguity",
        expected_verdict="AMBIGUOUS", expected_mechanism="EXCESS_FAULT_SPECIFIC",
        role="ambiguous_case",
        notes="Unbounded excess (d_cap grows with L: 4,10,22 at L=6,12,24) -- sink-node state, "
              "NOT eliminable by sensor count.",
    ),
    ExperimentCondition(
        condition_id="leak_tank1_h2_partial",
        process_fault_id="leak_tank1", sensor_idx=1, ny=2, gamma=0.50,
        gamma_role="representative_midrange",
        expected_verdict="SEPARATED", expected_mechanism="NOT_APPLICABLE",
        role="separated_control",
        notes="Cross-sensor separated control for leak_tank1 (own-sensor h1 case is ambiguous; "
              "cross-sensor h2 is separated).",
    ),
    ExperimentCondition(
        condition_id="leak_tank3_h1_partial",
        process_fault_id="leak_tank3", sensor_idx=0, ny=2, gamma=0.05,
        gamma_role="worst_case_for_ambiguity",
        expected_verdict="AMBIGUOUS", expected_mechanism="EXCESS_FAULT_SPECIFIC",
        role="ambiguous_case",
        notes="Forward-coupled state, ambiguous under partial monitoring against the cross-tank sensor.",
    ),
    ExperimentCondition(
        condition_id="leak_tank3_h1_full",
        process_fault_id="leak_tank3", sensor_idx=0, ny=4, gamma=0.50,
        gamma_role="representative_midrange",
        expected_verdict="SEPARATED", expected_mechanism="NOT_APPLICABLE",
        role="separated_control",
        notes="Sensor-count negative control for leak_tank3_h1_partial.",
    ),
    ExperimentCondition(
        condition_id="leak_tank3_h3_full_own",
        process_fault_id="leak_tank3", sensor_idx=2, ny=4, gamma=0.05,
        gamma_role="worst_case_for_ambiguity",
        expected_verdict="AMBIGUOUS", expected_mechanism="EXCESS_FAULT_SPECIFIC",
        role="ambiguous_case",
        notes="Ambiguous against its OWN direct sensor even under full monitoring (bounded excess, "
              "d_cap=1 robust across L) -- the sharpest example that adding sensors does not always help.",
    ),
]


def condition_ids():
    return [c.condition_id for c in EXPERIMENT_MATRIX]


def ambiguous_conditions():
    return [c for c in EXPERIMENT_MATRIX if c.role == "ambiguous_case"]


def separated_conditions():
    return [c for c in EXPERIMENT_MATRIX if c.role == "separated_control"]


def matrix_as_dicts():
    return [dict(condition_id=c.condition_id, process_fault_id=c.process_fault_id,
                 sensor_idx=c.sensor_idx, ny=c.ny, gamma=c.gamma, gamma_role=c.gamma_role,
                 expected_verdict=c.expected_verdict, expected_mechanism=c.expected_mechanism,
                 role=c.role, notes=c.notes, L_primary=L_PRIMARY, L_robust=list(L_ROBUST))
            for c in EXPERIMENT_MATRIX]
