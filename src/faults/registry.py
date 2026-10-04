"""
Central fault-class registry for Layer-A (quadruple-tank).

This module defines what a fault channel IS and how it
may legitimately be exercised. It replaces earlier ad hoc per-script
channel dictionaries that defined the same information separately and
inconsistently in several places.

Two distinct notions are deliberately kept separate and must never be
conflated:

  STRUCTURAL CHANNEL
      The object the Sheikhi discernibility theorem actually quantifies
      over: an ARBITRARY additive fault sequence f_{k,L} in R^L injected
      through a fixed direction/entry point, with no restriction on its
      temporal shape. This is what src/structural uses. It is never
      called "bias" -- an unrestricted additive sequence is strictly more
      general than a bias (constant offset), and the theorem's
      indiscernibility claim is a claim about this whole admissible
      L-dimensional signal space, not about any one member of it.

  OPERATIONAL FAULT FAMILY
      The restricted, physically motivated temporal family actually
      sampled by the statistical/simulation/dataset layers: a step (with
      random onset within the horizon) of random severity drawn from a
      declared prior, optionally with a duration parameter. This is a
      single-parameter-family SUBSET of the structural channel's full
      admissible space, used because it is what a real fault plausibly
      looks like and what a classifier will actually be trained/tested
      against -- it is NOT what the structural theorem's existential
      claim ranges over.

Every entry below is used for BOTH analyses (used_for = ("structural",
"operational")) unless noted otherwise, because in this codebase every
selected Layer-A channel is exercised both ways: the structural engine
asks "does there exist ANY admissible fault sequence on this channel
that collides with the other channel's admissible subspace," while the
operational/statistical engine asks "under the DECLARED step-fault
family, how separable are these two channels in practice."
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import numpy as np


class FaultKind(Enum):
    PROCESS_ACTUATOR = "process_actuator"   # pump degradation/offset fault
    PROCESS_LEAK = "process_leak"           # tank leak (state-entry fault)
    SENSOR = "sensor"                       # sensor fault


class TemporalFamily(Enum):
    ARBITRARY_ADDITIVE = "arbitrary_additive_sequence"   # structural theorem's admissible space
    STEP_WITH_ONSET = "step_with_random_onset"           # operational family sampled downstream


class SeverityAnchor(Enum):
    NOMINAL_ACTUATOR_VOLTAGE = "pct_of_nominal_pump_voltage_v0"
    NOMINAL_SENSOR_VOLTAGE = "pct_of_nominal_sensor_voltage_kc_h0"
    NOMINAL_LEVEL_TYPE_C = "pct_of_nominal_level_h0_type_c_output_effect"


@dataclass(frozen=True)
class FaultRecord:
    identifier: str
    kind: FaultKind
    physical_interpretation: str
    math_entry_point: str          # how it enters the model: B-column, state-entry direction, or C-row feedthrough
    units: str
    direction_fn: Callable         # (model) -> direction vector, exactly as used by diff_system_matrices/structural code
    temporal_family_structural: TemporalFamily
    temporal_family_operational: TemporalFamily
    severity_anchor: SeverityAnchor
    severity_param_name: str       # e.g. "pct_p", "pct_s", "pct_leak"
    onset_distribution: str
    used_for: tuple = ("structural", "operational")
    normalization_type: str = "B"  # A=primary physical range, B=%-of-nominal, C=output-effect (informal)
    notes: str = ""


def _pump1_dir(model):
    return model["Bd"][:, 0]


def _pump2_dir(model):
    return model["Bd"][:, 1]


def _leak_dir(tank_index):
    def fn(model):
        v = np.zeros(4)
        v[tank_index] = 1.0
        return v
    return fn


PROCESS_FAULTS = {
    "pump1": FaultRecord(
        identifier="pump1",
        kind=FaultKind.PROCESS_ACTUATOR,
        physical_interpretation="Pump-1 actuator fault (degraded/offset commanded flow into tanks 1 and 4).",
        math_entry_point="B_d[:,0] (true discretized actuator input column)",
        units="V (equivalent commanded-voltage error)",
        direction_fn=_pump1_dir,
        temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
        temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
        severity_anchor=SeverityAnchor.NOMINAL_ACTUATOR_VOLTAGE,
        severity_param_name="pct_p",
        onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
        normalization_type="B",
        notes="Type B normalization: alpha_p = (pct_p/100)*v0, injected through the REAL actuator "
              "channel B_d[:,0], so this is dimensionally exact (a genuine actuator-voltage error), "
              "not an ad hoc scaling.",
    ),
    "pump2": FaultRecord(
        identifier="pump2",
        kind=FaultKind.PROCESS_ACTUATOR,
        physical_interpretation="Pump-2 actuator fault (degraded/offset commanded flow into tanks 2 and 3).",
        math_entry_point="B_d[:,1]",
        units="V (equivalent commanded-voltage error)",
        direction_fn=_pump2_dir,
        temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
        temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
        severity_anchor=SeverityAnchor.NOMINAL_ACTUATOR_VOLTAGE,
        severity_param_name="pct_p",
        onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
        normalization_type="B",
        notes="Same construction as pump1, mirrored actuator.",
    ),
    "leak_tank1": FaultRecord(
        identifier="leak_tank1",
        kind=FaultKind.PROCESS_LEAK,
        physical_interpretation="Leak in tank 1 (sink node: no forward A-coupling into another tank's "
                                 "level equation).",
        math_entry_point="state-entry direction e1 = [1,0,0,0] added each step",
        units="cm (Type C: informal level-jump surrogate for a true flow fault, see notes)",
        direction_fn=_leak_dir(0),
        temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
        temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
        severity_anchor=SeverityAnchor.NOMINAL_LEVEL_TYPE_C,
        severity_param_name="pct_leak",
        onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
        normalization_type="C",
        notes="Type C (output-effect) normalization, labeled informal: a real leak "
              "is a FLOW (cm^3/s) fault, not a direct cm-level jump. No primary source in this "
              "project gives an authoritative leak-flow-severity distribution for this rig, so this "
              "is a controlled mathematical experiment, anchored at least to the tank's own nominal "
              "level, not a bare unitless number. tank1/tank2 are sink nodes for the state dynamics "
              "(no forward A-coupling; confirmed structurally).",
    ),
    "leak_tank2": FaultRecord(
        identifier="leak_tank2",
        kind=FaultKind.PROCESS_LEAK,
        physical_interpretation="Leak in tank 2 (sink node, mirrors leak_tank1 under the tank1<->tank2 "
                                 "model symmetry).",
        math_entry_point="state-entry direction e2 = [0,1,0,0] added each step",
        units="cm (Type C)",
        direction_fn=_leak_dir(1),
        temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
        temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
        severity_anchor=SeverityAnchor.NOMINAL_LEVEL_TYPE_C,
        severity_param_name="pct_leak",
        onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
        normalization_type="C",
        notes="Mirror of leak_tank1.",
    ),
    "leak_tank3": FaultRecord(
        identifier="leak_tank3",
        kind=FaultKind.PROCESS_LEAK,
        physical_interpretation="Leak in tank 3 (forward-coupled: A[0,2] nonzero, feeds into tank 1's "
                                 "level equation).",
        math_entry_point="state-entry direction e3 = [0,0,1,0] added each step",
        units="cm (Type C)",
        direction_fn=_leak_dir(2),
        temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
        temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
        severity_anchor=SeverityAnchor.NOMINAL_LEVEL_TYPE_C,
        severity_param_name="pct_leak",
        onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
        normalization_type="C",
        notes="Forward-coupled state (unlike leak_tank1/2); mirrors leak_tank4 under the "
              "tank3<->tank4 model symmetry.",
    ),
    "leak_tank4": FaultRecord(
        identifier="leak_tank4",
        kind=FaultKind.PROCESS_LEAK,
        physical_interpretation="Leak in tank 4 (forward-coupled: A[1,3] nonzero, feeds into tank 2's "
                                 "level equation).",
        math_entry_point="state-entry direction e4 = [0,0,0,1] added each step",
        units="cm (Type C)",
        direction_fn=_leak_dir(3),
        temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
        temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
        severity_anchor=SeverityAnchor.NOMINAL_LEVEL_TYPE_C,
        severity_param_name="pct_leak",
        onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
        normalization_type="C",
        notes="Mirror of leak_tank3.",
    ),
}

# Sensor faults are parameterized by (sensor_idx, ny) at use-site rather than
# pre-enumerated here, since the direction is a standard basis vector in the
# ny-dimensional output space and ny/sensor_idx are experiment-matrix
# choices, not separate physical channels. This record documents the class.
SENSOR_FAULT_TEMPLATE = FaultRecord(
    identifier="sensor_<idx>",
    kind=FaultKind.SENSOR,
    physical_interpretation="Arbitrary additive fault sequence on measurement channel h_{idx+1} "
                             "(NEVER called 'bias' -- see module docstring). The theorem's admissible "
                             "space is the full L-dimensional additive sequence space on this channel; "
                             "the operational family below is a restricted, physically motivated subset.",
    math_entry_point="direct feedthrough D-column, -e_idx, on the diagnostic output map C",
    units="V (equivalent sensor-reading error)",
    direction_fn=lambda model: None,  # constructed per (sensor_idx, ny) via diff_system_matrices
    temporal_family_structural=TemporalFamily.ARBITRARY_ADDITIVE,
    temporal_family_operational=TemporalFamily.STEP_WITH_ONSET,
    severity_anchor=SeverityAnchor.NOMINAL_SENSOR_VOLTAGE,
    severity_param_name="pct_s",
    onset_distribution="Discrete-uniform{0,1,2,3} within the L=8 observation window",
    normalization_type="B",
    notes="Type B: alpha_s = (pct_s/100)*kc*h0[sensor_tank_idx], anchored to THAT sensor's own "
          "nominal voltage reading -- dimensionally exact. Reserve 'bias'/'drift'/'stuck-at'/'gain' "
          "strictly for a restricted parametric operational family (G2a); this record and its "
          "structural use (G2b) is the unrestricted arbitrary additive sequence, full stop.",
)


def all_process_faults():
    return dict(PROCESS_FAULTS)


def sensor_fault_units():
    return "V"


def registry_as_dicts():
    out = []
    for fid, rec in PROCESS_FAULTS.items():
        out.append(dict(
            identifier=rec.identifier, kind=rec.kind.value,
            physical_interpretation=rec.physical_interpretation,
            math_entry_point=rec.math_entry_point, units=rec.units,
            temporal_family_structural=rec.temporal_family_structural.value,
            temporal_family_operational=rec.temporal_family_operational.value,
            severity_anchor=rec.severity_anchor.value,
            severity_param_name=rec.severity_param_name,
            onset_distribution=rec.onset_distribution,
            used_for=list(rec.used_for), normalization_type=rec.normalization_type,
            notes=rec.notes,
        ))
    out.append(dict(
        identifier=SENSOR_FAULT_TEMPLATE.identifier, kind=SENSOR_FAULT_TEMPLATE.kind.value,
        physical_interpretation=SENSOR_FAULT_TEMPLATE.physical_interpretation,
        math_entry_point=SENSOR_FAULT_TEMPLATE.math_entry_point, units=SENSOR_FAULT_TEMPLATE.units,
        temporal_family_structural=SENSOR_FAULT_TEMPLATE.temporal_family_structural.value,
        temporal_family_operational=SENSOR_FAULT_TEMPLATE.temporal_family_operational.value,
        severity_anchor=SENSOR_FAULT_TEMPLATE.severity_anchor.value,
        severity_param_name=SENSOR_FAULT_TEMPLATE.severity_param_name,
        onset_distribution=SENSOR_FAULT_TEMPLATE.onset_distribution,
        used_for=list(SENSOR_FAULT_TEMPLATE.used_for),
        normalization_type=SENSOR_FAULT_TEMPLATE.normalization_type,
        notes=SENSOR_FAULT_TEMPLATE.notes,
    ))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(registry_as_dicts(), indent=2))
