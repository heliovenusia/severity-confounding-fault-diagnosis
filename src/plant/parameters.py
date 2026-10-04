"""
Parameter manifest for the quadruple-tank Layer-A plant, with an explicit
provenance tag on every numerical value: PRIMARY_SOURCE_VERIFIED,
SECONDARY_SOURCE_VERIFIED, or ASSUMED_DESIGN_CHOICE.

Provenance history (do not silently drop this record):
  - The physical constants originate from Johansson (2000), Table I/II,
    and were initially cross-referenced only against secondary
    derivations reproducing the same model equations, without directly
    confirming them against a source that explicitly cites Johansson.
    Those entries were flagged UNRESOLVED.
  - A secondary source was later located and checked directly:
        Strathprints (University of Strathclyde) full-text PDF,
        "Multivariable Predictive PID Control for Quadruple Tank",
        https://strathprints.strath.ac.uk/42140/1/quad_tank_rk.pdf
    which reproduces "Table I" and "Table II" verbatim with an explicit
    citation to Johansson's paper as reference [16]. The reproduced
    numbers are an exact match, to every reported significant figure,
    with the values this codebase uses. This upgrades the affected
    entries from "cross-referenced, unresolved" to
    SECONDARY_SOURCE_VERIFIED. They remain short of
    PRIMARY_SOURCE_VERIFIED: Johansson's own paper was not directly
    read for Table I. That remains an open item.

Two named literature operating points are retained (REF_MIN_PHASE,
REF_NONMIN_PHASE) purely as reference/sanity-check points; the actual
swept model self-consistently recomputes the steady state at every
gamma via steady_state(), so these two points are not the source of any
reported result -- they are a cross-check that build_model(gamma=0.70)
with k1=3.33,k2=3.35,v0=3.00 approximately reproduces Johansson's
published minimum-phase levels, and similarly for the non-minimum-phase
point.
"""
from dataclasses import dataclass
from enum import Enum


class Provenance(Enum):
    PRIMARY_SOURCE_VERIFIED = "PRIMARY_SOURCE_VERIFIED"
    SECONDARY_SOURCE_VERIFIED = "SECONDARY_SOURCE_VERIFIED"
    ASSUMED_DESIGN_CHOICE = "ASSUMED_DESIGN_CHOICE"


@dataclass(frozen=True)
class Param:
    name: str
    value: float
    units: str
    provenance: Provenance
    note: str


PARAMETER_MANIFEST = [
    Param("A1", 28.0, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 1 cross-section. Matches Strathprints reproduction of Johansson Table I exactly."),
    Param("A3", 28.0, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 3 cross-section, A1=A3 per Johansson's symmetric rig. Same source as A1."),
    Param("A2", 32.0, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 2 cross-section. Same source as A1."),
    Param("A4", 32.0, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 4 cross-section, A2=A4. Same source as A1."),
    Param("a1", 0.071, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 1 outlet orifice area. Same source as A1."),
    Param("a3", 0.071, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 3 outlet orifice area, a1=a3. Same source as A1."),
    Param("a2", 0.057, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 2 outlet orifice area. Same source as A1."),
    Param("a4", 0.057, "cm^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Tank 4 outlet orifice area, a2=a4. Same source as A1."),
    Param("g", 981.0, "cm/s^2", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Gravitational acceleration in the paper's cgs convention. Same source as A1."),
    Param("kc", 0.50, "V/cm", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Level-sensor gain (output map C = kc*I). Same source as A1."),
    Param("k1_minphase", 3.33, "cm^3/(V s)", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Pump-1 gain, minimum-phase operating point. Matches Strathprints Table II."),
    Param("k2_minphase", 3.35, "cm^3/(V s)", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Pump-2 gain, minimum-phase operating point. Matches Strathprints Table II."),
    Param("gamma1_minphase", 0.70, "-", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Valve-1 split ratio, minimum-phase reference point. Matches Strathprints Table II."),
    Param("gamma2_minphase", 0.60, "-", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Valve-2 split ratio, minimum-phase reference point. Matches Strathprints Table II."),
    Param("v0_minphase", 3.00, "V", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Nominal pump voltage, minimum-phase reference point. Matches Strathprints Table II."),
    Param("k1_nonminphase", 3.14, "cm^3/(V s)", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Pump-1 gain, non-minimum-phase operating point. Matches Strathprints Table II."),
    Param("k2_nonminphase", 3.29, "cm^3/(V s)", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Pump-2 gain, non-minimum-phase operating point. Matches Strathprints Table II."),
    Param("gamma1_nonminphase", 0.43, "-", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Valve-1 split ratio, non-minimum-phase reference point. Matches Strathprints Table II."),
    Param("gamma2_nonminphase", 0.34, "-", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Valve-2 split ratio, non-minimum-phase reference point. Matches Strathprints Table II."),
    Param("v0_nonminphase", 3.15, "V", Provenance.SECONDARY_SOURCE_VERIFIED,
          "Nominal pump voltage, non-minimum-phase reference point. Matches Strathprints Table II."),
    # ---- Design choices specific to this model, never claimed to be Johansson's ----
    Param("k1_swept", 3.33, "cm^3/(V s)", Provenance.ASSUMED_DESIGN_CHOICE,
          "Pump-1 gain used across the swept-gamma model (Rounds 10-15): held fixed at the "
          "minimum-phase value while gamma1=gamma2=gamma is swept continuously. This is a "
          "modeling choice (single representative pump-gain pair), not a literature value at "
          "every swept gamma."),
    Param("k2_swept", 3.35, "cm^3/(V s)", Provenance.ASSUMED_DESIGN_CHOICE,
          "Pump-2 gain used across the swept-gamma model. Same rationale as k1_swept."),
    Param("v0_swept", 3.00, "V", Provenance.ASSUMED_DESIGN_CHOICE,
          "Nominal pump voltage used across the swept-gamma model, and the anchor for "
          "alpha_p in the statistical engine."),
    Param("Ts", 5.0, "s", Provenance.ASSUMED_DESIGN_CHOICE,
          "Zero-order-hold discretization sample period. Johansson's rig used a comparable "
          "sample time in his own experiments but this release has not primary- or "
          "secondary-source-verified the exact value 5s specifically; treated as our own "
          "discretization design choice for the discrete-time Sheikhi-style analysis, which "
          "requires a fixed Ts regardless."),
    Param("Kgain", -0.03, "-", Provenance.ASSUMED_DESIGN_CHOICE,
          "Decentralized static proportional output-feedback gain, u = K y + r, K = Kgain*I_2. "
          "This is a controller design choice made for this model (a simple stabilizing "
          "decentralized loop), not a value taken from Johansson, whose original paper studies "
          "the open-loop / MPC-controlled plant, not this specific static output-feedback law."),
]


def manifest_as_dicts():
    return [
        dict(name=p.name, value=p.value, units=p.units,
             provenance=p.provenance.value, note=p.note)
        for p in PARAMETER_MANIFEST
    ]


def provenance_summary():
    counts = {}
    for p in PARAMETER_MANIFEST:
        counts[p.provenance.value] = counts.get(p.provenance.value, 0) + 1
    return counts


if __name__ == "__main__":
    import json
    print(json.dumps(manifest_as_dicts(), indent=2))
    print("\nSummary:", provenance_summary())
