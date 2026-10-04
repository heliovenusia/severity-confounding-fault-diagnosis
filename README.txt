Reproducibility companion for:

  Auditing Class-Dependent Severity Confounding in Closed-Loop Fault
  Diagnosis Benchmarks

What this experiment does
--------------------------

Fault diagnosis benchmarks are usually built by drawing fault severity
from one shared distribution, regardless of which fault class produced
it. This paper asks what happens when that is not true: when the
severity range associated with a process fault differs systematically
from the severity range associated with a sensor fault, so that severity
itself becomes a usable, if artificial, cue for identifying which fault
occurred.

The experiment constructs two severity policies for a closed-loop
quadruple-tank process with nine (process fault, sensor configuration)
conditions:

  matched policy:     both fault classes draw severity from the same
                       range, 1 to 30 percent.
  confounded policy:   process faults draw severity from 1 to 8 percent,
                       sensor faults draw severity from 15 to 30 percent.

For each condition and each of three diagnostic methods (a classical
Generalized Likelihood Ratio test, XGBoost, and a temporal convolutional
network), a model is trained under each policy and evaluated under each
policy, giving four cells per (condition, method):

  A_MM: matched training,     matched test
  A_MC: matched training,     confounded test
  A_CM: confounded training,  matched test
  A_CC: confounded training,  confounded test

This 2x2 crossing, done independently at training time and test time,
is what separates the paper's three main quantities: the accuracy cost
of training under a confounded policy and then being evaluated on the
matched task (A_MM minus A_CM), how much easier the confounded test
distribution is for a matched-trained model (A_MC minus A_MM), and the
train-by-test interaction between the two (A_CC minus A_CM minus A_MC
plus A_MM). The last of these, the factorial interaction, is the paper's
headline quantity and is not the same as the simple A_CC minus A_MC
contrast; results/README notes in REPRODUCIBILITY.txt explain how to
check both.

Structural discernibility and Bayes-achievable accuracy (A*) are used
only as descriptive characterizations of how informative each of the
nine conditions is, not as the paper's main evidence. The Tennessee
Eastman Process is used only as limited external corroboration that the
sensitivity being studied is not specific to one plant; see
TEP_NOT_REDISTRIBUTED.txt.

What is in this folder
-----------------------

  README.txt                 this file
  REPRODUCIBILITY.txt         how to run and verify the experiment
  TEP_NOT_REDISTRIBUTED.txt   what is and is not included for the TEP part
  CITATION.cff                 citation metadata
  environment/                 Python package requirements
  src/                          the plant model, fault registry, simulator,
                                 and the three diagnostic methods
  scripts/                      the experiment entry points
  results/                      the released result files (severity
                                 matrix, shortcut/leakage audits, task
                                 characterization) and a script that
                                 recomputes the paper's contrasts from
                                 them

A technically competent reader who wants to check one specific number in
the paper can usually do it by reading results/severity_matrix_table.csv
directly, without running anything. A reader who wants to regenerate the
numbers from scratch should start with REPRODUCIBILITY.txt.

Scope boundary
---------------

This release covers the severity-confounding experiment, the three
diagnostic methods used to probe it, the supporting shortcut and leakage
audits, and the structural/Bayes task characterization. It does not
include an earlier, larger comparison of five model architectures or any
of the metrics (eta, calibration error, selective risk, sample
efficiency, noise robustness) that were part of that earlier comparison
and are not part of this paper. It also does not include a reusable
figure-generation script for the paper's main figures; see
REPRODUCIBILITY.txt for what that means in practice.

Licensing
---------
Code in `src/` and `scripts/` is licensed under MIT. Generated result files and repository documentation are licensed under CC BY 4.0. See `LICENSE`, `LICENSE_CODE`, and `LICENSE_DATA`.
