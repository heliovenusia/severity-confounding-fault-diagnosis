"""
Converts a (L, ny) observed trajectory into the input tensor used by the
TCN. The TCN sees exactly the same (L, ny) array that XGBoost and GLR see
flattened, only reshaped to channel-first for torch's Conv1d -- no
additional information is introduced here.
"""
import numpy as np


def tcn_input(Y: np.ndarray) -> np.ndarray:
    """(L, ny) -> (ny, L): channel-first, as expected by torch Conv1d
    (channels=ny sensors, length=L timesteps). No padding, no masking --
    the TCN's input width is simply ny, exactly like XGBoost/GLR's."""
    return Y.T.astype(np.float32)
