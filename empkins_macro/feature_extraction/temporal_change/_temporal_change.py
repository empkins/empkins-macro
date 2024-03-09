from typing import Tuple, Sequence, Optional

import pandas as pd
import numpy as np

"""Module containing various functions that calculate metrics from feature data for a set of VPs. 
The functions are returning a dataframe with the calculated value for each VP."""

_INDEX_LEVELS: Sequence[str] = ["temporal_change", "metric", "type", "body_part", "channel", "axis"]
_INDEX_LEVELS_OUT: Sequence[str] = ["body_part", "channel", "type", "metric", "temporal_change", "axis"]


def max_min(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference of min and max values for each row in a dataframe."""
    out = data.max(axis=1) - data.min(axis=1)
    return out.to_frame(name="max_min")


def diff_head_tail(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference of the mean of the first and last values for each row in a dataframe."""
    out = data.iloc[:, 0:3].mean(axis=1) - data.iloc[:, -4:-1].mean(axis=1)
    return out.to_frame(name="diff_head_tail")


def average_decrease(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average decrease for each row in a dataframe."""
    out = data.diff(axis=1).mean(axis=1)
    return out.to_frame(name="average_dec")


def percent_decrease(data: pd.DataFrame) -> pd.Series:
    """Calculate the average decrease/increase for each row in a dataframe as a percentage."""
    out = (data.max(axis=1) - data.min(axis=1)) / data.max(axis=1)
    return out.to_frame(name="percent_dec")


def polyfit_2d(data: pd.DataFrame):
    """Calculates the coefficients of a second order polynom fitted to all points of the input dataframe."""
    return _slope_helper(data, degree=2, half=False)


def polyfit_2d_half(data: pd.DataFrame):
    """Calculates the coefficients of a second order polynom fitted to all points of the input dataframe."""
    return _slope_helper(data, degree=2, half=True)


def linefit_1d(data: pd.DataFrame) -> pd.DataFrame:
    return _slope_helper(data, degree=1, half=False)


def linefit_1d_half(data: pd.DataFrame) -> pd.DataFrame:
    return _slope_helper(data, degree=1, half=True)


def _slope_helper(data: pd.DataFrame, degree: Optional[int] = 1, half: Optional[bool] = False) -> pd.DataFrame:
    """Calculate the slope for all points in a dataframe (with the option to only use the first half of the values)."""
    n_windows = len(data.columns)
    if half:
        n_windows = int(np.ceil(n_windows / 2))

    data = data.iloc[:, :n_windows]
    out = np.polynomial.polynomial.polyfit(x=np.arange(len(data.columns)), y=data.T, deg=degree)
    out = pd.DataFrame(out.T, index=data.index)
    out.columns = [f"a{col}" for col in reversed(out.columns)]
    if degree == 1:
        out = out.add_prefix("linefit_")
    else:
        out = out.add_prefix(f"polyfit_{degree}d_")

    if half:
        out = out.add_suffix("_half")
    return out
