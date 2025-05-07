import numpy as np
import pandas as pd

"""Module containing various functions that calculate metrics from feature data for a set of VPs.
The functions are returning a dataframe with the calculated value for each VP."""


def min_max(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference of min and max values for each row in a dataframe."""
    max = dataframe.groupby("subject").max()
    min = dataframe.groupby("subject").min()
    diff = max - min
    diff = diff.rename(columns={"data": "min-max"})

    return diff


def averages(series: pd.Series) -> pd.Series:
    """Calculate the difference of the mean of the first and last values for each row in a dataframe."""
    return series.head(3).mean() - series.tail(3).mean()


def diff_head_tail(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate the difference of the mean of the first and last values for each row in a dataframe."""
    diff_head_tail = dataframe.groupby("subject").apply(averages)
    diff_head_tail = diff_head_tail.rename(columns={"data": "diff-head-tail"})

    return diff_head_tail


def average_decrease(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average decrease for each row in a dataframe."""
    av = dataframe.groupby("subject").apply(lambda x: np.mean(np.diff(x["data"])))
    return av.to_frame(name="average_dec")


def percent_decrease(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate the average decrease/increase for each row in a dataframe as a percentage."""
    max_ns = dataframe.groupby("subject").max()
    min_ns = dataframe.groupby("subject").min()

    percent_dec = (max_ns - min_ns) / max_ns

    percent_dec = percent_dec.rename(columns={"data": "percent_dec"})
    return percent_dec


def slp(series):
    """Calculates slope of a straight line fitted to all points of the input series."""
    x = np.arange(series.shape[0])
    y = series
    s, i = np.polyfit(x, y, 1)
    return s


def intersct(series):
    """Calculates intersect of a straight line fitted to all points of the input series."""
    x = np.arange(series.shape[0])
    y = series
    s, i = np.polyfit(x, y, 1)
    return i


def coefficients_2d(series):
    """Calculates the coefficients of a second order polynom fitted to all points of the input series."""
    x = np.arange(series.shape[0])
    y = series
    c = np.polyfit(x, y, 2)
    return c


def approximation_2d(dataframe: pd.DataFrame):
    """Calculates the coefficients of a second order polynom fitted to all points of the input dataframe."""
    coeff = dataframe.groupby("subject", group_keys=False)["data"].apply(coefficients_2d)
    coeff = pd.DataFrame(coeff.tolist(), index=coeff.index, columns=["coeff1", "coeff2", "coeff3"])

    return coeff


def slope(dataframe: pd.DataFrame, half=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the slope for all points in a dataframe (with the option to only use the first half of the values)."""
    if half:
        # only use the first half of the windows for the slope calculation
        idx = "window_count"

        if "Recording time" in dataframe.index.names:
            idx = "Recording time"

        half_window_count = int(np.ceil(dataframe.index.get_level_values(idx).nunique() / 2))
        dataframe = dataframe.loc[
            dataframe.index.get_level_values(idx).isin(dataframe.index.get_level_values(idx)[:half_window_count])
        ]

    slopes = dataframe.groupby("subject", group_keys=False)["data"].apply(slp)
    intersect = dataframe.groupby("subject", group_keys=False)["data"].apply(intersct)

    if half:
        return slopes.to_frame(name="slope_half"), intersect.to_frame(name="intersect_half")

    return slopes.to_frame(name="slope"), intersect.to_frame(name="intersect")
