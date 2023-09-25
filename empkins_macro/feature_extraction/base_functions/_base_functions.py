from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
from biopsykit.utils._datatype_validation_helper import _assert_len_list
from sklearn.preprocessing import minmax_scale
from tsfresh.feature_extraction.feature_calculators import number_crossing_m


def norm(data: pd.DataFrame) -> pd.DataFrame:
    out = np.linalg.norm(data, axis=1)
    return pd.DataFrame(out, columns=["norm"], index=data.index)


def euclidean_distance(
    data: pd.DataFrame,
    body_part: Sequence[str],
    data_format: Optional[str] = "global_pose",
    channel: Optional[str] = "pos_global",
) -> pd.DataFrame:
    _assert_len_list(body_part, 2)
    # assert all(part in get_all_body_parts(system="xsens") for part in body_part)

    data = data.loc[:, pd.IndexSlice[data_format, body_part, channel, :]]
    # compute axis-wise difference
    data = data.groupby("axis", axis=1).diff().dropna(axis=1)
    # distance = l2 norm
    distance = pd.DataFrame(np.linalg.norm(data, axis=1), index=data.index, columns=["data"])
    return distance


def max_val(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.max(data, axis=0))
    out.index = data.columns.get_level_values("axis")
    return out


def max_val_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.max(norm, axis=0)
    return pd.Series([out], index=pd.Index(["norm"]))


def abs_max(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.max(np.abs(data), axis=0))
    out.index = data.columns.get_level_values("axis")
    return out


def abs_max_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.max(np.abs(norm), axis=0)
    return pd.Series([out], index=pd.Index(["norm"]))


def std(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nanstd(data, axis=0))
    out.index = data.columns.get_level_values("axis")
    return out


def std_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.nanstd(norm)
    return pd.Series([out], index=pd.Index(["norm"]))


def mean(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nanmean(data, axis=0))
    out.index = data.columns.get_level_values("axis")
    return out


def mean_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.nanmean(norm, axis=0)
    return pd.Series([out], index=pd.Index(["norm"]))


def mean_abs(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nanmean(np.abs(data), axis=0))
    out.index = data.columns.get_level_values("axis")
    return out


def cov(data: pd.DataFrame) -> pd.Series:
    out = np.std(data, axis=0) / np.nanmean(data, axis=0)
    out.index = data.columns.get_level_values("axis")
    return out


def cov_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    if np.nanmean(norm) != 0:
        out = np.std(norm, axis=0) / np.nanmean(norm, axis=0)
    else:
        out = np.nan
    return pd.Series([out], index=pd.Index(["norm"]))


def entropy(data: pd.DataFrame) -> pd.Series:
    out = minmax_scale(data)
    if np.nansum(out) != 0:
        out = stats.entropy(out)
    else:
        out = np.nan
    return pd.Series(out, index=data.columns.get_level_values(level=-1))


def entropy_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(
        data.dropna(), axis=1
    )  # TODO: is it okay to just drop nans when calculating entropy? I think it is.
    norm = minmax_scale(norm)
    if np.nansum(norm) != 0:
        out = stats.entropy(norm)
    else:
        out = np.nan
    return pd.Series([out], index=pd.Index(["norm"]))


def zero_crossings(data: pd.DataFrame) -> pd.Series:
    out = np.apply_along_axis(number_crossing_m, axis=0, arr=data, m=0)
    return pd.Series(out, index=data.columns.get_level_values(level=-1))


def mean_crossings_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = number_crossing_m(norm, np.nanmean(norm, axis=0).squeeze())
    return pd.Series([out], index=pd.Index(["norm"]))


def abs_energy(data: pd.DataFrame) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import abs_energy

    out = np.apply_along_axis(abs_energy, axis=0, arr=data.dropna())
    return pd.Series(out, index=data.columns.get_level_values(level=-1))


def abs_energy_norm(data: pd.DataFrame) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import abs_energy

    norm = np.linalg.norm(data.dropna(), axis=1)
    out = abs_energy(norm)
    series = pd.Series([out], index=pd.Index(["norm"]))
    return series


def fft_aggregated(data: pd.DataFrame, param: Optional[Sequence[str]] = None) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import fft_aggregated

    if param is None:
        param = ["centroid", "variance", "skew", "kurtosis"]
    if isinstance(param, str):
        param = [param]
    param = [{"aggtype": pn} for pn in param]

    if np.nansum(data) != 0:
        out = np.apply_along_axis(fft_aggregated, axis=0, arr=data, param=param)
        out = [[x[1] for x in o][0] for o in out]
    else:
        out = np.nan
    out = pd.Series(out, index=data.columns.get_level_values(level=-1))
    return out


def fft_aggregated_nan_safe(data: pd.DataFrame, param: Optional[Sequence[str]] = None) -> pd.Series:
    # pick the longest section without nans to approximate fft
    arr = data.iloc[:, 0].values  # Extract out first column from dataframe as array
    m = np.concatenate(([True], np.isnan(arr), [True]))  # Mask
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1, 2)  # Start-stop limits
    start, stop = ss[(ss[:, 1] - ss[:, 0]).argmax()]  # Get max interval, interval limits

    section = data.iloc[start:stop]
    return fft_aggregated(section, param)


def fft_aggregated_norm(data: pd.DataFrame, param: Optional[Sequence[str]] = None) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import fft_aggregated

    if param is None:
        param = ["centroid", "variance", "skew", "kurtosis"]
    if isinstance(param, str):
        param = [param]
    param = [{"aggtype": param_name} for param_name in param]

    norm = np.linalg.norm(data, axis=1)
    if np.nansum(norm) != 0:
        out = list(fft_aggregated(norm, param=param))[0][1]
    else:
        out = np.nan
    return pd.Series([out], index=pd.Index(["norm"]))


def fft_aggregated_norm_nan_safe(data: pd.DataFrame, param: Optional[Sequence[str]] = None) -> pd.Series:
    # pick the longest section without nans to approximate fft
    arr = np.linalg.norm(data, axis=1)  # Extract out first column from dataframe as array
    m = np.concatenate(([True], np.isnan(arr), [True]))  # Mask
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1, 2)  # Start-stop limits
    start, stop = ss[(ss[:, 1] - ss[:, 0]).argmax()]  # Get max interval, interval limits

    section = data.iloc[start:stop]
    return fft_aggregated_norm(section, param)
