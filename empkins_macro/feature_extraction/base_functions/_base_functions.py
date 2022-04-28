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
    out = pd.Series(np.max(data))
    out.index = out.index.get_level_values("axis")
    return out


def max_val_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.max(norm)
    return pd.Series([out], index=pd.Index(["norm"]))


def abs_max(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.max(np.abs(data)))
    out.index = out.index.get_level_values("axis")
    return out


def abs_max_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.max(np.abs(norm))
    return pd.Series([out], index=pd.Index(["norm"]))


def std(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.std(data))
    out.index = out.index.get_level_values("axis")
    return out


def std_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.std(norm)
    return pd.Series([out], index=pd.Index(["norm"]))


def mean(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.mean(data))
    out.index = out.index.get_level_values("axis")
    return out


def mean_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = np.mean(norm)
    return pd.Series([out], index=pd.Index(["norm"]))


def mean_abs(data: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.mean(np.abs(data)))
    out.index = out.index.get_level_values("axis")
    return out


def cov(data: pd.DataFrame) -> pd.Series:
    out = np.std(data) / np.mean(data)
    out.index = out.index.get_level_values("axis")
    return out


def cov_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    if np.mean(norm) != 0:
        out = np.std(norm) / np.mean(norm)
    else:
        out = np.nan
    return pd.Series([out], index=pd.Index(["norm"]))


def entropy(data: pd.DataFrame) -> pd.Series:
    out = minmax_scale(data)
    if np.sum(out) != 0:
        out = stats.entropy(out)
    else:
        out = np.nan
    return pd.Series(out, index=data.columns.get_level_values(level=-1))


def entropy_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    norm = minmax_scale(norm)
    if np.sum(norm) != 0:
        out = stats.entropy(norm)
    else:
        out = np.nan
    return pd.Series([out], index=pd.Index(["norm"]))


def zero_crossings(data: pd.DataFrame) -> pd.Series:
    out = np.apply_along_axis(number_crossing_m, axis=0, arr=data, m=0)
    return pd.Series(out, index=data.columns.get_level_values(level=-1))


def mean_crossings_norm(data: pd.DataFrame) -> pd.Series:
    norm = np.linalg.norm(data, axis=1)
    out = number_crossing_m(norm, np.mean(norm).squeeze())
    return pd.Series([out], index=pd.Index(["norm"]))


def abs_energy(data: pd.DataFrame) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import abs_energy

    out = np.apply_along_axis(abs_energy, axis=0, arr=data)
    return pd.Series(out, index=data.columns.get_level_values(level=-1))


def abs_energy_norm(data: pd.DataFrame) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import abs_energy

    norm = np.linalg.norm(data, axis=1)
    out = abs_energy(norm)
    return pd.Series([out], index=pd.Index(["norm"]))


def fft_aggregated(data: pd.DataFrame, param: Optional[Sequence[str]] = None) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import fft_aggregated

    if param is None:
        param = ["centroid", "variance", "skew", "kurtosis"]
    if isinstance(param, str):
        param = [param]
    param = [{"aggtype": pn} for pn in param]

    if np.sum(data) != 0:
        out = np.apply_along_axis(fft_aggregated, axis=0, arr=data, param=param)
        out = [[x[1] for x in o][0] for o in out]
    else:
        out = np.nan
    out = pd.Series(out, index=data.columns.get_level_values(level=-1))
    return out


def fft_aggregated_norm(data: pd.DataFrame, param: Optional[Sequence[str]] = None) -> pd.Series:
    from tsfresh.feature_extraction.feature_calculators import fft_aggregated

    if param is None:
        param = ["centroid", "variance", "skew", "kurtosis"]
    if isinstance(param, str):
        param = [param]
    param = [{"aggtype": param_name} for param_name in param]

    norm = np.linalg.norm(data, axis=1)
    if np.sum(norm) != 0:
        out = list(fft_aggregated(norm, param=param))[0][1]
    else:
        out = np.nan
    return pd.Series([out], index=pd.Index(["norm"]))
