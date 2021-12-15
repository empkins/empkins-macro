from typing import Sequence

import numpy as np
import pandas as pd
from biopsykit.signals.imu.static_moment_detection import find_static_moments

from empkins_macro.feature_extraction._utils import _extract_body_part
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS,
    _INDEX_LEVELS_OUT,
    compute_params_from_start_end_time_array,
)


def static_periods(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    data_format = kwargs.pop("data_format", "calc")
    channel = kwargs.pop("channel", "acc")
    axis = "norm"
    body_part_name, body_part = _extract_body_part(kwargs.pop("body_part", None))
    assert kwargs.get("sampling_rate"), "missing parameter 'sampling_rate'!"

    static_periods_start_end = _static_periods_per_body_part(data, data_format, body_part, channel, **kwargs)

    out = compute_params_from_start_end_time_array(static_periods_start_end, data)
    out = {(body_part_name, "static_periods", channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)

    return pd.DataFrame(out, columns=["data"])


def _static_periods_per_body_part(
    data: pd.DataFrame, data_format: str, body_part: Sequence[str], channel: str, **kwargs
) -> pd.DataFrame:
    sampling_rate = kwargs.get("sampling_rate")
    window_sec = kwargs.get("window_sec", 5)
    overlap_percent = kwargs.get("overlap_percent", 50)
    threshold = kwargs.get("threshold", 0.0001)

    static_periods_list = []
    for part in body_part:
        data_slice = data.loc[:, pd.IndexSlice[data_format, [part], channel, :]]
        sp_arr = find_static_moments(
            data_slice,
            window_sec=window_sec,
            overlap_percent=overlap_percent,
            sampling_rate=sampling_rate,
            threshold=threshold,
        )
        static_periods_list.append(sp_arr)

    if len(body_part) > 1:
        static_periods_list = [
            sp.apply(lambda df: np.arange(df["start"], df["end"]), axis=1).explode() for sp in static_periods_list
        ]

        intersec_arr = np.intersect1d(*static_periods_list)
        split_idx = np.where(np.ediff1d(intersec_arr) != 1)[0] + 1
        sp_arr = np.split(intersec_arr, split_idx)
        sp_arr = [(s[0], s[-1]) for s in sp_arr]
        sp_arr = pd.DataFrame(sp_arr, columns=["start", "end"])
    else:
        sp_arr = static_periods_list[0]

    if len(sp_arr) == 0:
        return pd.DataFrame(columns=["start", "end"])
    return pd.DataFrame({key: data.index[sp_arr[key]] for key in ["start", "end"]})
