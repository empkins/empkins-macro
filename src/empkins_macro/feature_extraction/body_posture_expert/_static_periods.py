from collections.abc import Sequence

import numpy as np
import pandas as pd
from biopsykit.signals.imu.static_moment_detection import find_static_moments

from empkins_macro.feature_extraction._utils import _extract_body_part
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS,
    _INDEX_LEVELS_OUT,
    compute_params_from_start_end_time_array,
)
from empkins_macro.utils._types import str_t


def static_periods(
    data: pd.DataFrame,
    body_part: str_t,
    sampling_rate: float,
    data_format: str | None = "calc",
    channel: str | None = "vel",
    axis: str | None = "norm",
    window_sec: int | None = 1,
    overlap_percent: float | None = 0.5,
    threshold: float | None = 0.0001,
    system: str | None = "xsens",
    **kwargs,
) -> pd.DataFrame:
    name = "static_periods"
    if kwargs.get("suffix", False):
        for val in [window_sec, overlap_percent, threshold]:
            name += f"_{np.format_float_positional(val)}"

    body_part_name, body_part = _extract_body_part(system=system, body_parts=body_part)

    static_periods_start_end = _static_periods_per_body_part(
        data,
        data_format,
        body_part,
        channel,
        sampling_rate,
        window_sec,
        overlap_percent,
        threshold,
    )

    out = compute_params_from_start_end_time_array(static_periods_start_end, data)
    out = {(body_part_name, name, channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)

    return pd.DataFrame(out, columns=["data"])


def _static_periods_per_body_part(
    data: pd.DataFrame,
    data_format: str,
    body_part: Sequence[str],
    channel: str,
    sampling_rate: float,
    window_sec: int,
    overlap_percent: float,
    threshold: float,
) -> pd.DataFrame:
    sp_list = []

    for part in body_part:
        data_slice = data.loc[:, pd.IndexSlice[data_format, [part], channel, :]]
        sp_arr = find_static_moments(
            data_slice,
            window_sec=window_sec,
            overlap_percent=overlap_percent,
            sampling_rate=sampling_rate,
            threshold=threshold,
        )
        sp_list.append(sp_arr)

    if len(body_part) > 1:
        sp_list = [
            sp.apply(lambda df: np.arange(df["start"], df["end"]), axis=1).explode(["start", "end"]) for sp in sp_list
        ]
        intersec_arr = sp_list[0]

        for sp in sp_list:
            intersec_arr = np.intersect1d(intersec_arr, sp)
        split_idx = np.where(np.ediff1d(intersec_arr) != 1)[0] + 1
        sp_arr = np.split(intersec_arr, split_idx)

        if np.sum([len(sp) for sp in sp_arr]) == 0:
            return pd.DataFrame(columns=["start", "end"])

        sp_arr = [(s[0], s[-1]) for s in sp_arr]
        sp_arr = pd.DataFrame(sp_arr, columns=["start", "end"])

    else:
        sp_arr = sp_list[0]

    if len(sp_arr) == 0:
        return pd.DataFrame(columns=["start", "end"])
    return pd.DataFrame({key: data.index[sp_arr[key]] for key in ["start", "end"]})
