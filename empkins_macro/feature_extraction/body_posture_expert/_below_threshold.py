from typing import Optional, Sequence

import biopsykit as bp
import numpy as np
import pandas as pd

from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM
from empkins_macro.feature_extraction._utils import _extract_body_part
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS,
    _INDEX_LEVELS_OUT,
    compute_params_from_start_end_time_array,
    start_end_array_indices_to_time,
)
from empkins_macro.utils._types import str_t


def below_threshold(
    data: pd.DataFrame,
    body_part: str_t,
    data_format: Optional[str] = "calc",
    channel: Optional[str] = "vel",
    axis: Optional[str] = "norm",
    threshold: Optional[float] = 0.1,
    system: Optional[MOTION_CAPTURE_SYSTEM] = "xsens",
    **kwargs,
) -> pd.DataFrame:
    name = "below_threshold"
    if kwargs.get("suffix", False):
        name += f"_{threshold}"

    body_part_name, body_part = _extract_body_part(system=system, body_parts=body_part)
    start_end_below = _below_threshold_per_body_part(data, data_format, body_part, channel, threshold)

    out = compute_params_from_start_end_time_array(start_end_below, data)
    out = {(body_part_name, name, channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)

    return pd.DataFrame(out, columns=["data"])


def _below_threshold_per_body_part(
    data: pd.DataFrame,
    data_format: str,
    body_part: Sequence[str],
    channel: str,
    threshold: float,
) -> pd.DataFrame:
    below_thres_list = []
    for part in body_part:
        data_slice = data.loc[:, pd.IndexSlice[data_format, [part], channel, :]]
        data_norm = np.linalg.norm(data_slice, axis=1)
        data_below_thres = data_norm < threshold * np.max(data_norm)
        below_thres_list.append(data_below_thres)

    out = np.array(below_thres_list)
    out = np.sum(out, axis=0) == len(body_part)
    start_end = bp.utils.array_handling.bool_array_to_start_end_array(out)

    return start_end_array_indices_to_time(data, start_end)
