from typing import Sequence

import biopsykit as bp
import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_len_list

from empkins_io.sensors.motion_capture.body_parts import get_all_body_parts
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS,
    _INDEX_LEVELS_OUT,
    compute_params_from_start_end_time_array,
    start_end_array_indices_to_time
)


def euclidean_distance(data: pd.DataFrame, body_part: Sequence[str], **kwargs) -> pd.DataFrame:
    data_format = kwargs.pop("data_format", "global_pose")
    channel = kwargs.pop("channel", "pos")
    axis = kwargs.pop("axis", "norm")
    name = "euclidean_distance"
    min_distance_cutoff = kwargs.pop("minimal_distance_cutoff", 120)
    distance_thres = kwargs.pop("distance_threshold", 20)

    if kwargs.get("suffix", False):
        name += f"_{min_distance_cutoff}_{distance_thres}"

    _assert_len_list(body_part, 2)
    assert all(part in get_all_body_parts() for part in body_part)

    data = data.loc[:, pd.IndexSlice[data_format, body_part, channel, :]]
    # compute axis-wise difference
    data = data.groupby("axis", axis=1).diff().dropna(axis=1)
    # distance = l2 norm
    distance = pd.DataFrame(np.linalg.norm(data, axis=1), index=data.index, columns=["data"])

    start_end = _euclidean_distance_threshold(distance, min_distance_cutoff, distance_thres)
    out = compute_params_from_start_end_time_array(start_end, data)
    result_dict = {"mean": np.squeeze(np.mean(distance)), "std": np.squeeze(np.std(distance))}
    out_generic = pd.Series(result_dict)
    out = pd.concat([out_generic, out])
    out = {("_".join(body_part), name, channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)
    out = pd.DataFrame(out, columns=["data"])
    return out


def _euclidean_distance_threshold(
    data: pd.DataFrame, min_distance_cutoff: float, distance_thres: float
) -> pd.DataFrame:
    data_min = np.squeeze(np.min(data))
    data_thres = np.logical_and(data.values < data_min + distance_thres, data_min < min_distance_cutoff)
    start_end = bp.utils.array_handling.bool_array_to_start_end_array(data_thres)
    start_end = start_end_array_indices_to_time(data, start_end)
    return start_end
