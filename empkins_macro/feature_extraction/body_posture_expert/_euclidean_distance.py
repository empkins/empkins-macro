from typing import Sequence, Dict

import numpy as np
import pandas as pd
import biopsykit as bp
from biopsykit.utils._datatype_validation_helper import _assert_len_list
from empkins_io.sensors.motion_capture.body_parts import get_all_body_parts
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS,
    _INDEX_LEVELS_OUT,
    start_end_array_indices_to_time,
    compute_params_from_start_end_time_array,
)


def euclidean_distance(data: pd.DataFrame, body_parts: Sequence[str], **kwargs) -> pd.DataFrame:
    data_format = kwargs.pop("data_format", "global_pose")
    channel = kwargs.pop("channel", "pos")
    axis = kwargs.pop("axis", "norm")

    _assert_len_list(body_parts, 2)
    assert all(body_part in get_all_body_parts() for body_part in body_parts)

    data = data.loc[:, pd.IndexSlice[data_format, body_parts, channel, :]]
    # compute axis-wise difference
    data = data.groupby("axis", axis=1).diff().dropna(axis=1)
    # distance = l2 norm
    distance = pd.DataFrame(np.linalg.norm(data, axis=1), index=data.index, columns=["data"])

    start_end = _euclidean_distance_threshold(distance, kwargs.get("distance_params", {}))
    out = compute_params_from_start_end_time_array(start_end, data)
    result_dict = {"mean": np.squeeze(np.mean(distance)), "std": np.squeeze(np.std(distance))}
    out_generic = pd.Series(result_dict)
    out = pd.concat([out_generic, out])
    out = {("_".join(body_parts), "euclidean_distance", channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)
    out = pd.DataFrame(out, columns=["data"])
    return out


def _euclidean_distance_threshold(data: pd.DataFrame, param_dict: Dict[str, float]) -> pd.DataFrame:
    min_distance_cutoff = param_dict.get("minimal_distance_cutoff", 120)
    distance_thres = param_dict.get("distance_threshold", 20)
    data_min = np.squeeze(np.min(data))
    data_thres = np.logical_and(data.values < data_min + distance_thres, data_min < min_distance_cutoff)
    start_end = bp.utils.array_handling.bool_array_to_start_end_array(data_thres)
    start_end = start_end_array_indices_to_time(data, start_end)
    return start_end
