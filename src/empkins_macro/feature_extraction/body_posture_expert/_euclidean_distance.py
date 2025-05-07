from collections.abc import Sequence

import biopsykit as bp
import numpy as np
import pandas as pd

from empkins_macro.feature_extraction.base_functions import euclidean_distance as distance
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS,
    _INDEX_LEVELS_OUT,
    compute_params_from_start_end_time_array,
    start_end_array_indices_to_time,
)


def euclidean_distance(
    data: pd.DataFrame,
    body_part: Sequence[str],
    data_format: str | None = "global_pose",
    channel: str | None = "pos_global",
    axis: str | None = "norm",
    min_distance_cutoff: float | None = 120,
    distance_thres: float | None = 20,
    **kwargs,
) -> pd.DataFrame:
    name = "euclidean_distance"
    if kwargs.get("suffix", False):
        name += f"_{min_distance_cutoff}_{distance_thres}"

    dist = distance(data, body_part, data_format, channel)

    result_dict = {"mean": np.squeeze(np.mean(dist, axis=0)), "std": np.squeeze(np.std(dist, axis=0))}
    out_generic = pd.Series(result_dict)

    start_end = _euclidean_distance_threshold(dist, distance_thres)
    if len(start_end) > 0:
        out = compute_params_from_start_end_time_array(start_end, data)
        out = pd.concat([out_generic, out])
    else:
        out = out_generic

    out = {("_".join(body_part), name, channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)
    out = pd.DataFrame(out, columns=["data"])

    return out


def _euclidean_distance_threshold(data: pd.DataFrame, distance_thres: float) -> pd.DataFrame:
    # get minimum distance over recording
    np.squeeze(np.min(data, axis=0))
    # extract phases where distance is below the sum of the threshold and the overall minimal distance
    # (we need to add this minimal distance to the threshold because we have offsets in the mocap data),
    # but only if the overall minimal distance is below a defined cutoff value
    # (otherwise the offset is considered too high)
    data_thres = data.values < distance_thres

    start_end = bp.utils.array_handling.bool_array_to_start_end_array(data_thres)
    start_end = start_end_array_indices_to_time(data, start_end)
    return start_end
