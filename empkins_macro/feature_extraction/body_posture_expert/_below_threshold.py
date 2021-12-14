from typing import Sequence

import numpy as np
import pandas as pd
import biopsykit as bp

from empkins_macro.feature_extraction._utils import _extract_body_part, _INDEX_LEVELS_OUT


def below_threshold(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    data_format = kwargs.pop("data_format", "calc")
    channel = kwargs.pop("channel", "vel")
    axis = "norm"
    body_part_name, body_part = _extract_body_part(kwargs.pop("body_part", None))

    data_binarized = _below_threshold_per_body_part(data, data_format, body_part, channel, **kwargs)
    data_binarized_duration = data_binarized.diff(axis=1)["end"]
    data_len = data.index[-1] - data.index[0]

    dict_out = {
        "count_per_min": (60 * len(data_binarized_duration)) / data_len,
        "ratio_percent": np.sum(data_binarized_duration) / data_len,
        "max_duration_sec": data_binarized_duration.max(),
        "mean_duration_sec": data_binarized_duration.mean(),
        "std_duration_sec": data_binarized_duration.std(),
    }
    out = pd.Series(dict_out)
    out = {(body_part_name, "", channel, axis): out}
    out = pd.concat(out, names=_INDEX_LEVELS_OUT)

    return pd.DataFrame(out, columns=["data"])


def _below_threshold_per_body_part(
    data: pd.DataFrame, data_format: str, body_part: Sequence[str], channel: str, **kwargs
) -> pd.DataFrame:
    threshold = kwargs.get("threshold", 0.1)
    binarize_list = []
    for part in body_part:
        data_slice = data.loc[:, pd.IndexSlice[data_format, [part], channel, :]]
        data_norm = np.linalg.norm(data_slice, axis=1)
        binarize_list.append(data_norm < threshold * np.ptp(data_norm))

    out = np.array(binarize_list)
    out = np.sum(out, axis=0) == len(body_part)

    out = bp.utils.array_handling.bool_array_to_start_end_array(out)
    out = pd.DataFrame({"start": data.index[out[:, 0]], "end": data.index[out[:, 1]]})
    return out
