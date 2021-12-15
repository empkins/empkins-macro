from typing import Sequence, Dict

import numpy as np
import pandas as pd

_INDEX_LEVELS: Sequence[str] = ["body_part", "type", "channel", "axis", "metric"]
_INDEX_LEVELS_OUT: Sequence[str] = ["body_part", "channel", "type", "metric", "axis"]


def start_end_array_indices_to_time(data: pd.DataFrame, start_end: np.ndarray) -> pd.DataFrame:
    if len(start_end) == 0:
        return pd.DataFrame(columns=["start", "end"])
    # end indices are *inclusive*!
    start_end[:, 1] -= 1
    return pd.DataFrame({"start": data.index[start_end[:, 0]], "end": data.index[start_end[:, 1]]})


def compute_params_from_start_end_time_array(start_end: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
    data_len = data.index[-1] - data.index[0]
    start_end_duration = start_end.diff(axis=1)["end"]
    dict_out = {
        "count_per_min": (60 * len(start_end_duration)) / data_len,
        "ratio_percent": np.sum(start_end_duration) / data_len,
        "max_duration_sec": start_end_duration.max(),
        "mean_duration_sec": start_end_duration.mean(),
        "std_duration_sec": start_end_duration.std(),
    }

    return pd.Series(dict_out)
