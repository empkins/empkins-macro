from typing import Optional

import numpy as np
import pandas as pd

from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM
from empkins_macro.feature_extraction._utils import _extract_body_part
from empkins_macro.feature_extraction.body_posture_expert._utils import _INDEX_LEVELS, _INDEX_LEVELS_OUT


def absolute_movement(
    data: pd.DataFrame, system: MOTION_CAPTURE_SYSTEM, data_format: Optional[str] = "global_pose", **kwargs
) -> pd.DataFrame:
    channel = "pos_global"
    axis = "norm"
    name = "absolute_movement"
    body_part_name, body_part = _extract_body_part(system=system, body_parts=kwargs.get("body_part", None))

    data = data.loc[:, pd.IndexSlice[data_format, body_part, channel, :]]
    data = data.stack(["data_format", "body_part", "channel"])
    out = np.linalg.norm(data, axis=1)
    out = pd.DataFrame(out, index=data.index, columns=["norm"])
    out = out.unstack("time")
    out = out.diff(axis=1).abs().sum().mean()
    out = pd.Series([out])
    out.index = pd.MultiIndex.from_tuples([(body_part_name, name, channel, axis, name)], names=_INDEX_LEVELS)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)

    return pd.DataFrame(out, columns=["data"])
