from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from empkins_macro.feature_extraction.body_posture_expert._utils import (
    _INDEX_LEVELS_OUT,
)


def range_of_motion(
    data: pd.DataFrame,
    body_part: Sequence[str],
    data_format: str | None = "global_pose",
    channel: str | None = "pos_global",
    **kwargs,
) -> pd.DataFrame:
    euler_deg = data.loc[:, pd.IndexSlice[data_format, body_part, channel, ["x", "y", "z"]]].to_numpy()
    rotations = R.from_euler("xyz", euler_deg, degrees=True)
    rotation_magnitude_rad = rotations.magnitude()

    rotation_magnitude_deg = np.rad2deg(rotation_magnitude_rad)
    rom_3d = rotation_magnitude_deg.max() - rotation_magnitude_deg.min()
    out = pd.DataFrame({"data": [rom_3d]})
    out = pd.concat({body_part[0]: out}, names=["body_part"])
    out = pd.concat({channel: out}, names=["channel"])
    out = pd.concat({"range_of_motion": out}, names=["type"])
    out = pd.concat({"range_of_motion": out}, names=["metric"])
    out = pd.concat({"x_y_z": out}, names=["axis"])
    out = out.droplevel(None)
    out = out.reorder_levels(_INDEX_LEVELS_OUT)

    return out
