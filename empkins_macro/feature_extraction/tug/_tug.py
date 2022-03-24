import numpy as np
import pandas as pd
from gaitmap.utils.rotations import find_unsigned_3d_angle
from scipy.ndimage import find_objects
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.spatial.transform.rotation import Rotation


class TUG:
    data: pd.DataFrame
    start: float
    end: float
    features: pd.DataFrame

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._extract_tug_times()

    def _extract_tug_times(self, thres: float = 0.1):
        # get velocity norm of chest
        t8_norm = self.data["mvnx_segment"]["T8"]["vel"].apply(np.linalg.norm, axis=1)

        # get possible tug regions
        regions = find_objects(label(t8_norm > thres)[0])

        # longest region is the TUG
        regs = [t8_norm.iloc[r] for r in regions]
        lengths = [len(r) for r in regs]

        tug = regs[np.argmax(lengths)]

        self.start = tug.index[0]
        self.end = tug.index[-1]

        # cut data to valid region
        self.data = self.data[self.start:self.end]

    def extract_tug_features(self):
        features = [self._calc_tug_time(), self._calc_time_to_stand_up(), self._calc_first_step_length()]
        self.features = pd.DataFrame(pd.concat(features), columns=["data"])

    def _calc_tug_time(self) -> pd.Series:
        return pd.Series(self.end - self.start, ["tug_time"])

    def _calc_time_to_stand_up(self) -> pd.Series:
        left_fa = _get_floor_angle(self.data["mvnx_segment"]["LeftUpperLeg"]["ori"])
        right_fa = _get_floor_angle(self.data["mvnx_segment"]["LeftUpperLeg"]["ori"])

        # first time that leg to floor angle is 0
        left_stand_up_time = left_fa.index[zero_crossings(left_fa)][0]
        right_stand_up_time = right_fa.index[zero_crossings(right_fa)][0]

        return pd.Series(min(left_stand_up_time, right_stand_up_time), ["time_to_stand_up"])

    def _calc_first_step_length(self, thres: float = 0.1) -> pd.Series:
        left_pos = self.data["mvnx_segment"]["LeftFoot"]["pos"]
        right_pos = self.data["mvnx_segment"]["RightFoot"]["pos"]

        diff = left_pos - right_pos
        eucl = diff.apply(np.linalg.norm, axis=1)

        peaks, _ = find_peaks(eucl, height=thres)

        peak_times = eucl.iloc[peaks]

        first_step_end = peak_times.index[1]
        # index 1 is the end of the first step

        left_len = left_pos.loc[first_step_end] - left_pos.loc[self.start]
        left_len = np.linalg.norm(left_len)

        right_len = right_pos.loc[first_step_end] - right_pos.loc[self.start]
        right_len = np.linalg.norm(right_len)

        # max between left and right
        return pd.Series(max(left_len, right_len), ["first_step_length"])

def _get_floor_angle(data: pd.DataFrame) -> pd.Series:
    forward = pd.DataFrame(
        Rotation.from_quat(data.to_numpy()).apply([1, 0, 0]),
        columns=list("xyz"), index=data.index)
    floor_angle = np.rad2deg(find_unsigned_3d_angle(forward.to_numpy(), np.array([0, 0, 1]))) - 90
    return pd.Series(floor_angle, index=forward.index)


def zero_crossings(df: pd.Series):
    return np.where(np.diff(np.signbit(df)))[0]
