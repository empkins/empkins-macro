from typing import Dict

import numpy as np
import pandas as pd
from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation
from gaitmap.utils.rotations import find_angle_between_orientations
from scipy.spatial.transform import Rotation


class StrideDetection:
    data = pd.DataFrame
    joint_data = pd.DataFrame
    _min_vel_event_list: Dict[str, pd.DataFrame]
    _sequence_list: Dict[str, pd.DataFrame]
    temporal_features: pd.DataFrame
    spatial_features: pd.DataFrame

    def __init__(self, data: pd.DataFrame, joint_data: pd.DataFrame):
        self.data = data
        self.joint_data = joint_data
        self._stride_detection()
        self._clean_min_vel_event_list()

    def temporal_features(self, sampling_rate: float = 1):
        temporal_paras = TemporalParameterCalculation()
        temporal_paras = temporal_paras.calculate(
            stride_event_list=self._min_vel_event_list,
            sampling_rate_hz=sampling_rate,  # calculations were done in seconds not samples
        )

        self.temporal_features = pd.concat(
            temporal_paras.parameters_, names=["side", "s_id"]
        )

    def spatial_features(self, sampling_rate: float = 60):
        # convert to match gaitmap definition
        df_left = _convert_position_and_orientation(
            self.data["LeftFoot"],
            self._min_vel_event_list["left"],
            self._sequence_list["left"],
        )
        df_right = _convert_position_and_orientation(
            self.data["RightFoot"],
            self._min_vel_event_list["right"],
            self._sequence_list["right"],
        )

        # match to sampling rate
        _min_vel_event_list_matched = self._min_vel_event_list.copy()
        _min_vel_event_list_matched["left"] = (
            self._min_vel_event_list["left"][
                ["start", "end", "tc", "ic", "min_vel", "pre_ic"]
            ]
            * sampling_rate
        )
        _min_vel_event_list_matched["right"] = (
            self._min_vel_event_list["right"][
                ["start", "end", "tc", "ic", "min_vel", "pre_ic"]
            ]
            * sampling_rate
        )

        spatial_paras_l = SpatialParameterCalculation()
        spatial_paras_l = spatial_paras_l.calculate(
            stride_event_list=_min_vel_event_list_matched["left"],
            positions=df_left["pos"],
            orientations=df_left["ori"],
            sampling_rate_hz=sampling_rate,
        )

        spatial_paras_r = SpatialParameterCalculation()
        spatial_paras_r = spatial_paras_r.calculate(
            stride_event_list=_min_vel_event_list_matched["right"],
            positions=df_right["pos"],
            orientations=df_right["ori"],
            sampling_rate_hz=sampling_rate,
        )

        paramater_df_l = spatial_paras_l.parameters_
        paramater_df_r = spatial_paras_r.parameters_

        paramater_df_l = paramater_df_l.join(
            _calc_max_knee_flexion(
                self.joint_data["jLeftKnee"]["ang"], self._min_vel_event_list["left"]
            ),
        )
        paramater_df_r = paramater_df_r.join(
            _calc_max_knee_flexion(
                self.joint_data["jRightKnee"]["ang"], self._min_vel_event_list["right"]
            )
        )

        paramater_df_l = paramater_df_l.join(
            _calc_min_knee_flexion(
                self.joint_data["jLeftKnee"]["ang"], self._min_vel_event_list["left"]
            ),
        )
        paramater_df_r = paramater_df_r.join(
            _calc_min_knee_flexion(
                self.joint_data["jRightKnee"]["ang"], self._min_vel_event_list["right"]
            )
        )

        paramater_df_l = paramater_df_l.join(
            _calc_max_arm_flexion(
                self.joint_data["jLeftElbow"]["ang"], self._min_vel_event_list["left"]
            ),
        )
        paramater_df_r = paramater_df_r.join(
            _calc_max_arm_flexion(
                self.joint_data["jRightElbow"]["ang"], self._min_vel_event_list["right"]
            )
        )

        paramater_df_l = paramater_df_l.join(
            _calc_min_arm_flexion(
                self.joint_data["jLeftElbow"]["ang"], self._min_vel_event_list["left"]
            ),
        )
        paramater_df_r = paramater_df_r.join(
            _calc_min_arm_flexion(
                self.joint_data["jRightElbow"]["ang"], self._min_vel_event_list["right"]
            )
        )

        self.spatial_features = pd.concat(
            {
                "left": paramater_df_l,
                "right": paramater_df_r,
            },
            names=["side", "s_id"],
        )

    def _stride_detection(self):

        stride_events = {
            "left": _get_stride_events(self.data["FootContacts"]["left"]),
            "right": _get_stride_events(self.data["FootContacts"]["right"]),
        }

        stride_events = {
            "left": _clean_stride_events(self.data["LeftFoot"], stride_events["left"]),
            "right": _clean_stride_events(
                self.data["RightFoot"], stride_events["right"]
            ),
        }

        stride_event_times = {
            "left": _get_stride_event_times(stride_events["left"]),
            "right": _get_stride_event_times(stride_events["right"]),
        }

        min_vel = {
            "left": _get_min_vel(stride_event_times["left"], self.data["LeftFoot"]),
            "right": _get_min_vel(stride_event_times["right"], self.data["RightFoot"]),
        }

        min_vel_event_list = {
            "left": _build_min_vel_event_list(
                stride_event_times["left"], min_vel["left"]
            ),
            "right": _build_min_vel_event_list(
                stride_event_times["right"], min_vel["right"]
            ),
        }

        self._min_vel_event_list = min_vel_event_list

    def _clean_min_vel_event_list(self, turning_thres: float = 20):

        # drop invalid strides, either tc or ic was not detected
        self._min_vel_event_list = {
            "left": self._min_vel_event_list["left"].dropna(),
            "right": self._min_vel_event_list["right"].dropna(),
        }

        # convert to match gaitmap definition
        df_left = _convert_position_and_orientation(
            self.data["Pelvis"],
            self._min_vel_event_list["left"],
            _get_gait_sequence(self._min_vel_event_list["left"]),
        )
        df_right = _convert_position_and_orientation(
            self.data["Pelvis"],
            self._min_vel_event_list["right"],
            _get_gait_sequence(self._min_vel_event_list["right"]),
        )

        # drop turning strides (turning angle > thres)
        self._min_vel_event_list["left"]["hip_turn"] = _calc_hip_turning_angle(
            df_left["ori"]
        )
        self._min_vel_event_list["right"]["hip_turn"] = _calc_hip_turning_angle(
            df_right["ori"]
        )

        self._min_vel_event_list = {
            "left": self._min_vel_event_list["left"][
                self._min_vel_event_list["left"]["hip_turn"] < turning_thres
            ],
            "right": self._min_vel_event_list["right"][
                self._min_vel_event_list["right"]["hip_turn"] < turning_thres
            ],
        }

        self._sequence_list = {
            "left": _get_gait_sequence(self._min_vel_event_list["left"]),
            "right": _get_gait_sequence(self._min_vel_event_list["right"]),
        }

        print(
            "{} left and {} right strides detected. ({}/{} bouts)".format(
                len(self._min_vel_event_list["left"]),
                len(self._min_vel_event_list["right"]),
                len(self._sequence_list["left"]),
                len(self._sequence_list["right"]),
            )
        )


def _convert_position_and_orientation(
    data: pd.DataFrame,
    min_vel_event_list: pd.DataFrame,
    sequence_list: pd.DataFrame,
) -> pd.DataFrame:
    """Convert data to match gaitmap definition.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe from Xsens system
    min_vel_event_list : :class:`~pandas.DataFrame`
        dataframe with segmented steps, according to gaitmap definition

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with renamed columns and new index

    """
    cleaned_data = pd.DataFrame()

    # cut to valid region
    for start, end in zip(sequence_list["start"], sequence_list["end"]):
        cleaned_data = cleaned_data.append(data[start:end])

        # omit last sample
        cleaned_data = cleaned_data.iloc[:-1]

    # construct index
    s_id = []
    sample = []

    min_vel_event_list = min_vel_event_list.copy()
    min_vel_event_list["num_samples"] = round(
        (min_vel_event_list["end"] - min_vel_event_list["start"]) * 60
    )

    for idx, row in min_vel_event_list.iterrows():
        s_id.append(np.full(int(row["num_samples"]), idx))
        sample.append(np.arange(0, int(row["num_samples"])))

    s_id = np.concatenate(s_id)
    sample = np.concatenate(sample)

    multi_index = pd.MultiIndex.from_arrays((s_id, sample), names=["s_id", "sample"])

    if len(cleaned_data) != len(multi_index):
        raise ValueError(
            "Length of index ({}) and data ({}) does not match!".format(
                len(multi_index), len(cleaned_data)
            )
        )

    cleaned_data = cleaned_data.set_index(multi_index)
    cleaned_data = cleaned_data.rename(
        columns={"x": "pos_x", "y": "pos_y", "z": "pos_z"}
    )
    cleaned_data = cleaned_data.rename(
        columns={"q0": "q_x", "q1": "q_y", "q2": "q_z", "q3": "q_w"}
    )

    return cleaned_data


def _calc_max_knee_flexion(
    joint_angle: pd.DataFrame, min_vel_event_list: pd.DataFrame
) -> pd.DataFrame:
    angle = pd.DataFrame(
        [
            joint_angle[start:end]["z"].max()
            for start, end in zip(
                min_vel_event_list["start"], min_vel_event_list["end"]
            )
        ],
        index=min_vel_event_list.index,
        columns=["max_knee_flexion"],
    )

    return angle


def _calc_min_knee_flexion(
    joint_angle: pd.DataFrame, min_vel_event_list: pd.DataFrame
) -> pd.DataFrame:
    angle = pd.DataFrame(
        [
            joint_angle[start:end]["z"].min()
            for start, end in zip(
                min_vel_event_list["start"], min_vel_event_list["end"]
            )
        ],
        index=min_vel_event_list.index,
        columns=["min_knee_flexion"],
    )

    return angle


def _calc_max_arm_flexion(
    joint_angle: pd.DataFrame, min_vel_event_list: pd.DataFrame
) -> pd.DataFrame:
    angle = pd.DataFrame(
        [
            joint_angle[start:end]["z"].max()
            for start, end in zip(
                min_vel_event_list["start"], min_vel_event_list["end"]
            )
        ],
        index=min_vel_event_list.index,
        columns=["max_arm_flexion"],
    )

    return angle


def _calc_min_arm_flexion(
    joint_angle: pd.DataFrame, min_vel_event_list: pd.DataFrame
) -> pd.DataFrame:
    angle = pd.DataFrame(
        [
            joint_angle[start:end]["z"].min()
            for start, end in zip(
                min_vel_event_list["start"], min_vel_event_list["end"]
            )
        ],
        index=min_vel_event_list.index,
        columns=["min_arm_flexion"],
    )

    return angle


def _calc_hip_turning_angle(orientations: pd.DataFrame) -> pd.Series:
    start = orientations.groupby(level="s_id").first()
    end = orientations.groupby(level="s_id").last()
    angles = pd.Series(
        np.rad2deg(
            find_angle_between_orientations(
                Rotation.from_quat(end.to_numpy()),
                Rotation.from_quat(start.to_numpy()),
                np.asarray([0, 0, 1]),
            )
        ),
        index=start.index,
    )

    return angles


def _get_stride_events(foot_contacts: pd.DataFrame) -> pd.DataFrame:
    # change from 1 (contact to ground) to 0 (no ground contact) is possible initial contact (ic)
    diff_heel = np.ediff1d(foot_contacts["heel"], to_end=0)
    diff_heel = pd.DataFrame(diff_heel)
    diff_heel.index = foot_contacts.index
    ic = diff_heel == 1
    ic.columns = ["ic"]

    diff_toe = np.ediff1d(foot_contacts["toe"], to_end=0)
    diff_toe = pd.DataFrame(diff_toe)
    diff_toe.index = foot_contacts.index
    tc = diff_toe == -1
    tc.columns = ["tc"]

    return ic.join(tc)


def _clean_stride_events(
    segment_data: pd.DataFrame, stride_events: pd.DataFrame, thres: float = 50
) -> pd.DataFrame:

    # change of contact is no stride event when gyr norm is low
    mask_gyr = segment_data["gyr"].apply(np.linalg.norm, axis=1) > thres

    ic_cleaned = pd.DataFrame(stride_events["ic"] & mask_gyr)
    ic_cleaned.columns = ["ic"]
    tc_cleaned = pd.DataFrame(stride_events["tc"] & mask_gyr)
    tc_cleaned.columns = ["tc"]
    stride_events = ic_cleaned.join(tc_cleaned)

    return stride_events


def _find_matching_stride_events(stride_events: pd.DataFrame) -> pd.DataFrame:
    diff = stride_events["ic"].count() - stride_events["tc"].count()

    positive = (stride_events["tc"] - stride_events["ic"]) > 0

    # does the number of stride events match and the ic event is always after the tc event
    if diff == 0 & positive.count() == 0:
        return stride_events

    # shift tc by -1
    stride_events["tc_shifted"] = stride_events["tc"].shift(-1)

    # find ic in between tc(x) and tc(x-1)
    stride_events["ic"] = pd.DataFrame(
        [
            stride_events.query("@tc < ic < @tc_shifted")["ic"].values
            for tc, tc_shifted in zip(stride_events["tc"], stride_events["tc_shifted"])
        ]
    )

    return stride_events[["tc", "ic"]]


def _get_stride_event_times(stride_events_cleaned: pd.DataFrame) -> pd.DataFrame:
    ic_times = stride_events_cleaned["ic"]
    ic_times = pd.DataFrame(ic_times[ic_times].index)
    ic_times.columns = ["ic"]

    tc_times = stride_events_cleaned["tc"]
    tc_times = pd.DataFrame(tc_times[tc_times].index)
    tc_times.columns = ["tc"]

    stride_event_times = ic_times.combine_first(tc_times)

    stride_event_times = _find_matching_stride_events(stride_event_times)

    return stride_event_times


def _get_min_vel(
    stride_events_cleaned: pd.DataFrame, segment_data: pd.DataFrame
) -> pd.DataFrame:
    min_vel_list = []
    gyr_norm = segment_data["vel"].apply(np.linalg.norm, axis=1)

    # shift ic by 1 to get the matching ic and tc times
    stride_events_cleaned = stride_events_cleaned.copy()
    stride_events_cleaned["ic"] = stride_events_cleaned["ic"].shift(1)

    # drop first row
    stride_events_cleaned = stride_events_cleaned.iloc[1:, :]

    for ic, tc in zip(stride_events_cleaned["ic"], stride_events_cleaned["tc"]):
        if np.isnan(ic) or np.isnan(tc):
            min_vel_list.append(float("nan"))
        else:
            min_vel_list.append(gyr_norm.loc[ic:tc].idxmin())

    return pd.DataFrame(min_vel_list, columns=["min_vel"])


def _build_min_vel_event_list(
    stride_events_cleaned: pd.DataFrame, min_vel_events: pd.DataFrame
) -> pd.DataFrame:
    min_vel_event_list = stride_events_cleaned.join(min_vel_events)
    min_vel_event_list["end"] = min_vel_event_list["min_vel"]
    # min_vel is min_vel(-1)
    min_vel_event_list["min_vel"] = min_vel_event_list["min_vel"].shift(1)
    # start = min_vel
    min_vel_event_list["start"] = min_vel_event_list["min_vel"]
    # pre_ic is ic(-1)
    min_vel_event_list["pre_ic"] = min_vel_event_list["ic"].shift(1)

    # drop first and last row
    min_vel_event_list = min_vel_event_list.iloc[1:-1, :]
    min_vel_event_list.index = np.arange(0, len(min_vel_event_list))
    min_vel_event_list.index.name = "s_id"

    return min_vel_event_list


def _get_gait_sequence(min_vel_events: pd.DataFrame) -> pd.DataFrame:

    # find breaks in index
    min_vel_events = min_vel_events.assign(
        cont=np.ediff1d(min_vel_events.index, to_begin=1)
    )

    sequ = min_vel_events[min_vel_events["cont"] > 1]

    if sequ.empty:
        return pd.DataFrame(
            [[min_vel_events.iloc[0]["start"], min_vel_events.iloc[-1]["end"]]],
            columns=["start", "end"],
        )

    min_vel_events["end_shifted"] = min_vel_events["end"].shift(1)

    # get start and end of all "middle" sequences
    sequ = min_vel_events[min_vel_events["cont"] > 1][["start", "end_shifted"]]
    sequ.rename({"end_shifted": "end"}, inplace=True, axis=1)

    # add first start and last end
    sequ = sequ.append(pd.DataFrame([[np.nan, np.nan]], columns=["start", "end"]))
    sequ["start"] = sequ["start"].shift(1, fill_value=min_vel_events.iloc[0]["start"])
    sequ.iloc[-1]["end"] = min_vel_events.iloc[-1]["end"]

    return sequ.set_index(np.arange(len(sequ.index)))
