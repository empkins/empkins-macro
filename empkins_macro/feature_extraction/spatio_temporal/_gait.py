from typing import Dict

import numpy as np
import pandas as pd
from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation


class StrideDetection:
    data = pd.DataFrame
    _min_vel_event_list: Dict[str, pd.DataFrame]
    _sequence_list: Dict[str, pd.DataFrame]
    temporal_features: Dict[str, pd.DataFrame]
    spatial_features: Dict[str, pd.DataFrame]

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._stride_detection()

    def temporal_features(self, sampling_rate: float = 1):
        temporal_paras = TemporalParameterCalculation()
        temporal_paras = temporal_paras.calculate(
            stride_event_list=self._min_vel_event_list,
            sampling_rate_hz=sampling_rate,  # calculations were done in seconds not samples
        )

        self.temporal_features = temporal_paras.parameters_

    def spatial_features(self, sampling_rate: float = 60, turn_thres: float = 100):
        # convert to match gaitmap definition
        df_left = _convert_position_and_orientation(
            self.data["LeftFoot"],
            self._min_vel_event_list["left"],
            _get_gait_sequence(self._min_vel_event_list["left"]),
        )
        df_right = _convert_position_and_orientation(
            self.data["RightFoot"],
            self._min_vel_event_list["right"],
            _get_gait_sequence(self._min_vel_event_list["right"]),
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

        # TODO omit temporal with turning angle > 100 as well
        self.spatial_features = {
            "left": spatial_paras_l.parameters_[
                abs(spatial_paras_l.parameters_["turning_angle"]) < turn_thres
            ],
            "right": spatial_paras_r.parameters_[
                abs(spatial_paras_r.parameters_["turning_angle"]) < turn_thres
            ],
        }

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
            ).dropna(),
            "right": _build_min_vel_event_list(
                stride_event_times["right"], min_vel["right"]
            ).dropna(),
        }

        print(
            "{} left and {} right strides detected.".format(
                len(min_vel_event_list["left"]), len(min_vel_event_list["right"])
            )
        )

        self._min_vel_event_list = min_vel_event_list


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


def _get_stride_events(foot_contacts: pd.DataFrame) -> pd.DataFrame:
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

    if diff == 0 & positive.count() == 0:
        return stride_events

    stride_events["tc_shifted"] = stride_events["tc"].shift(-1)

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

    sequ = min_vel_events[min_vel_events["cont"] > 1][["start", "end_shifted"]]
    sequ.rename({"end_shifted": "end"}, inplace=True, axis=1)

    sequ = sequ.append(pd.DataFrame([[np.nan, np.nan]], columns=["start", "end"]))
    sequ["start"] = sequ["start"].shift(1, fill_value=min_vel_events.iloc[0]["start"])
    sequ.iloc[-1]["end"] = min_vel_events.iloc[-1]["end"]

    return sequ.set_index(np.arange(len(sequ.index)))
