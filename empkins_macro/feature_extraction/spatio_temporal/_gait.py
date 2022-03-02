import pandas as pd
import numpy as np

from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation

from empkins_macro.feature_extraction._utils import _euler_from_quaternion


def _stride_detection(data: pd.DataFrame) -> pd.DataFrame:

    stride_events = {
        "left": _get_stride_events(data["FootContacts"]["left"]),
        "right": _get_stride_events(data["FootContacts"]["right"]),
    }

    stride_events = {
        "left": _clean_stride_events(data["LeftFoot"], stride_events["left"]),
        "right": _clean_stride_events(data["RightFoot"], stride_events["right"]),
    }

    # TODO add support for non-continous gait sequences

    stride_event_times = {
        "left": _get_stride_event_times(stride_events["left"]),
        "right": _get_stride_event_times(stride_events["right"]),
    }

    min_vel = {
        "left": _get_min_vel(stride_event_times["left"], data["LeftFoot"]),
        "right": _get_min_vel(stride_event_times["right"], data["RightFoot"]),
    }

    min_vel_event_list = {
        "left": _build_min_vel_event_list(stride_event_times["left"], min_vel["left"]),
        "right": _build_min_vel_event_list(
            stride_event_times["right"], min_vel["right"]
        ),
    }

    print(
        "{} left and {} right strides detected.".format(
            len(min_vel_event_list["left"]), len(min_vel_event_list["right"])
        )
    )

    return min_vel_event_list


def _convert_position_and_orientation(
    data: pd.DataFrame, min_vel_event_list: pd.DataFrame
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
    # TODO add support for non-continous gait sequences

    # cut to valid region
    data = data[
        min_vel_event_list.iloc[0]["start"] : min_vel_event_list.iloc[-1]["end"]
    ]
    # omit last sample
    data = data.iloc[:-1]

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

    if len(data) != len(multi_index):
        raise ValueError(
            "Length of index ({}) and data ({}) does not match!".format(
                len(multi_index), len(data)
            )
        )

    data = data.set_index(multi_index)
    data = data.rename(columns={"x": "pos_x", "y": "pos_y", "z": "pos_z"})
    data = data.rename(columns={"q0": "q_w", "q1": "q_x", "q2": "q_y", "q3": "q_z"})

    return data


def temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    stride_list = _stride_detection(data)

    temporal_paras = TemporalParameterCalculation()
    temporal_paras = temporal_paras.calculate(
        stride_event_list=stride_list,
        sampling_rate_hz=1,  # calculations were done in seconds not samples
    )
    return temporal_paras.parameters_


def spatial_features(data: pd.DataFrame) -> pd.DataFrame:
    stride_list = _stride_detection(data)

    # convert to match gaitmap definition
    df_left = _convert_position_and_orientation(data, stride_list["left"])
    df_right = _convert_position_and_orientation(data, stride_list["right"])

    # match to sampling rate
    stride_list["left"][["start", "end", "tc", "ic", "min_vel", "pre_ic"]] *= 60
    stride_list["right"][["start", "end", "tc", "ic", "min_vel", "pre_ic"]] *= 60

    spatial_paras_l = SpatialParameterCalculation()
    spatial_paras_l = spatial_paras_l.calculate(
        stride_event_list=stride_list["left"],
        positions=df_left["LeftFoot"]["pos"],
        orientations=df_left["LeftFoot"]["ori"],
        sampling_rate_hz=60,
    )

    spatial_paras_r = SpatialParameterCalculation()
    spatial_paras_r = spatial_paras_l.calculate(
        stride_event_list=stride_list["left"],
        positions=df_left["LeftFoot"]["pos"],
        orientations=df_left["LeftFoot"]["ori"],
        sampling_rate_hz=60,
    )

    return {"left": spatial_paras_l.parameters_, "right": spatial_paras_r.parameters_}


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


def _find_matching_stride_events(
    stride_events: pd.DataFrame, thres: float = 2
) -> pd.DataFrame:
    diff = stride_events["ic"].count() - stride_events["tc"].count()
    # positive value ic länger als tc

    if diff == 0:
        return stride_events

    # find optimal shift
    shift_diff = np.empty(abs(diff))
    for shift in range(abs(diff)):
        if diff > 0:
            shift_diff[shift] = (
                stride_events["ic"] - stride_events["tc"].shift(shift)
            ).sum()
        else:
            shift_diff[shift] = (
                stride_events["ic"].shift(shift) - stride_events["tc"]
            ).sum()

    min = np.where(shift_diff > 0, shift_diff, np.inf).argmin()

    # perform shift
    if diff > 0:
        stride_events["tc"] = stride_events["tc"].shift(min)
    else:
        stride_events["ic"] = stride_events["ic"].shift(min)

    # delete events without match
    stride_events.dropna(inplace=True)

    # delete events with difference > thres
    stride_events = stride_events[stride_events["ic"] - stride_events["tc"] < thres]

    return stride_events


def _get_stride_event_times(stride_events_cleaned: pd.DataFrame) -> pd.DataFrame:
    ic_times = stride_events_cleaned["ic"]
    ic_times = pd.DataFrame(ic_times[ic_times].index)
    ic_times.columns = ["ic"]

    tc_times = stride_events_cleaned["tc"]
    tc_times = pd.DataFrame(tc_times[tc_times].index)
    tc_times.columns = ["tc"]

    stride_event_times = ic_times.combine_first(tc_times)

    stride_event_times = _find_matching_stride_events(stride_event_times)

    if stride_event_times["tc"].count() != stride_event_times["ic"].count():
        raise ValueError(
            "Number of tc points ({}) does not match number of ic points ({}), manual cleaning needed!".format(
                stride_event_times["tc"].count(), stride_event_times["ic"].count()
            )
        )

    return stride_event_times


def _get_min_vel(
    stride_events_cleaned: pd.DataFrame, segment_data: pd.DataFrame
) -> pd.DataFrame:
    min_vel_list = []
    gyr_norm = segment_data["gyr"].apply(np.linalg.norm, axis=1)

    # shift ic by 1 to get the matching ic and tc times
    stride_events_cleaned = stride_events_cleaned.copy()
    stride_events_cleaned["ic"] = stride_events_cleaned["ic"].shift(1)

    # drop first row
    stride_events_cleaned = stride_events_cleaned.iloc[1:, :]

    for ic, tc in stride_events_cleaned.values:
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
