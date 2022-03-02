import pandas as pd
import numpy as np

from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation


def _stride_detection(data: pd.DataFrame) -> pd.DataFrame:
    stride_events_l = _get_stride_events(data["FootContacts"]["left"])
    stride_events_l_cleaned = _clean_stride_events(data["LeftFoot"], stride_events_l)
    stride_event_times_l = _get_stride_event_times(stride_events_l_cleaned)
    min_vel_l = _get_min_vel(stride_event_times_l, data["LeftFoot"])
    min_vel_event_list_l = _build_min_vel_event_list(stride_event_times_l, min_vel_l)

    return min_vel_event_list_l


def _convert_position_and_orientation(
    data: pd.DataFrame, min_vel_event_list: pd.DataFrame
) -> pd.DataFrame:

    # cut to valid region
    data = data[min_vel_event_list[0, "start"] : min_vel_event_list[-1, "end"]]

    # assign s_id

    return data


def temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    stride_list = _stride_detection(data)

    temporal_paras = TemporalParameterCalculation()
    temporal_paras = temporal_paras.calculate(
        stride_event_list=stride_list, sampling_rate_hz=60
    )
    return temporal_paras.parameters_


def spatial_features(data: pd.DataFrame) -> pd.DataFrame:
    stride_list = _stride_detection(data)

    _convert_position_and_orientation(data, stride_list)

    spatial_paras = SpatialParameterCalculation()
    spatial_paras = spatial_paras.calculate(
        stride_event_list=stride_list, sampling_rate_hz=60
    )
    return spatial_paras.parameters_


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
    mask = segment_data["gyr"].apply(np.linalg.norm, axis=1) > thres

    ic_cleaned = pd.DataFrame(stride_events["ic"] & mask)
    ic_cleaned.columns = ["ic"]
    tc_cleaned = pd.DataFrame(stride_events["tc"] & mask)
    tc_cleaned.columns = ["tc"]

    return ic_cleaned.join(tc_cleaned)


def _get_stride_event_times(stride_events_cleaned: pd.DataFrame) -> pd.DataFrame:
    ic_times = stride_events_cleaned["ic"]
    ic_times = ic_times[ic_times].index

    tc_times = stride_events_cleaned["tc"]
    tc_times = tc_times[tc_times].index

    if len(tc_times) != len(ic_times):
        raise ValueError(
            "Number of tc points does not match number of ic points, manual cleaning needed!"
        )

    return pd.DataFrame(np.array([ic_times, tc_times]).T, columns=["ic", "tc"])


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
