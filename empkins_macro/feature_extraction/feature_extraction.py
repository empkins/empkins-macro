from inspect import getmembers, isfunction
from typing import Any, Dict, Sequence, Union, Optional

import pandas as pd

from empkins_macro.feature_extraction.spatio_temporal import StrideDetection
from empkins_macro.feature_extraction.tug._tug import TUG

param_dict_gait_all = {
    "stride_time": ["TemporalFeatures"],
    "stance_time": ["TemporalFeatures"],
    "swing_time": ["TemporalFeatures"],
    "stride_length": ["SpatialFeatures"],
    "gait_velocity": ["SpatialFeatures"],
    "ic_angle": ["SpatialFeatures"],
    "tc_angle": ["SpatialFeatures"],
    "turning_angle": ["SpatialFeatures"],
    "arc_length": ["SpatialFeatures"],
    "max_sensor_lift": ["SpatialFeatures"],
    "max_lateral_excursion": ["SpatialFeatures"],
    "max_knee_flexion": ["SpatialFeatures"],
    "min_knee_flexion": ["SpatialFeatures"],
    "max_arm_flexion": ["SpatialFeatures"],
    "min_arm_flexion": ["SpatialFeatures"],
    "cadence": ["SpatialFeatures"],
}

feature_dict_all = {
    "mean": [
        {"data_format": "gait_features", "param_dict": param_dict_gait_all},
    ],
    "std": [
        {"data_format": "gait_features", "param_dict": param_dict_gait_all},
    ],
}


def extract_generic_features(
    data: pd.DataFrame,
    feature_dict: Dict[str, Union[Dict[str, Any], Sequence[Dict[str, Any]]]],
) -> pd.DataFrame:
    import empkins_macro.feature_extraction.generic as generic

    feature_funcs = dict(getmembers(generic, isfunction))
    feature_funcs = {
        key: val for key, val in feature_funcs.items() if not str(key).startswith("_")
    }
    result_list = []
    for feature_name, param_list in feature_dict.items():
        assert (
            feature_name in feature_funcs.keys()
        ), f"Function {feature_name} not found!"
        if isinstance(param_list, dict):
            param_list = [param_list]
        for param_dict in param_list:
            result_list.append(feature_funcs[feature_name](data=data, **param_dict))

    result_data = pd.concat(result_list)
    result_data = pd.concat({"generic": result_data}, names=["feature_type"])
    result_data = result_data.sort_index()
    return result_data


def extract_expert_features(
    data: pd.DataFrame, feature_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    import empkins_macro.feature_extraction.body_posture_expert as expert

    feature_funcs = dict(getmembers(expert, isfunction))
    feature_funcs = {
        key: val for key, val in feature_funcs.items() if not str(key).startswith("_")
    }

    result_list = []
    for feature_name, param_list in feature_dict.items():
        assert (
            feature_name in feature_funcs.keys()
        ), f"Function {feature_name} not found!"
        if isinstance(param_list, dict):
            param_list = [param_list]
        for param_dict in param_list:
            result_list.append(feature_funcs[feature_name](data=data, **param_dict))

    result_data = pd.concat(result_list)
    result_data = pd.concat({"expert": result_data}, names=["feature_type"])
    result_data = result_data.sort_index()
    return result_data


def extract_spatio_temporal_features(
    data: pd.DataFrame, feature_dict: Optional[Dict[str, Dict[str, Any]]] = None
) -> (pd.DataFrame, StrideDetection):
    import empkins_macro.feature_extraction.spatio_temporal as spatio_temporal
    import empkins_macro.feature_extraction.generic as generic

    stride_detection = spatio_temporal.StrideDetection(data)
    stride_detection.calc_spatio_temporal_features()

    feature_funcs = dict(getmembers(generic, isfunction))
    feature_funcs = {
        key: val for key, val in feature_funcs.items() if not str(key).startswith("_")
    }

    result_list = []

    if feature_dict == None:
        feature_dict = feature_dict_all

    for feature_name, param_list in feature_dict.items():
        assert (
            feature_name in feature_funcs.keys()
        ), f"Function {feature_name} not found!"
        if isinstance(param_list, dict):
            param_list = [param_list]
        for param_dict in param_list:
            result_list.append(
                feature_funcs[feature_name](
                    data=stride_detection.features, **param_dict
                )
            )

    result_data = pd.concat(result_list)
    result_data = pd.concat({"gait": result_data}, names=["feature_type"])
    result_data = result_data.sort_index()

    return result_data, stride_detection


def stride_detection(data: pd.DataFrame) -> StrideDetection:
    return StrideDetection(data)


def extract_tug_features(data: pd.DataFrame) -> pd.DataFrame:

    tug = TUG(data)
    tug.extract_tug_features()

    return tug.features


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean extracted features and drop features that did not produce any output (i.e., NaN output).

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with extracted features

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with extracted features and dropped NaN values

    """
    index_order = data.index.names
    data = data.unstack("condition").dropna(how="any").stack("condition")
    data = data.reorder_levels(index_order).sort_index()
    return data
