from inspect import getmembers, isfunction
from typing import Any, Dict, Sequence, Union

import pandas as pd


def extract_generic_features(
    data: pd.DataFrame, feature_dict: Dict[str, Union[Dict[str, Any], Sequence[Dict[str, Any]]]]
) -> pd.DataFrame:
    import empkins_macro.feature_extraction.generic as generic

    feature_funcs = dict(getmembers(generic, isfunction))
    feature_funcs = {key: val for key, val in feature_funcs.items() if not str(key).startswith("_")}
    result_list = []
    for feature_name, param_list in feature_dict.items():
        assert feature_name in feature_funcs.keys(), f"Function {feature_name} not found!"
        if isinstance(param_list, dict):
            param_list = [param_list]
        for param_dict in param_list:
            result_list.append(feature_funcs[feature_name](data=data, **param_dict))

    result_data = pd.concat(result_list)
    result_data = pd.concat({"generic": result_data}, names=["feature_type"])
    result_data = result_data.sort_index()
    return result_data


def extract_expert_features(data: pd.DataFrame, feature_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    import empkins_macro.feature_extraction.body_posture_expert as expert

    feature_funcs = dict(getmembers(expert, isfunction))
    feature_funcs = {key: val for key, val in feature_funcs.items() if not str(key).startswith("_")}

    result_list = []
    for feature_name, param_list in feature_dict.items():
        assert feature_name in feature_funcs.keys(), f"Function {feature_name} not found!"
        if isinstance(param_list, dict):
            param_list = [param_list]
        for param_dict in param_list:
            result_list.append(feature_funcs[feature_name](data=data, **param_dict))

    result_data = pd.concat(result_list)
    result_data = pd.concat({"expert": result_data}, names=["feature_type"])
    result_data = result_data.sort_index()
    return result_data


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
