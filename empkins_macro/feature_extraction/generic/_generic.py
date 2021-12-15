from typing import Any, Dict, Sequence

import pandas as pd

from empkins_macro.feature_extraction._utils import _apply_func_per_group, _sanitize_output

_INDEX_LEVELS: Sequence[str] = ["metric", "type", "body_part", "channel", "axis"]
_INDEX_LEVELS_OUT: Sequence[str] = ["body_part", "channel", "type", "metric", "axis"]

# param_dict in the format: {"name-of-channel": [list-of-body-parts], ... }


def max_val(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import max_val

    return_dict = _apply_func_per_group(data, max_val, data_format, param_dict)
    return _sanitize_output(return_dict, "max_val", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def max_val_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import max_val_norm

    return_dict = _apply_func_per_group(data, max_val_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "max_val", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_max(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_max

    return_dict = _apply_func_per_group(data, abs_max, data_format, param_dict)
    return _sanitize_output(return_dict, "abs_max", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_max_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_max_norm

    return_dict = _apply_func_per_group(data, abs_max_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "abs_max", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def std(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import std

    return_dict = _apply_func_per_group(data, std, data_format, param_dict)
    return _sanitize_output(return_dict, "std", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def std_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import std_norm

    return_dict = _apply_func_per_group(data, std_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "std", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean

    return_dict = _apply_func_per_group(data, mean, data_format, param_dict)
    return _sanitize_output(return_dict, "mean", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean_abs(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean_abs

    return_dict = _apply_func_per_group(data, mean_abs, data_format, param_dict)
    return _sanitize_output(return_dict, "abs_mean", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean_norm

    return_dict = _apply_func_per_group(data, mean_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "mean", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def cov(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import cov

    return_dict = _apply_func_per_group(data, cov, data_format, param_dict)
    return _sanitize_output(return_dict, "cov", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def cov_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import cov_norm

    return_dict = _apply_func_per_group(data, cov_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "cov", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def entropy(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import entropy

    return_dict = _apply_func_per_group(data, entropy, data_format, param_dict)
    return _sanitize_output(return_dict, "entropy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def entropy_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import entropy_norm

    return_dict = _apply_func_per_group(data, entropy_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "entropy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def zero_crossings(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import zero_crossings

    return_dict = _apply_func_per_group(data, zero_crossings, data_format, param_dict)
    return _sanitize_output(return_dict, "zero_crossing", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean_crossings_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean_crossings_norm

    return_dict = _apply_func_per_group(data, mean_crossings_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "mean_crossing", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_energy(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_energy

    return_dict = _apply_func_per_group(data, abs_energy, data_format, param_dict)
    return _sanitize_output(return_dict, "abs_energy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_energy_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_energy_norm

    return_dict = _apply_func_per_group(data, abs_energy_norm, data_format, param_dict)
    return _sanitize_output(return_dict, "abs_energy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def fft_aggregated(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import fft_aggregated

    list_return = []
    for param in ["centroid", "variance", "skew", "kurtosis"]:
        return_dict = _apply_func_per_group(data, fft_aggregated, data_format, param_dict, param=param)
        list_return.append(_sanitize_output(return_dict, f"fft_aggregated_{param}", _INDEX_LEVELS, _INDEX_LEVELS_OUT))

    return pd.concat(list_return)


def fft_aggregated_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import fft_aggregated_norm

    list_return = []
    for param in ["centroid", "variance", "skew", "kurtosis"]:
        return_dict = _apply_func_per_group(data, fft_aggregated_norm, data_format, param_dict, param=param)
        list_return.append(_sanitize_output(return_dict, f"fft_aggregated_{param}", _INDEX_LEVELS, _INDEX_LEVELS_OUT))

    return pd.concat(list_return)
