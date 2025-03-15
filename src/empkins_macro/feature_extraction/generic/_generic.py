from typing import Any, Dict, Sequence

import pandas as pd

from empkins_macro.feature_extraction._utils import _apply_func_per_group, _sanitize_output

_INDEX_LEVELS: Sequence[str] = ["metric", "type", "body_part", "channel", "axis"]
_INDEX_LEVELS_OUT: Sequence[str] = ["body_part", "channel", "type", "metric", "axis"]

# param_dict in the format: {"name-of-channel": [list-of-body-parts], ... }


def max_val(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import max_val

    return_dict = _apply_func_per_group(data, data_format, max_val, param_dict, **kwargs)
    return _sanitize_output(return_dict, "max_val", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def max_val_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import max_val_norm

    return_dict = _apply_func_per_group(data, data_format, max_val_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "max_val", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_max(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_max

    return_dict = _apply_func_per_group(data, data_format, abs_max, param_dict, **kwargs)
    return _sanitize_output(return_dict, "abs_max", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_max_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_max_norm

    return_dict = _apply_func_per_group(data, data_format, abs_max_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "abs_max", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def std(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import std

    return_dict = _apply_func_per_group(data, data_format, std, param_dict, **kwargs)
    return _sanitize_output(return_dict, "std", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def std_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import std_norm

    return_dict = _apply_func_per_group(data, data_format, std_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "std", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean

    return_dict = _apply_func_per_group(data, data_format, mean, param_dict, **kwargs)
    return _sanitize_output(return_dict, "mean", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean_abs(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean_abs

    return_dict = _apply_func_per_group(data, data_format, mean_abs, param_dict, **kwargs)
    return _sanitize_output(return_dict, "abs_mean", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean_norm

    return_dict = _apply_func_per_group(data, data_format, mean_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "mean", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def cov(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import cov

    return_dict = _apply_func_per_group(data, data_format, cov, param_dict, **kwargs)
    return _sanitize_output(return_dict, "cov", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def cov_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import cov_norm

    return_dict = _apply_func_per_group(data, data_format, cov_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "cov", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def entropy(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import entropy

    return_dict = _apply_func_per_group(data, data_format, entropy, param_dict, **kwargs)
    return _sanitize_output(return_dict, "entropy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def entropy_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import entropy_norm

    return_dict = _apply_func_per_group(data, data_format, entropy_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "entropy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def zero_crossings(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import zero_crossings

    return_dict = _apply_func_per_group(data, data_format, zero_crossings, param_dict, **kwargs)
    return _sanitize_output(return_dict, "zero_crossing", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def mean_crossings_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import mean_crossings_norm

    return_dict = _apply_func_per_group(data, data_format, mean_crossings_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "mean_crossing", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_energy(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_energy

    return_dict = _apply_func_per_group(data, data_format, abs_energy, param_dict, **kwargs)
    return _sanitize_output(return_dict, "abs_energy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def abs_energy_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import abs_energy_norm

    return_dict = _apply_func_per_group(data, data_format, abs_energy_norm, param_dict, **kwargs)
    return _sanitize_output(return_dict, "abs_energy", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def fft_aggregated(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import fft_aggregated

    list_return = []
    for param in ["centroid", "variance", "skew", "kurtosis"]:
        return_dict = _apply_func_per_group(data, data_format, fft_aggregated, param_dict, param=param, **kwargs)
        list_return.append(_sanitize_output(return_dict, f"fft_aggregated_{param}", _INDEX_LEVELS, _INDEX_LEVELS_OUT))

    return pd.concat(list_return)


def fft_aggregated_nan_safe(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import fft_aggregated_nan_safe

    list_return = []
    for param in ["centroid", "variance", "skew", "kurtosis"]:
        return_dict = _apply_func_per_group(
            data, data_format, fft_aggregated_nan_safe, param_dict, param=param, **kwargs
        )
        list_return.append(_sanitize_output(return_dict, f"fft_aggregated_{param}", _INDEX_LEVELS, _INDEX_LEVELS_OUT))

    return pd.concat(list_return)


def fft_aggregated_norm(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import fft_aggregated_norm

    list_return = []
    for param in ["centroid", "variance", "skew", "kurtosis"]:
        return_dict = _apply_func_per_group(data, data_format, fft_aggregated_norm, param_dict, param=param, **kwargs)
        list_return.append(_sanitize_output(return_dict, f"fft_aggregated_{param}", _INDEX_LEVELS, _INDEX_LEVELS_OUT))

    return pd.concat(list_return)


def fft_aggregated_norm_nan_safe(
    data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any], **kwargs
) -> pd.DataFrame:
    from empkins_macro.feature_extraction.base_functions import fft_aggregated_norm_nan_safe

    list_return = []
    for param in ["centroid", "variance", "skew", "kurtosis"]:
        return_dict = _apply_func_per_group(
            data, data_format, fft_aggregated_norm_nan_safe, param_dict, param=param, **kwargs
        )
        list_return.append(_sanitize_output(return_dict, f"fft_aggregated_{param}", _INDEX_LEVELS, _INDEX_LEVELS_OUT))

    return pd.concat(list_return)
