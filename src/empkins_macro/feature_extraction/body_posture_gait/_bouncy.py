from typing import Any

import pandas as pd

from empkins_macro.feature_extraction._utils import _apply_func_per_group, _sanitize_output
from empkins_macro.feature_extraction.generic._generic import _INDEX_LEVELS, _INDEX_LEVELS_OUT


def torso_bounce(data: pd.DataFrame, data_format: str, param_dict: dict[str, Any]) -> pd.DataFrame:
    return_dict = _apply_func_per_group(data, data_format, _torso_bounce, param_dict)
    return _sanitize_output(return_dict, "abs_max", _INDEX_LEVELS, _INDEX_LEVELS_OUT)


def _torso_bounce(data: pd.DataFrame) -> pd.DataFrame:
    # as defined by 10.1016/j.humov.2017.11.008: mean vertical velocity of chest

    return data["T8"]["vel"]["z"].mean()
