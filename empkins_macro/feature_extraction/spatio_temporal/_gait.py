from typing import Any, Dict, Sequence

import pandas as pd
import gaitmap as gm

from empkins_macro.feature_extraction._utils import _apply_func_per_group, _sanitize_output

_INDEX_LEVELS: Sequence[str] = ["metric", "type", "body_part", "channel", "axis"]
_INDEX_LEVELS_OUT: Sequence[str] = ["body_part", "channel", "type", "metric", "axis"]

def todo(data: pd.DataFrame, data_format: str, param_dict: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame
