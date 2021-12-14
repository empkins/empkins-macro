from typing import Sequence, Tuple, Any, Dict

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_has_columns_any_level

from empkins_macro.utils._types import str_t


def _sanitize_multicolumn_input(data: pd.DataFrame, **kwargs) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
    if kwargs.get("param_dict"):
        data
    if isinstance(data_format, str):
        data_format = [data_format]
    if isinstance(body_parts, str):
        body_parts = [body_parts]
    if isinstance(channel, str):
        channel = [channel]
    _assert_has_columns_any_level(data, [data_format])
    _assert_has_columns_any_level(data, [body_parts])
    _assert_has_columns_any_level(data, [channel])

    return data_format, body_parts, channel


def _sanitize_param_dict(param_dict: Dict[str, Any]) -> Tuple:
    for key, param_dict_sub in param_dict.values():
        assert param_dict_sub.get("body_part"), "key 'body_part' missing in param_dict"
        body_part = param_dict_sub.get("body_part")

    # assert param_dict.get("data_format"), "key 'data_format' missing in param_dict"
    # for
    assert param_dict.get("body_part")
