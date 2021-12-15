from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_has_columns_any_level
from empkins_io.sensors.motion_capture.body_parts import BODY_PART_GROUP, get_body_parts_by_group, get_all_body_parts
from itertools import product
from typing_extensions import get_args

from empkins_macro.utils._types import str_t


def _sanitize_multicolumn_input(data: pd.DataFrame, data_format: str, param_dict: Dict[str, str_t]) -> Sequence[Tuple]:
    _assert_has_columns_any_level(data, [[data_format]])

    for channel in param_dict:
        if isinstance(param_dict[channel], str):
            param_dict[channel] = [param_dict[channel]]
        _assert_has_columns_any_level(data, [param_dict[channel]])

    param_list = [[v, [k]] for k, v in param_dict.items()]
    param_list = [list(product(*v)) for v in param_list]
    param_list = tuple((*l, slice(None)) for v in param_list for l in v)
    return param_list


def _sanitize_output(
    data_dict: Dict[str, pd.Series],
    type_name: str,
    index_names: Sequence[str],
    new_index_order: Optional[Sequence[str]] = None,
    metric_name: Optional[str] = None,
) -> pd.DataFrame:
    if metric_name is None:
        metric_name = type_name

    data = pd.concat(data_dict)
    data = pd.concat({type_name: data})
    data = pd.concat({metric_name: data})
    data.index.set_names(index_names, inplace=True)
    if new_index_order:
        data = data.reorder_levels(new_index_order)

    return pd.DataFrame(data, columns=["data"])


def _apply_func_per_group(
    data: pd.DataFrame, data_format: str, func_name: Callable, param_dict: Dict[str, Any], **kwargs
) -> Dict[str, pd.Series]:

    col_idx_groups = _sanitize_multicolumn_input(data, data_format, param_dict)
    data = data.loc[:, data_format]

    return_dict = {}
    for col_idxs in col_idx_groups:
        data_slice = data.loc[:, col_idxs]
        return_dict[col_idxs[:-1]] = func_name(data_slice, **kwargs)

    return return_dict


def _extract_body_part(
    body_parts: Union[str, Sequence[str]],
) -> Tuple[str, Sequence[str]]:
    if body_parts is None:
        return "TotalBody", get_all_body_parts()
    if isinstance(body_parts, str):
        if body_parts in get_args(BODY_PART_GROUP):
            return body_parts, get_body_parts_by_group(body_parts)
        return body_parts, [body_parts]

    return "_".join(body_parts), body_parts
