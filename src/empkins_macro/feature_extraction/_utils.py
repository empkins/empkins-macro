import sys
from collections.abc import Callable, Sequence
from typing import Any, get_args

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_has_columns_any_level
from empkins_io.sensors.motion_capture.body_parts import BODY_PART_GROUP, get_all_body_parts, get_body_parts_by_group
from empkins_io.sensors.motion_capture.motion_capture_systems import MOTION_CAPTURE_SYSTEM

from empkins_macro.utils._types import str_t


def _sanitize_multicolumn_input(
    data: pd.DataFrame, data_format: str, param_dict: dict[str, str_t], system: str
) -> dict[tuple, tuple]:
    _assert_has_columns_any_level(data, [[data_format]])
    _assert_has_columns_any_level(data, [list(param_dict)])

    param_dict_out = {}
    for channel in param_dict:
        body_parts = param_dict[channel]
        if isinstance(body_parts, str):
            body_parts = [body_parts]
        body_part_dict = dict([_extract_body_part(system=system, body_parts=body_part) for body_part in body_parts])
        for key, body_parts in body_part_dict.items():
            _assert_has_columns_any_level(data, [body_parts])
            param_dict_out[(key, channel)] = tuple(body_parts)
        param_dict[channel] = body_part_dict

    param_dict_out = {key: (param_dict_out[key], key[1], slice(None)) for key in param_dict_out}

    return param_dict_out


def _sanitize_output(
    data_dict: dict[tuple, pd.Series],
    type_name: str,
    index_names: Sequence[str],
    new_index_order: Sequence[str] | None = None,
    metric_name: str | None = None,
) -> pd.DataFrame:
    if metric_name is None:
        metric_name = type_name

    data = pd.concat(data_dict)
    data = pd.concat({type_name: data})
    data = pd.concat({metric_name: data})
    data.index = data.index.set_names(index_names)
    if new_index_order:
        data = data.reorder_levels(new_index_order)

    return pd.DataFrame(data, columns=["data"])


def _apply_func_per_group(
    data: pd.DataFrame,
    data_format: str,
    func_name: Callable,
    param_dict: dict[str, Any],
    system: MOTION_CAPTURE_SYSTEM,
    **kwargs,
) -> dict[tuple, pd.Series]:
    col_idx_groups = _sanitize_multicolumn_input(data, data_format, param_dict, system)
    data = data.loc[:, data_format]

    return_dict = {}
    for key, col_idxs in col_idx_groups.items():
        data_slice = data.loc[:, col_idxs]
        try:
            res = data_slice.T.groupby(["body_part", "channel"]).apply(lambda df: func_name(df.T, **kwargs))
            res = res.T
            if data_slice.empty:
                raise ValueError("")
        except ValueError as e:
            # If the slice is empty, we get a ValueError. In this case, we skip the slice.
            print(
                f"Feature extraction error when computing {func_name.__name__} for col_idxs {col_idxs}, "
                f"skipping this slice: ",
                e,
                file=sys.stderr,
            )
            continue
        return_dict[key] = res.mean(axis=1)
    return return_dict


def _extract_body_part(system: MOTION_CAPTURE_SYSTEM, body_parts: str | Sequence[str]) -> tuple[str, Sequence[str]]:
    if body_parts is None:
        return "TotalBody", get_all_body_parts(system=system)
    if isinstance(body_parts, list) and len(body_parts) == 1:
        # unwrap singleton list
        body_parts = body_parts[0]
    if isinstance(body_parts, str):
        if body_parts in get_args(BODY_PART_GROUP):
            return body_parts, get_body_parts_by_group(system, body_parts)
        return body_parts, [body_parts]

    return "_".join(body_parts), body_parts
