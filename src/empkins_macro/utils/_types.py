"""Some custom helper types to make type hints and type checking easier."""

from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
import pandas as pd

_Hashable = Union[Hashable, str]

path_t = TypeVar("path_t", str, Path)  # pylint:disable=invalid-name
arr_t = TypeVar("arr_t", pd.DataFrame, pd.Series, np.ndarray)  # pylint:disable=invalid-name
str_t = TypeVar("str_t", str, Sequence[str])  # pylint:disable=invalid-name
T = TypeVar("T")


def _check_file_exists(file_path: path_t) -> None:
    # ensure pathlib
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"No file {file_path.absolute()} exists!")
