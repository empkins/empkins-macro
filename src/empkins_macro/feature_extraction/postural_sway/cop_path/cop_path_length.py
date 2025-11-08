# Module for COP path length calculation

import numpy as np
import pandas as pd
from tpcp import Algorithm
from empkins_macro.feature_extraction.postural_sway.cop_path._cop_distances import CopDistanceCalculation
from typing_extensions import Self


class CopLengthCalculation(Algorithm):
    _action_methods = ("apply",)

    feature_data_: pd.DataFrame

    def __init__(self):
        """
        Initializes the CopLengthCalculation algorithm.
        """
        self.feature_data_ = pd.DataFrame()

    @staticmethod
    def _calculate_length(data: pd.DataFrame) -> float:

        distance_algo = CopDistanceCalculation()
        distance_algo.apply(data=data)
        dist_series = distance_algo.feature_data_["distance"]

        path_length = float(np.nansum(dist_series.to_numpy()))
        return path_length

    def apply(self, data: pd.DataFrame) -> Self:
        """
        Run the COP length calculation algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing COP data with 'x' and 'y' columns.

        Returns
        -------
        np.ndarray
            Sum of distances between consecutive COP points = COP path length.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame containing x and y or only one x or y column.")

        cop_data = data.dropna()

        # flatten multi index columns if present
        if isinstance(cop_data.columns, pd.MultiIndex):
            cop_data.columns = ['_'.join(map(str, col)).strip() for col in cop_data.columns.values]

        rename_x = {c: "x" for c in cop_data.columns if c.endswith("_x")}
        rename_y = {c: "y" for c in cop_data.columns if c.endswith("_y")}
        cop_data = cop_data.rename(columns={**rename_x, **rename_y})

        if not (("x" in cop_data.columns) or ("y" in cop_data.columns)):
            raise ValueError("Expected at least 'x' or 'y' in data columns.")

        cop_path_length = type(self)._calculate_length(data=cop_data)
        self.feature_data_ = pd.DataFrame({"path_length": [cop_path_length]})
        return self