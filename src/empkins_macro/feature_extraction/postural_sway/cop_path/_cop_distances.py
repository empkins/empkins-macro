# Module for COP distance calculation between samples.

import numpy as np
import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self


class CopDistanceCalculation(Algorithm):
    _action_methods = ("apply",)

    feature_data_: pd.DataFrame

    def __init__(self):
        """
        Initializes the CopDistanceCalculation algorithm.
        """
        self.feature_data_ = pd.DataFrame()

    @staticmethod
    def _calculate_distances(data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the Euclidean distances between consecutive COP data points.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing COP data with 'x' and 'y' columns or only a single column.

        Returns
        -------
        np.ndarray
            Array of distances between consecutive COP points.
        """
        # if data has only one column, return absolute differences
        if data.shape[1] == 1:
            dist = data.iloc[:, 0].diff().abs().fillna(0.0)  # Series
            return dist.to_numpy()
        elif data.shape[1] == 2:
            dx = data["x"].diff()
            dy = data["y"].diff()
            dist = pd.Series(np.hypot(dx, dy), index=data.index).fillna(0.0)
            return dist.to_numpy()
        else:
            raise ValueError("DataFrame must contain at least one of the 'x' or 'y' columns.")



    def apply(self, data: pd.DataFrame) -> Self:
        """
        Run the COP distance calculation algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing COP data with 'x' and 'y' columns.

        Returns
        -------
        np.ndarray
            Array of distances between consecutive COP points.
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

        distances = type(self)._calculate_distances(data=cop_data)
        self.feature_data_ = pd.DataFrame(distances, columns=["distance"], index=cop_data.index)

        return self
