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
            DataFrame containing COP data with 'x' and 'y' columns.

        Returns
        -------
        np.ndarray
            Array of distances between consecutive COP points.
        """
        cop_path = data.assign(distance=np.abs(np.hypot(data["x"].diff(), data["y"].diff()).fillna(0.0)))
        # cop_path = data.assign(distance=np.abs(np.ediff1d(np.linalg.norm(data, axis=1), to_begin=0)))

        return cop_path["distance"].to_numpy()

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
            raise TypeError("psd_data must be a pandas DataFrame containing x and y columns.")

        distances = self._calculate_distances(data)
        self.feature_data_ = pd.DataFrame(distances, columns=["distance"], index=data.index)
        return self
