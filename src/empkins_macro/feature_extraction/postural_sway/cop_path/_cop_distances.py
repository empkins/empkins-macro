# Module for COP distance calculation between samples.

import numpy as np
import pandas as pd
from tpcp import Algorithm


class CopDistanceCalculation(Algorithm):
    _action_methods = "run"

    feature_data_: pd.DataFrame

    def __init__(self):
        """
        Initializes the CopDistanceCalculation algorithm.
        """
        self.feature_data_ = pd.DataFrame()

    def _calculate_distances(self, cop_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate the Euclidean distances between consecutive COP data points.

        Parameters
        ----------
        cop_data : pd.DataFrame
            DataFrame containing COP data with 'x' and 'y' columns.

        Returns
        -------
        np.ndarray
            Array of distances between consecutive COP points.
        """
        cop_path = cop_data.assign(distance=np.hypot(cop_data['x'].diff(), cop_data['y'].diff()).fillna(0.0))

        return cop_path['distance'].to_numpy()

    def run(self, cop_data: pd.DataFrame) -> np.ndarray:
        """
        Run the COP distance calculation algorithm.

        Parameters
        ----------
        cop_data : pd.DataFrame
            DataFrame containing COP data with 'x' and 'y' columns.

        Returns
        -------
        np.ndarray
            Array of distances between consecutive COP points.
        """

        if not isinstance(cop_data, pd.DataFrame):
            raise TypeError("psd_data must be a pandas DataFrame containing x and y columns.")

        distances = self._calculate_distances(cop_data)
        self.feature_data_ = pd.DataFrame(distances, columns=['distance'])
        return distances
