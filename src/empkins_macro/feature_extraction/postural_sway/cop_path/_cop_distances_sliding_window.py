# Module for sliding window path length calculation of postural cop data.

from typing import Any

import numpy as np
import pandas as pd
from biopsykit.utils.array_handling import sliding_window
from tpcp import Algorithm, Parameter
from typing_extensions import Self

from empkins_macro.feature_extraction.postural_sway.cop_path import CopDistanceCalculation
from empkins_macro.feature_extraction.postural_sway.utils import BaseFilter


class CopDistanceCalculationSlidingWindow(Algorithm):
    _action_methods = ("apply",)

    window_size_sec: Parameter[int]
    overlap_percent: Parameter[float]
    prepro_algo: Parameter[BaseFilter | None]
    distance_algo: Parameter[CopDistanceCalculation | None]


    feature_data_: pd.DataFrame

    def __init__(
        self,
        *,
        window_size_sec: int = 60,
        overlap_percent: float = 75,
        prepro_algo: BaseFilter| None = None,
        distance_algo: CopDistanceCalculation | None = None,
    ):
        """
        Initializes the SlidingWindowDistanceCalculation algorithm.
        """
        self.window_size_sec = window_size_sec
        self.overlap_percent = overlap_percent
        self.prepro_algo = prepro_algo # may be None or an Algorithm-like object with .apply/.output_
        self.distance_algo = distance_algo # may be None or any Algorithm with .apply() + feature_data_

    @staticmethod
    def _sliding_window_path_length(
        cop_data: pd.DataFrame| pd.Series, sampling_rate_hz: float, window_size_sec: int, overlap_percent: float
    ) -> list[Any]:
        """
        Calculate the sum of COP distances within sliding windows.
        """

        if isinstance(cop_data, pd.DataFrame):
            distances = cop_data["distance"] if "distance" in cop_data.columns else cop_data.iloc[:, 0]
        else:
            distances = pd.Series(cop_data)

        sw = sliding_window(
            data=distances,
            sampling_rate=sampling_rate_hz,
            window_sec=window_size_sec,
            overlap_percent=overlap_percent
        )

        sum_distances = []
        for i, window in enumerate(sw):
            window_distance = np.nansum(window)
            sum_distances.append({"window_index": i, "distance_sum": window_distance})

        return sum_distances

    def apply(
            self,
            cop_data: pd.DataFrame,
            sampling_rate_hz: float) -> Self:
        """Apply the sliding window path length calculation algorithm.

        Parameters
        ----------
        cop_data : pd.DataFrame
            DataFrame containing COP distance data with 'x', 'y' and 'distance' column.
        sampling_rate_hz : float
            Sampling rate in Hz.
        window_seconds : int
            Length of each sliding window in seconds.
        overlap_percent : float
            Overlap percentage between consecutive windows.

        Returns
        -------
        list[Any]
            List of dictionaries containing window index and distance sum for each window.
        """

        if not isinstance(cop_data, pd.DataFrame):
            raise TypeError("cop_data must be a pandas DataFrame!")

        if self.prepro_algo is not None:
            self.prepro_algo.apply(data=cop_data)

        if self.distance_algo is None:
            self.distance_algo = CopDistanceCalculation()

        self.distance_algo.apply(data = cop_data)
        cop_dist_df = self.distance_algo.feature_data_

        distance_sums = self._sliding_window_path_length(
            cop_data=cop_dist_df,
            sampling_rate_hz=sampling_rate_hz,
            window_size_sec=self.window_size_sec,
            overlap_percent=self.overlap_percent,
        )
        self.feature_data_ = pd.DataFrame(distance_sums).set_index("window_index")

        return self
