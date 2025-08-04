# Module for rotary spectral analysis (RSA) of postural sway data.
from typing import Self

import pandas as pd
from tpcp import Algorithm


class RotarySpectralAnalysis(Algorithm):

    _action_methods = "run"

    rsa_data_: pd.DataFrame

    def __init__(self):
        """Initializes the RotarySpectralAnalysis algorithm.
        """
        pass



    def run(self, data: pd.DataFrame) -> Self:
        self.rsa_data_ = data
        return self