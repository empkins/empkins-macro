import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tpcp import Algorithm
from typing_extensions import Self


__all__ = ["BaseFilter", "ButterworthFilter"]


class BaseFilter(Algorithm):

    _action_methods = ("apply",)

    output_: pd.DataFrame

    def apply(self, data: pd.DataFrame) -> Self:
        raise NotImplementedError("Subclasses must implement the apply method.")


class ButterworthFilter(BaseFilter):

    _action_methods = ("apply",)

    N: int
    Wn: float
    btype: str
    fs: float

    output_: pd.DataFrame

    def __init__(self, *, N: int, Wn: float, btype: str, fs: float):
        self.N = N
        self.Wn = Wn
        self.btype = btype
        self.fs = fs

    def apply(self, data: pd.DataFrame) -> Self:
        sos = butter(self.N, self.Wn, btype=self.btype, fs=self.fs, output="sos")

        output = sosfiltfilt(sos, data, axis=0)
        output = pd.DataFrame(output, columns=data.columns, index=data.index)
        self.output_ = output

        return self
