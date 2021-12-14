import numpy as np
import pandas as pd


def std_norm(data: pd.DataFrame) -> pd.DataFrame:
    out = np.std(np.linalg.norm(data, axis=1))
    out = pd.DataFrame(out, index=data.index, columns=[])
    return out
