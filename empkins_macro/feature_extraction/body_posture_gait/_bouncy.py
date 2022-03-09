import pandas as pd


def torso_bounce(data: pd.DataFrame) -> pd.DataFrame:
    return _torso_bounce(data)


def _torso_bounce(data: pd.DataFrame) -> pd.DataFrame:
    # as defined by 10.1016/j.humov.2017.11.008: mean vertical velocity of chest

    return data["T8"]["vel"]["z"].mean()
