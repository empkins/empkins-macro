from collections.abc import Sequence
import pandas as pd


def flexion(
        data: pd.DataFrame,
        body_part: Sequence[str],
        data_format: str | None = "global_pose",
        channel: str | None = "pos_global",
        axis: str | None = "norm",
        **kwargs,
) -> pd.DataFrame:
    out = data.loc[:, pd.IndexSlice[data_format, body_part[0], channel, axis]]
    out = out.max() - out.min()
    out = pd.DataFrame({"data": [out]})
    out = pd.concat({body_part[0]: out}, names=["body_part"])
    out = pd.concat({channel: out}, names=["channel"])
    out = pd.concat({"flexion": out}, names=["type"])
    out = pd.concat({"flexion": out}, names=["metric"])
    out = pd.concat({axis: out}, names=["axis"])
    out = out.droplevel(None)
    out = out.reorder_levels(["body_part", "channel", "type", "metric", "axis"])

    return out
