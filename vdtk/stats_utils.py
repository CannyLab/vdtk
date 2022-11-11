from typing import Any, Sequence, Tuple, TypedDict, Union

import numpy as np

DescrResult = TypedDict(
    "DescrResult",
    {
        "mean": float,
        "median": float,
        "max": float,
        "min": float,
        "stddev": float,
        "q25": float,
        "q75": float,
        "s95ci": Tuple[float, float],
    },
)


def descr(a: Union[Sequence[Any], np.ndarray]) -> DescrResult:
    """Generate descriptive statistics for the array a"""
    return DescrResult(
        mean=float(np.nanmean(a)),
        median=float(np.nanmedian(a)),
        max=float(np.nanmax(a)),
        min=float(np.nanmin(a)),
        stddev=float(np.nanstd(a)),
        q25=float(np.nanpercentile(a, 25)),
        q75=float(np.nanpercentile(a, 75)),
        s95ci=(
            np.nanmean(a) - 1.96 * np.nanstd(a) / np.sqrt(len(a)),
            np.nanmean(a) + 1.96 * np.nanstd(a) / np.sqrt(len(a)),
        ),
    )
