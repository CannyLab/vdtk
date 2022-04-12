from typing import Any, Dict, List, Tuple, Union

import numpy as np


def descr(a: Union[List[Any], np.ndarray]) -> Dict[str, Union[float, Tuple[float, float]]]:
    """Generate descriptive statistics for the array a"""
    return {
        "mean": np.nanmean(a),
        "median": np.nanmedian(a),
        "max": np.nanmax(a),
        "min": np.nanmin(a),
        "stddev": np.nanstd(a),
        "25q": np.quantile(a, 0.25),
        "75q": np.quantile(a, 0.75),
        "s95ci": (
            np.nanmean(a) - 1.96 * np.nanstd(a) / np.sqrt(len(a)),
            np.nanmean(a) + 1.96 * np.nanstd(a) / np.sqrt(len(a)),
        ),
    }
