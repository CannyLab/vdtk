from typing import Any, List

import numpy as np


def baseline_column(
    target_values: List[float],
    baseline_values: List[float],
    aggregate: Any = np.mean,
    baseline_aggregate: Any = np.mean,
    positive: bool = True,
):
    target_metric = aggregate(target_values)
    baseline_metric = baseline_aggregate(baseline_values)
    target_stddev = np.std(target_values)
    baseline_stddev = np.std(baseline_values)
    color = "green" if (target_metric > baseline_metric) == positive else "red"
    sign = "+" if (target_metric > baseline_metric) else "-"
    relative_pct = np.abs(target_metric - baseline_metric) / (baseline_metric + 1e-8) * 100
    return f"[{color}]{target_metric:.4f} ± {target_stddev:.4f} ({sign}{relative_pct:.2f}%) [/{color}]"
