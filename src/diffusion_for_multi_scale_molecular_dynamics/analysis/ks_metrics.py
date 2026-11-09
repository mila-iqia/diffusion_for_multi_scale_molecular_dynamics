from typing import Tuple

import numpy as np
from scipy.stats import ks_2samp


def ks_metrics(dataset1: np.ndarray, dataset2: np.ndarray) -> Tuple[float, float]:
    """Compare two datasets with a two-sample Kolmogorov-Smirnov test.

    Args:
        dataset1: first dataset.
        dataset2: second dataset.

    Returns:
        ks_distance, p_value: the Kolmogorov-Smirnov test statistic (a "distance") and the test's p-value.
    """
    test_result = ks_2samp(dataset1, dataset2, alternative="two-sided", method="auto")
    return test_result.statistic, test_result.pvalue
