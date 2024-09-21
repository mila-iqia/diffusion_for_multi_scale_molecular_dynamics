from typing import Tuple

import scipy.stats as ss
from torchmetrics import CatMetric


class KolmogorovSmirnovMetrics:
    """Kolmogorov Smirnov metrics."""

    def __init__(self):
        """Init method."""
        self.reference_samples_metric = CatMetric()
        self.predicted_samples_metric = CatMetric()

    def register_reference_samples(self, reference_samples):
        """Register reference samples."""
        self.reference_samples_metric.update(reference_samples)

    def register_predicted_samples(self, predicted_samples):
        """Register predicted samples."""
        self.predicted_samples_metric.update(predicted_samples)

    def reset(self):
        """reset."""
        self.reference_samples_metric.reset()
        self.predicted_samples_metric.reset()

    def compute_kolmogorov_smirnov_distance_and_pvalue(self) -> Tuple[float, float]:
        """Compute Kolmogorov Smirnov Distance.

        Compute the two sample Kolmogorov–Smirnov test in order to gauge whether the
        predicted samples were drawn from the same distribution as the reference samples.

        Args:
            predicted_samples : samples drawn from the diffusion model.
            reference_samples : samples drawn from the reference distribution.

        Returns:
            ks_distance, p_value: the Kolmogorov-Smirnov test statistic (a "distance")
                and the statistical test's p-value.
        """
        reference_samples = self.reference_samples_metric.compute()
        predicted_samples = self.predicted_samples_metric.compute()

        test_result = ss.ks_2samp(predicted_samples.detach().cpu().numpy(),
                                  reference_samples.detach().cpu().numpy(),
                                  alternative='two-sided', method='auto')

        # The "test statistic" of the two-sided KS test is the largest vertical distance between
        # the empirical CDFs of the two samples. The larger this is, the less likely the two
        # samples were drawn from the same underlying distribution, hence the idea of 'distance'.
        ks_distance = test_result.statistic

        # The null hypothesis of the KS test is that both samples are drawn from the same distribution.
        # Thus, a small p-value (which leads to the rejection of the null hypothesis) indicates that
        # the samples probably come from different distributions (ie, our samples are bad!).
        p_value = test_result.pvalue
        return ks_distance, p_value
