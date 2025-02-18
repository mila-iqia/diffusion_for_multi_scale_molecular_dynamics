from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL, NOISE


class ScoreNetworkDifference:
    """Score network based on analytical integration of Gaussian distributions.

    This 'score network' is for exploring and debugging.
    """

    def __init__(
        self,
        positive_score_network: ScoreNetwork,
        negative_score_network: ScoreNetwork,
        weight: float = 1,
        sigma_threshold: float = 0.5,
        average_scores: bool = False
    ):
        """__init__.

        Args:
            positive_score_network: score network trained with normal (positive) samples
            negative_score_network: score network trained with negative samples
            weight: prefactor weighting the negative score network
            sigma_threshold: only apply the negative model to sigma smaller than this value
            average_scores: if True, take the average of the scores, not the sum
        """
        self.positive_score_network = positive_score_network
        self.negative_score_network = negative_score_network
        self.weight = weight
        self.sigma_threshold = sigma_threshold
        self.use_average = average_scores

    def __call__(self, batch: Dict[AnyStr, Any], conditional: bool = False) -> AXL:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): CURRENTLY DOES NOTHING.

        Returns:
            output : an AXL namedtuple with
                    - the coordinates scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
                    - perfect atom type predictions, assuming a single atom type possibility.
                    - a tensor of zeros for the lattice parameters.
        """
        positive_scores = self.positive_score_network(batch)
        negative_scores = self.negative_score_network(batch)

        sigma_cutoff = (
            (batch[NOISE] <= self.sigma_threshold).to(negative_scores.X).unsqueeze(-1)
        )  # shape (batchsize, 1, 1)

        coord_scores = positive_scores.X - self.weight * sigma_cutoff * negative_scores.X

        if self.use_average:
            coord_scores *= 0.5

        axl_scores = AXL(
            A=positive_scores.A,
            X=coord_scores,
            L=positive_scores.L,
        )

        return axl_scores
