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
        negative_score_weight: float = 0.5,
        sigma_threshold: float = 0.5,
    ):
        """__init__.

        Args:
            positive_score_network: score network trained with normal (positive) samples
            negative_score_network: score network trained with negative samples
            negative_score_weight: prefactor weighting the negative score network
            sigma_threshold: only apply the negative model to sigma smaller than this value
        """
        self.positive_score_network = positive_score_network
        self.negative_score_network = negative_score_network
        self.negative_score_weight = negative_score_weight
        self.sigma_threshold = sigma_threshold

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

        negative_score_weight = self.negative_score_weight * sigma_cutoff

        coord_scores = ((1 - negative_score_weight) * positive_scores.X
                        - negative_score_weight * negative_scores.X)

        axl_scores = AXL(
            A=positive_scores.A,
            X=coord_scores,
            L=positive_scores.L,
        )

        return axl_scores
