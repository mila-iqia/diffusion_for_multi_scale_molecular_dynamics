import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


class AdaptativeCorrectorGenerator(LangevinGenerator):
    """Langevin Dynamics Generator using only a corrector step with adaptative step size for relative coordinates.

    This class implements the Langevin Corrector generation of position samples, following
    Song et. al. 2021, namely:
        "SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS"
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        axl_network: ScoreNetwork,
    ):
        """Init method."""
        super().__init__(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )
        self.corrector_r = noise_parameters.corrector_r

    def predictor_step(
        self,
        composition_i: AXL,
        index_i: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Predictor step.

        Args:
            composition_i : sampled composition (atom types, relative coordinates, lattice vectors), at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the sampling process

        Returns:
            composition_im1 : sampled composition, at time step i - 1.
        """
        assert (
            1 <= index_i <= self.number_of_discretization_steps
        ), "The predictor step can only be invoked for index_i between 1 and the total number of discretization steps."

        idx = index_i - 1  # python starts indices at zero
        t_i = self.noise.time[idx].to(composition_i.X)
        sigma_i = self.noise.sigma[idx].to(composition_i.X)
        q_matrices_i = self.noise.q_matrix[idx].to(composition_i.X)
        q_bar_matrices_i = self.noise.q_bar_matrix[idx].to(composition_i.X)
        q_bar_tm1_matrices_i = self.noise.q_bar_tm1_matrix[idx].to(composition_i.X)

        model_predictions_i = self._get_model_predictions(
            composition_i, t_i, sigma_i, unit_cell, cartesian_forces
        )

        # atom types update
        a_im1 = self.atom_types_update(
            model_predictions_i.A,
            composition_i.A,
            q_matrices_i,
            q_bar_matrices_i,
            q_bar_tm1_matrices_i,
        )

        # in this approach, there is no predictor step applied on the X component
        composition_im1 = AXL(
            A=a_im1, X=composition_i.X, L=unit_cell
        )  # TODO : Deal with L correctly

        if self.record:
            # TODO : Deal with L correctly
            composition_i_for_recording = AXL(
                A=composition_i.A, X=composition_i.X, L=unit_cell
            )
            # Keep the record on the CPU
            entry = dict(time_step_index=index_i)
            list_keys = ["composition_i", "composition_im1", "model_predictions_i"]
            list_axl = [
                composition_i_for_recording,
                composition_im1,
                model_predictions_i,
            ]

            for key, axl in zip(list_keys, list_axl):
                record_axl = AXL(
                    A=axl.A.detach().cpu(),
                    X=axl.X.detach().cpu(),
                    L=axl.L.detach().cpu(),
                )
                entry[key] = record_axl
            self.sample_trajectory_recorder.record(key="predictor_step", entry=entry)

        return composition_im1

    def corrector_step(
        self,
        composition_i: AXL,
        index_i: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        r"""Corrector Step.

        Note this does not affect the atom types unless specified with the atom_type_transition_in_corrector argument.
        Always affect the reduced coordinates and lattice vectors. The prefactors determining the changes in the X and L
        variables are determined using the sigma normalized score at that corrector step. The relative coordinates
        update is given by:

        .. math::

            x_i \leftarrow x_i + \epsilon_i * s(x_i, t_i) + \sqrt(2 \epsilon_i) z

        where :math:`s(x_i, t_i)` is the score, :math:`z` is a random variable drawn from a normal distribution and
        :math:`\epsilon_i` is given by:

        .. math::

            \epsilon_i = 2 \left(r \frac{||z||_2}{||s(x_i, t_i)||_2}\right)^2

        where :math:`r` is an hyper-parameter (0.15 by default) and :math:`||\cdot||_2` is the L2 norm.

        Args:
            composition_i : sampled composition (atom types, relative coordinates, lattice vectors), at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.  # TODO replace with AXL-L
            cartesian_forces: forces conditioning the sampling

        Returns:
            corrected_composition_i : sampled composition, after corrector step.
        """
        assert 0 <= index_i <= self.number_of_discretization_steps - 1, (
            "The corrector step can only be invoked for index_i between 0 and "
            "the total number of discretization steps minus 1."
        )

        if index_i == 0:
            # TODO: we are extrapolating here; the score network will never have seen this time step...
            sigma_i = (
                self.noise_parameters.sigma_min
            )  # no need to change device, this is a float
            t_i = 0.0  # same for device - this is a float
            idx = index_i
        else:
            idx = index_i - 1  # python starts indices at zero
            sigma_i = self.noise.sigma[idx].to(composition_i.X)
            t_i = self.noise.time[idx].to(composition_i.X)

        model_predictions_i = self._get_model_predictions(
            composition_i, t_i, sigma_i, unit_cell, cartesian_forces
        )

        # to compute epsilon_i, we need the norm of the score. We average over the atoms.
        relative_coordinates_sigma_score_norm = (
            torch.linalg.norm(model_predictions_i.X, dim=-1).mean(dim=-1)
        ).view(-1, 1, 1)
        # draw random noise
        z = self._draw_gaussian_sample(relative_coordinates_sigma_score_norm.shape[0])
        # and compute the norm
        z_norm = torch.linalg.norm(z, dim=-1).mean(dim=-1).view(-1, 1, 1)
        # note that sigma_score is \sigma * s(x, t), so we need to divide the norm by sigma to get the correct step size
        eps_i = (
            2
            * (
                self.corrector_r
                * z_norm
                / (relative_coordinates_sigma_score_norm.clip(min=self.small_epsilon))
            )
            ** 2
        )
        sqrt_2eps_i = torch.sqrt(2 * eps_i)

        corrected_x_i = self.relative_coordinates_update(
            composition_i.X, model_predictions_i.X, sigma_i, eps_i, sqrt_2eps_i, z=z
        )

        if self.atom_type_transition_in_corrector:
            q_matrices_i = self.noise.q_matrix[idx].to(composition_i.X)
            q_bar_matrices_i = self.noise.q_bar_matrix[idx].to(composition_i.X)
            q_bar_tm1_matrices_i = self.noise.q_bar_tm1_matrix[idx].to(composition_i.X)
            # atom types update
            corrected_a_i = self.atom_types_update(
                model_predictions_i.A,
                composition_i.A,
                q_matrices_i,
                q_bar_matrices_i,
                q_bar_tm1_matrices_i,
            )
        else:
            corrected_a_i = composition_i.A

        corrected_composition_i = AXL(
            A=corrected_a_i,
            X=corrected_x_i,
            L=unit_cell,  # TODO replace with AXL-L
        )

        if self.record and self.record_corrector:
            # TODO : Deal with L correctly
            composition_i_for_recording = AXL(
                A=composition_i.A, X=composition_i.X, L=unit_cell
            )
            # Keep the record on the CPU
            entry = dict(time_step_index=index_i)
            list_keys = [
                "composition_i",
                "corrected_composition_i",
                "model_predictions_i",
            ]
            list_axl = [
                composition_i_for_recording,
                corrected_composition_i,
                model_predictions_i,
            ]

            for key, axl in zip(list_keys, list_axl):
                record_axl = AXL(
                    A=axl.A.detach().cpu(),
                    X=axl.X.detach().cpu(),
                    L=axl.L.detach().cpu(),
                )
                entry[key] = record_axl

            self.sample_trajectory_recorder.record(key="corrector_step", entry=entry)

        return corrected_composition_i
