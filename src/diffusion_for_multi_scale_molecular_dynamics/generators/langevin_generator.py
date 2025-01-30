import dataclasses
from typing import Optional, Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import (
    PredictorCorrectorAXLGenerator, PredictorCorrectorSamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    TrajectoryInitializer
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot, get_probability_at_previous_time_step)
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import \
    SampleTrajectory


class LangevinGenerator(PredictorCorrectorAXLGenerator):
    """Annealed Langevin Dynamics Generator.

    This class implements the annealed Langevin Dynamics generation of position samples, following
    Song & Ermon 2019, namely:
        "Generative Modeling by Estimating Gradients of the Data Distribution"
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        axl_network: ScoreNetwork,
        trajectory_initializer: Optional[TrajectoryInitializer] = None,
    ):
        """Init method."""
        super().__init__(
            number_of_discretization_steps=noise_parameters.total_time_steps,
            number_of_corrector_steps=sampling_parameters.number_of_corrector_steps,
            spatial_dimension=sampling_parameters.spatial_dimension,
            num_atom_types=sampling_parameters.num_atom_types,
            number_of_atoms=sampling_parameters.number_of_atoms,
            trajectory_initializer=trajectory_initializer,
        )

        self.noise_parameters = noise_parameters
        sampler = NoiseScheduler(noise_parameters, num_classes=self.num_classes)
        self.noise, self.langevin_dynamics = sampler.get_all_sampling_parameters()
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.masked_atom_type_index = self.num_classes - 1
        self.axl_network = axl_network
        self.small_epsilon = sampling_parameters.small_epsilon

        self.one_atom_type_transition_per_step = (
            sampling_parameters.one_atom_type_transition_per_step
        )
        self.atom_type_greedy_sampling = sampling_parameters.atom_type_greedy_sampling
        self.atom_type_transition_in_corrector = (
            sampling_parameters.atom_type_transition_in_corrector
        )

        self.record = sampling_parameters.record_samples
        self.record_corrector = sampling_parameters.record_samples_corrector_steps
        self.record_atom_type_update = sampling_parameters.record_atom_type_update

        if self.record_corrector or self.record_atom_type_update:
            assert (
                self.record
            ), "Corrector steps or atom_type_update can only be recorded if record_samples is True."

        if self.record:
            self.sample_trajectory_recorder = SampleTrajectory()
            self.sample_trajectory_recorder.record(key="noise", entry=self.noise)
            self.sample_trajectory_recorder.record(
                key="noise_parameters", entry=dataclasses.asdict(noise_parameters)
            )
            self.sample_trajectory_recorder.record(
                key="sampling_parameters", entry=dataclasses.asdict(sampling_parameters)
            )

    def _draw_gaussian_sample(self, number_of_samples):
        return torch.randn(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        )

    def _draw_gumbel_sample(self, number_of_samples):
        return -torch.log(
            -torch.log(
                torch.rand(
                    number_of_samples, self.number_of_atoms, self.num_classes
                ).clip(min=self.small_epsilon)
            )
        )

    def _draw_binary_sample(self, number_of_samples):
        # this is used to determine if a MASK sample should be demasked or not in greedy sampling
        return torch.rand(number_of_samples, self.number_of_atoms)

    def _get_model_predictions(
        self,
        composition: AXL,
        time: float,
        sigma_noise: float,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Get the outputs of an axl-network.

        Args:
            composition : AXL composition with:
                atom types, of shape [number of samples, number_of_atoms]
                relative coordinates, of shape [number_of_samples, number_of_atoms, spatial_dimension]
                lattice vectors, of shape [number_of_samples, spatial_dimension * (spatial_dimension - 1)]  # TODO check
            time : time at which to evaluate the score
            sigma_noise: the diffusion sigma parameter corresponding to the time at which to evaluate the score
            unit_cell: unit cell definition in Angstrom of shape [number_of_samples, spatial_dimension,
                spatial_dimension]
            cartesian_forces: forces to condition the sampling from. Shape [number_of_samples, number_of_atoms,
                spatial_dimension]

        Returns:
            axl network output:
                 atom type: logits of p(a_0 | a_t).
                 relative coordinates: sigma normalized score: sigma x Score(x, t).
                 lattice: TODO.
        """
        number_of_samples = composition.X.shape[0]

        time_tensor = time * torch.ones(number_of_samples, 1).to(composition.X)
        sigma_noise_tensor = sigma_noise * torch.ones(number_of_samples, 1).to(
            composition.X
        )
        augmented_batch = {
            NOISY_AXL_COMPOSITION: composition,
            TIME: time_tensor,
            NOISE: sigma_noise_tensor,
            UNIT_CELL: unit_cell,  # TODO replace with AXL-L
            CARTESIAN_FORCES: cartesian_forces,
        }

        # TODO do not hard-code conditional to False - need to be able to condition sampling
        model_predictions = self.axl_network(augmented_batch, conditional=False)
        return model_predictions

    def _relative_coordinates_update(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        r"""Generic update for the relative coordinates.

        This is useful for both the predictor and the corrector step. The score weight and gaussian weight noise differs
        in these two settings.

        Args:
            relative_coordinates: starting coordinates. Dimension: [number_of_samples, number_of_atoms,
                spatial_dimension]

            sigma_normalized_scores: output of the model - an estimate of the normalized
                score :math:`\sigma \nabla log p(x)`.
                Dimension: [number_of_samples, number_of_atoms, spatial_dimension]
            sigma_i: noise parameter for variance exploding noise scheduler. Dimension: [number_of_samples]
            score_weight: prefactor in front of the normalized score update. Should be g2_i in the predictor step and
                eps_i in the corrector step. Dimension: [number_of_samples]
            gaussian_noise_weight: prefactor in front of the random noise update. Should be g_i in the predictor step
                and sqrt_2eps_i in the corrector step. Dimension: [number_of_samples]
            z: gaussian noise used to update the coordinates. A sample drawn from the normal distribution.
                Dimension: [number_of_samples, number_of_atoms, spatial_dimension].

        Returns:
            updated_coordinates: relative coordinates after the update. Dimension: [number_of_samples, number_of_atoms,
                spatial_dimension].
        """
        number_of_samples = relative_coordinates.shape[0]
        if z is None:
            z = self._draw_gaussian_sample(number_of_samples).to(relative_coordinates)
        updated_coordinates = (
            relative_coordinates
            + score_weight * sigma_normalized_scores / sigma_i
            + gaussian_noise_weight * z
        )
        # map back to the range [0, 1)
        updated_coordinates = map_relative_coordinates_to_unit_cell(updated_coordinates)
        return updated_coordinates

    def _relative_coordinates_update_predictor_step(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Relative coordinates update for the predictor step.

        This returns the generic _relative_coordinates_update.
        """
        return self._relative_coordinates_update(
            relative_coordinates,
            sigma_normalized_scores,
            sigma_i,
            score_weight,
            gaussian_noise_weight,
            z,
        )

    def _relative_coordinates_update_corrector_step(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Relative coordinates update for the corrector step.

        This returns the generic _relative_coordinates_update.
        """
        return self._relative_coordinates_update(
            relative_coordinates,
            sigma_normalized_scores,
            sigma_i,
            score_weight,
            gaussian_noise_weight,
            z,
        )

    def _atom_types_update(
        self,
        predicted_logits: torch.Tensor,
        atom_types_i: torch.LongTensor,
        q_matrices_i: torch.Tensor,
        q_bar_matrices_i: torch.Tensor,
        q_bar_tm1_matrices_i: torch.Tensor,
        atom_type_greedy_sampling: bool,
        one_atom_type_transition_per_step: bool,
    ) -> torch.LongTensor:
        """Generic update of the atom types.

        This should be used in the predictor step only.

        Args:
            predicted_logits: output of the model - an estimate of p(a_0 | a_t). Dimension:
                [number_of_samples, number_of_atoms, num_classes].
            atom_types_i: indices of the atom types at timestep i. Dimension:
                [number_of_samples, number_of_atoms]
            q_matrices_i: one-step transition matrix. Dimension: [number_of_samples, number_of_atoms, num_classes,
                num_classes].
            q_bar_matrices_i: cumulative transition matrix at time step i. Dimension: [number_of_samples,
                number_of_atoms, num_classes, num_classes].
            q_bar_tm1_matrices_i:  cumulative transition matrix at time step 'i - 1'. Dimension: [number_of_samples,
                number_of_atoms, num_classes, num_classes].
            atom_type_greedy_sampling: boolean flag that sets whether the atom types should be selected greedily.
            one_atom_type_transition_per_step: boolean flag that sets whether a single atom type transition can
                occur per time step.

        Returns:
            atom_types_im1: updated atom type indices. Dimension: [number_of_samples, number_of_atoms]
        """
        number_of_samples = predicted_logits.shape[0]
        gumbel_random_variable = self._draw_gumbel_sample(number_of_samples).to(
            predicted_logits.device
        )
        one_hot_atom_types_i = class_index_to_onehot(
            atom_types_i, num_classes=self.num_classes
        )
        one_step_transition_probs = get_probability_at_previous_time_step(
            probability_at_zeroth_timestep=predicted_logits,
            one_hot_probability_at_current_timestep=one_hot_atom_types_i,
            q_matrices=q_matrices_i,
            q_bar_matrices=q_bar_matrices_i,
            q_bar_tm1_matrices=q_bar_tm1_matrices_i,
            small_epsilon=self.small_epsilon,
            probability_at_zeroth_timestep_are_logits=True,
        )  # p(a_{t-1} | a_t) as a [num_samples, num_atoms, num_classes] tensor

        if atom_type_greedy_sampling:
            # if we use greedy sampling, we will update the transition probabilities for the MASK token.
            # For a_i = MASK, we define "greedy sampling" as first determining if a_{i-1} should also be MASK based on
            # p(a_{i-1} = MASK | a_i = MASK). If a_{i-1} should be unmasked, its atom type is selected as the one with
            # the highest probability (i.e., no stochastic sampling). Stochasticity is removed by setting the relevant
            # row of gumbel_random_variable to zero.
            one_step_transition_probs, gumbel_random_variable = (
                self._adjust_atom_types_probabilities_for_greedy_sampling(
                    one_step_transition_probs, atom_types_i, gumbel_random_variable
                )
            )

        # Use the Gumbel-softmax trick to sample atomic types.
        # We also keep the associated values in memory, so we can compare which transitions are the most likely.
        # Dimensions: [num_samples, num_atoms].
        max_gumbel_values, sampled_atom_types = torch.max(
            torch.log(one_step_transition_probs + self.small_epsilon)
            + gumbel_random_variable,
            dim=-1,
        )

        if one_atom_type_transition_per_step:
            # force a single transition for each sample
            atom_types_im1 = self._get_updated_atom_types_for_one_transition_per_step(
                atom_types_i, max_gumbel_values, sampled_atom_types
            )
        else:
            atom_types_im1 = sampled_atom_types

        if self.record_atom_type_update:
            # Keep the record on the CPU
            entry = dict(
                predicted_logits=predicted_logits.detach().cpu(),
                one_step_transition_probabilities=one_step_transition_probs.detach().cpu(),
                gumbel_sample=gumbel_random_variable.cpu(),
                a_i=atom_types_i.cpu(),
                a_im1=atom_types_im1.cpu(),
            )

            self.sample_trajectory_recorder.record(key="atom_type_update", entry=entry)

        return atom_types_im1

    def _get_updated_atom_types_for_one_transition_per_step(
        self,
        current_atom_types: torch.Tensor,
        max_gumbel_values: torch.Tensor,
        sampled_atom_types: torch.Tensor,
    ):
        """Get updated atom types for one transition per step.

        Assuming the Gumbel softmax trick was used to create a new sample of atom types, this method
        restrict the transitions from the current atom types to only the most likely one per sample.

        Args:
            current_atom_types: current indices of the atom types. Dimension: [number_of_samples, number_of_atoms]
            max_gumbel_values: maximum Gumbel softmax values. Dimension: [number_of_samples, number_of_atoms]
            sampled_atom_types: indices of the atom types resulting from the gumbel softmax sampling.
                Dimension: [number_of_samples, number_of_atoms]

        Returns:
            updated_atom_types: atom types resulting from only making one transition per sample on current_atom_types.
                Dimension: [number_of_samples, number_of_atoms]
        """
        number_of_samples = current_atom_types.shape[0]
        sample_indices = torch.arange(number_of_samples)

        # Boolean mask of dimensions [number_of_samples, number_of_atoms]
        atoms_have_changed_types = sampled_atom_types != current_atom_types

        # Identify the most likely transition amongst the proposed changes.
        max_gumbel_values_restricted_to_proposed_changes = torch.where(
            atoms_have_changed_types, max_gumbel_values, -torch.inf
        )
        most_likely_transition_atom_indices = torch.argmax(
            max_gumbel_values_restricted_to_proposed_changes, dim=-1
        )

        # Restrict transitions to only the most likely ones.
        updated_atom_types = current_atom_types.clone()
        updated_atom_types[sample_indices, most_likely_transition_atom_indices] = (
            sampled_atom_types[sample_indices, most_likely_transition_atom_indices]
        )

        return updated_atom_types

    def _adjust_atom_types_probabilities_for_greedy_sampling(
        self,
        one_step_transition_probs: torch.Tensor,
        atom_types_i: torch.LongTensor,
        gumbel_random_variable: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the transition probabilities and the gumbel random variables to allow greedy sampling.

        At time step i, for every atom in a sample, we sample a random number. If it is larger than the probability of
        that atom being in the MASK class, then we will sample greedily a new atom type (i.e. the most likely). To do
        that, we simply replace the probability of the MASK class to zero and the gumbel noise u to zero. For non-MASK
        atoms, we do nothing. For samples with only MASK atoms, we also do nothing.

        Args:
            one_step_transition_probs: class distributions at time t-1 given distribution at time t. p(a_{t-1} | a_t)
            atom_types_i: indices of atom types at time i. Dimension: [number_of_samples, number_of_atoms]
            gumbel_random_variable: gumbel noise used for sampling.
                Dimension: [number_of_samples, number_of_atoms, num_classes]

        Returns:
            one_step_transition_probs: probabilities are updated so a MASK to non-MASK transition can happen
            u: set to a constant for samples with at least 1 non-MASK atom
        """
        # check which samples have at least 1 non-MASK atom
        all_masked = torch.all(
            atom_types_i == self.masked_atom_type_index, dim=-1
        )  # dim: number_of_samples,

        # we can only do greedy sampling for atoms that are masked
        atom_is_masked = atom_types_i == self.masked_atom_type_index

        # we will first erase the probability of staying MASK for some atoms randomly by drawing from a binary
        # distribution given by one_step_transition_probs[:, :, -1] i.e. the probabilities related to the MASK class.
        # sample to override the MASK probability as the most likely
        binary_sample = self._draw_binary_sample(atom_types_i.shape[0]).to(
            device=atom_types_i.device
        )
        unmask_this_atom = binary_sample > one_step_transition_probs[:, :, -1]
        # if we override the MASK probability & there's already a non-MASK sample & that atom is masked,
        # use a greedy sampling for that atom
        do_greedy_sampling = torch.logical_and(
            ~all_masked.view(-1, 1),
            unmask_this_atom,
        )
        do_greedy_sampling = torch.logical_and(do_greedy_sampling, atom_is_masked)
        # replace the probability of getting a mask for those by 0 - so that state cannot be sampled
        one_step_transition_probs[:, :, -1] = torch.where(
            do_greedy_sampling, 0, one_step_transition_probs[:, :, -1]
        )

        # replace u with a constant for samples with a non-MASK token present - this ensures a greedy sampling
        # In the current choice of \beta_t = 1 / (T-t+1), a greedy sampling will always select the MASK type if that
        # probability is not set to zero - except at the last generation step. This might not hold if the \beta schedule
        # is modified.
        gumbel_random_variable = torch.where(
            all_masked.view(-1, 1, 1), gumbel_random_variable, 0.0
        )
        return one_step_transition_probs, gumbel_random_variable

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

        number_of_samples = composition_i.X.shape[0]
        number_of_atoms = composition_i.X.shape[1]

        idx = index_i - 1  # python starts indices at zero
        t_i = self.noise.time[idx].to(composition_i.X)
        g_i = self.noise.g[idx].to(composition_i.X)
        g2_i = self.noise.g_squared[idx].to(composition_i.X)
        sigma_i = self.noise.sigma[idx].to(composition_i.X)

        # Broadcast the q matrices to the expected dimensions.
        q_matrices_i = einops.repeat(
            self.noise.q_matrix[idx].to(composition_i.X),
            "n1 n2 -> nsamples natoms n1 n2",
            nsamples=number_of_samples,
            natoms=number_of_atoms,
        )

        q_bar_matrices_i = einops.repeat(
            self.noise.q_bar_matrix[idx].to(composition_i.X),
            "n1 n2 -> nsamples natoms n1 n2",
            nsamples=number_of_samples,
            natoms=number_of_atoms,
        )

        q_bar_tm1_matrices_i = einops.repeat(
            self.noise.q_bar_tm1_matrix[idx].to(composition_i.X),
            "n1 n2 -> nsamples natoms n1 n2",
            nsamples=number_of_samples,
            natoms=number_of_atoms,
        )

        model_predictions_i = self._get_model_predictions(
            composition_i, t_i, sigma_i, unit_cell, cartesian_forces
        )

        # Even if the global flag 'one_atom_type_transition_per_step' is set to True, a single atomic transition
        # cannot be used at the last time step because it is necessary for all atoms to be unmasked at the end
        # of the trajectory. Here, we use 'first' and 'last' with respect to a denoising trajectory, where
        # the "first" time step is at index_i = T and the "last" time step is index_i = 1.
        this_is_last_time_step = idx == 0
        one_atom_type_transition_per_step = (
            self.one_atom_type_transition_per_step and not this_is_last_time_step
        )

        a_im1 = self._atom_types_update(
            model_predictions_i.A,
            composition_i.A,
            q_matrices_i,
            q_bar_matrices_i,
            q_bar_tm1_matrices_i,
            atom_type_greedy_sampling=self.atom_type_greedy_sampling,
            one_atom_type_transition_per_step=one_atom_type_transition_per_step,
        )

        if this_is_last_time_step:
            assert (
                a_im1 != self.masked_atom_type_index
            ).all(), \
                "There remains MASKED atoms at the last time step: review code, there must be a bug or invalid input."

        # draw a gaussian noise sample and update the positions accordingly
        z = self._draw_gaussian_sample(number_of_samples).to(composition_i.X)

        x_im1 = self._relative_coordinates_update_predictor_step(
            composition_i.X, model_predictions_i.X, sigma_i, g2_i, g_i, z
        )

        composition_im1 = AXL(
            A=a_im1, X=x_im1, L=unit_cell
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

    def _get_corrector_step_size(
        self,
        index_i: int,
        sigma_i: torch.Tensor,
        model_predictions_i: AXL,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the size of the corrector step for the relative coordinates update."""
        # Get the epsilon from the tabulated Langevin dynamics array indexed with [0,..., N-1].
        eps_i = self.langevin_dynamics.epsilon[index_i].to(model_predictions_i.X)
        return eps_i

    def corrector_step(
        self,
        composition_i: AXL,
        index_i: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Corrector Step.

        Note this does not affect the atom types unless specified with the atom_type_transition_in_corrector
        argument. Always affect the reduced coordinates and lattice vectors.

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

        number_of_samples = composition_i.X.shape[0]

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

        # draw a gaussian noise sample and update the positions accordingly
        z = self._draw_gaussian_sample(number_of_samples).to(composition_i.X)

        # get the step size eps_i
        eps_i = self._get_corrector_step_size(index_i, sigma_i, model_predictions_i, z)
        # the size for the noise part is sqrt(2 * eps_i)
        sqrt_2eps_i = torch.sqrt(2 * eps_i)

        corrected_x_i = self._relative_coordinates_update_corrector_step(
            composition_i.X, model_predictions_i.X, sigma_i, eps_i, sqrt_2eps_i, z
        )

        if self.atom_type_transition_in_corrector:
            q_matrices_i = self.noise.q_matrix[idx].to(composition_i.X)
            q_bar_matrices_i = self.noise.q_bar_matrix[idx].to(composition_i.X)
            q_bar_tm1_matrices_i = self.noise.q_bar_tm1_matrix[idx].to(composition_i.X)
            # atom types update
            corrected_a_i = self._atom_types_update(
                model_predictions_i.A,
                composition_i.A,
                q_matrices_i,
                q_bar_matrices_i,
                q_bar_tm1_matrices_i,
                atom_type_greedy_sampling=self.atom_type_greedy_sampling,
                one_atom_type_transition_per_step=self.one_atom_type_transition_per_step,
            )
        else:
            corrected_a_i = composition_i.A

        corrected_composition_i = AXL(
            A=corrected_a_i,
            X=corrected_x_i,
            L=unit_cell,  # TODO replace with AXL-L
        )

        if self.record_corrector:
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
