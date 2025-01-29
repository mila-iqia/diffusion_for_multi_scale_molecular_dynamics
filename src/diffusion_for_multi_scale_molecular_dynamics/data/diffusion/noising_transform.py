from typing import Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL, LATTICE_PARAMETERS, NOISE, NOISY_ATOM_TYPES,
    NOISY_LATTICE_PARAMETERS, NOISY_RELATIVE_COORDINATES, Q_BAR_MATRICES,
    Q_BAR_TM1_MATRICES, Q_MATRICES, RELATIVE_COORDINATES, TIME, TIME_INDICES,
    UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import \
    LatticeNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.transport.transporter import \
    Transporter
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import (
    broadcast_batch_matrix_tensor_to_all_dimensions,
    broadcast_batch_tensor_to_all_dimensions)


class NoisingTransform:
    """Noising Transform."""
    def __init__(
        self,
        noise_parameters: NoiseParameters,
        num_atom_types: int,
        spatial_dimension: int,
        use_optimal_transport: bool = True,
    ):
        """Noising transform.

        This class creates a method that takes in a batch of dataset data and
        augment it with noised data.

        Args:
            noise_parameters: noise parameters.
            num_atom_types:  number of distinct atom types.
            spatial_dimension: dimension of space.
            use_optimal_transport: should optimal transport be used for the relative coordinates.
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.use_optimal_transport = use_optimal_transport

        self.noise_scheduler = NoiseScheduler(
            noise_parameters,
            num_classes=self.num_atom_types + 1,  # add 1 for the MASK class
        )

        self.noisers = AXL(
            A=AtomTypesNoiser(),
            X=RelativeCoordinatesNoiser(),
            L=LatticeNoiser(),
        )

        if self.use_optimal_transport:
            # TODO: review this as we improve the transporter
            self.point_group_operations = torch.diag(
                torch.ones(spatial_dimension)
            ).unsqueeze(0)
            self.transporter = Transporter(
                point_group_operations=self.point_group_operations,
                maximum_number_of_steps=10,
            )

    def transform(self, batch: Dict) -> Dict:
        """Transform.

        This method adds the required data for score matching.

        Args:
            batch: dataset data.

        Returns:
            augmented_batch: batch augmented with noised data for score matching.
        """
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."

        assert (
            ATOM_TYPES in batch
        ), f"The field '{ATOM_TYPES}' is missing from the input."

        x0 = batch[RELATIVE_COORDINATES]
        shape = x0.shape
        assert len(shape) == 3, (
            f"the shape of the RELATIVE_COORDINATES array should be [batch_size, number_of_atoms, spatial_dimensions]. "
            f"Got shape = {shape}."
        )
        batch_size = shape[0]

        augmentation_data = dict()

        a0 = batch[ATOM_TYPES]

        atom_shape = a0.shape
        assert len(atom_shape) == 2, (
            f"the shape of the ATOM_TYPES array should be [batch_size, number_of_atoms]. "
            f"Got shape = {atom_shape}"
        )
        # TODO: should be batch[UNIT_CELL]
        l0 = batch["box"]

        # the datasets library does mysterious things if we use an AXL. Let's use raw tensors.
        augmentation_data[LATTICE_PARAMETERS] = l0

        # from (batch, spatial_dim) to (batch, spatial_dim, spatial_dim)
        unit_cell = torch.diag_embed(l0)

        # TODO remove and take from AXL instead
        augmentation_data[UNIT_CELL] = unit_cell

        noise_sample = self.noise_scheduler.get_random_noise_sample(batch_size)
        augmentation_data[TIME] = noise_sample.time.reshape(-1, 1)
        augmentation_data[TIME_INDICES] = noise_sample.indices
        augmentation_data[NOISE] = noise_sample.sigma.reshape(-1, 1)

        # noise_sample.sigma has dimension [batch_size]. Broadcast these values to be of shape
        # [batch_size, number_of_atoms, spatial_dimension] , which can be interpreted as
        # [batch_size, (configuration)]. All the sigma values must be the same for a given configuration.
        sigmas = broadcast_batch_tensor_to_all_dimensions(
            batch_values=noise_sample.sigma, final_shape=shape
        )

        # we can now get noisy coordinates
        xt = self.noisers.X.get_noisy_relative_coordinates_sample(x0, sigmas)

        if self.use_optimal_transport:
            # Transport xt to be as close to x0 as possible
            nearest_xt = []
            for batch_idx in range(batch_size):
                transported_xt, _, _ = self.transporter.get_optimal_transport(
                    x0[batch_idx], xt[batch_idx]
                )
                nearest_xt.append(transported_xt)

            xt = torch.stack(nearest_xt, dim=0)

        # to get noisy atom types, we need to broadcast the transition matrices q, q_bar and q_bar_tm1 from size
        # [batch_size, num_atom_types, num_atom_types] to [batch_size, number_of_atoms, num_atom_types, num_atom_types].
        # All the matrices must be the same for all atoms in a given configuration.
        q_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            batch_values=noise_sample.q_matrix, final_shape=atom_shape
        )
        q_bar_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            batch_values=noise_sample.q_bar_matrix, final_shape=atom_shape
        )

        q_bar_tm1_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            batch_values=noise_sample.q_bar_tm1_matrix, final_shape=atom_shape
        )

        augmentation_data[Q_MATRICES] = q_matrices
        augmentation_data[Q_BAR_MATRICES] = q_bar_matrices
        augmentation_data[Q_BAR_TM1_MATRICES] = q_bar_tm1_matrices

        a0_onehot = class_index_to_onehot(a0, self.num_atom_types + 1)
        at = self.noisers.A.get_noisy_atom_types_sample(a0_onehot, q_bar_matrices)

        # TODO do the same for the lattice vectors
        lt = self.noisers.L.get_noisy_lattice_vectors(l0)
        augmentation_data[NOISY_ATOM_TYPES] = at
        augmentation_data[NOISY_RELATIVE_COORDINATES] = xt
        augmentation_data[NOISY_LATTICE_PARAMETERS] = lt

        batch.update(augmentation_data)
        return batch
