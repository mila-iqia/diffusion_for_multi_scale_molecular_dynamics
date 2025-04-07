from typing import Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL, LATTICE_PARAMETERS, NOISE, NOISY_ATOM_TYPES,
    NOISY_LATTICE_PARAMETERS, NOISY_RELATIVE_COORDINATES, Q_BAR_MATRICES,
    Q_BAR_TM1_MATRICES, Q_MATRICES, RELATIVE_COORDINATES, TIME, TIME_INDICES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import (
    Noise, NoiseScheduler)
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import (
    LatticeDataParameters, LatticeNoiser)
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.transport.transporter import \
    Transporter
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.utils.noise_utils import \
    scale_sigma_by_number_of_atoms
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
        use_fixed_lattice_parameters: bool = False,
        use_optimal_transport: bool = True,
    ):
        """Noising transform.

        This class creates a method that takes in a batch of dataset data and
        augments it with noised data.

        Args:
            noise_parameters: noise parameters.
            num_atom_types:  number of distinct atom types.
            spatial_dimension: dimension of space.
            use_fixed_lattice_parameters: if True, do not noise the lattice parameters, consider them as constant.
                Defaults to False.
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
            L=LatticeNoiser(
                LatticeDataParameters(
                    spatial_dimension=spatial_dimension,
                    use_fixed_lattice_parameters=use_fixed_lattice_parameters,
                )
            ),
        )

        if self.use_optimal_transport:
            # TODO: review this as we improve the transporter
            self.point_group_operations = torch.eye(spatial_dimension).unsqueeze(0)
            self.transporter = Transporter(
                point_group_operations=self.point_group_operations,
            )

    def transform(self, batch: Dict) -> Dict:
        """Transform.

        This method adds the required data for score matching.

        Args:
            batch: dataset data.

        Returns:
            augmented_batch: batch augmented with noised data for score matching.
        """
        self._check_batch(batch)
        batch_size = batch[RELATIVE_COORDINATES].shape[0]
        noise_sample = self.noise_scheduler.get_random_noise_sample(batch_size)
        return self._transform_from_noise_sample(batch, noise_sample)

    def transform_given_time_index(self, batch: Dict, index_i: int) -> Dict:
        """Transform given time index.

        This method restricts all the noise parameters to correspond to the input time index.

        Args:
            batch: dataset data.
            index_i: time index for all the noise elements. CAREFUL! This index should correspond to the
                one-based indexing scheme for time, where t_1= delta,..., t_N=t_{max}.

        Returns:
            augmented_batch: batch augmented with noised data
        """
        assert index_i > 0, "The time index should never be smaller than 1."

        idx = index_i - 1  # python starts indices at zero
        self._check_batch(batch)
        batch_size = batch[RELATIVE_COORDINATES].shape[0]
        device = batch[RELATIVE_COORDINATES].device
        indices = torch.ones(batch_size, dtype=torch.long, device=device) * idx
        self.noise_scheduler.to(device)
        noise_sample = self.noise_scheduler.get_noise_from_indices(indices)
        return self._transform_from_noise_sample(batch, noise_sample)

    def _transform_from_noise_sample(self, batch: Dict, noise_sample: Noise) -> Dict:
        """Transform from a noise sample.

        This method noise all composition elements based on the noise sample.

        Args:
            batch: dataset data.

        Returns:
            augmented_batch: batch augmented with noised data for score matching.
        """
        augmentation_data = dict()

        x0 = batch[RELATIVE_COORDINATES]
        a0 = batch[ATOM_TYPES]
        l0 = batch[LATTICE_PARAMETERS]
        shape = x0.shape
        atom_shape = a0.shape

        # the datasets library does mysterious things if we use an AXL. Let's use raw tensors.
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
            xt = self.transporter.get_optimal_transport(x0, xt)

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

        # scale sigma by the number of atoms for lattice parameters noising
        num_atoms = (
            torch.ones_like(l0) * atom_shape[1]
        )  # TODO should depend on data - not a constant
        # num_atoms should be broadcasted to match sigmas_for_lattice
        sigmas_n = scale_sigma_by_number_of_atoms(
            noise_sample.sigma.reshape(-1, 1),
            num_atoms,
            spatial_dimension=x0.shape[-1],
        )
        lt = self.noisers.L.get_noisy_lattice_parameters(
            l0,
            sigmas_n,
        )
        augmentation_data[NOISY_ATOM_TYPES] = at
        augmentation_data[NOISY_RELATIVE_COORDINATES] = xt
        augmentation_data[NOISY_LATTICE_PARAMETERS] = lt

        batch.update(augmentation_data)
        return batch

    def _check_batch(self, batch):
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."
        assert (
            ATOM_TYPES in batch
        ), f"The field '{ATOM_TYPES}' is missing from the input."
        assert (
            LATTICE_PARAMETERS in batch
        ), f"The field '{LATTICE_PARAMETERS}' is missing from the input."
        x0 = batch[RELATIVE_COORDINATES]
        shape = x0.shape
        assert len(shape) == 3, (
            f"the shape of the RELATIVE_COORDINATES array should be [batch_size, number_of_atoms, spatial_dimensions]. "
            f"Got shape = {shape}."
        )

        a0 = batch[ATOM_TYPES]
        atom_shape = a0.shape
        assert len(atom_shape) == 2, (
            f"the shape of the ATOM_TYPES array should be [batch_size, number_of_atoms]. "
            f"Got shape = {atom_shape}"
        )

        l0 = batch[LATTICE_PARAMETERS]
        lattice_parameters_shape = l0.shape
        assert len(lattice_parameters_shape) == 2, (
            f"the shape of the LATTICE parameters array should be [batch_size,"
            f"spatial_dimension * (spatial_dimension + 1) / 2]."
            f"Got shape = {lattice_parameters_shape}"
        )
