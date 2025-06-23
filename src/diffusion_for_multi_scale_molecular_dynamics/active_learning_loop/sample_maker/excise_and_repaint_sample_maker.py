from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import \
    BaseAtomSelector
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcision
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    BaseExciseSampleMaker, BaseExciseSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import (
    ConstrainedLangevinGenerator, SamplingConstraint)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling import \
    create_batch_of_samples


@dataclass(kw_only=True)
class ExciseAndRepaintSampleMakerArguments(BaseExciseSampleMakerArguments):
    """Arguments for a sample generator based on the excise and repaint approach."""

    algorithm: str = "excise_and_repaint"


class ExciseAndRepaintSampleMaker(BaseExciseSampleMaker):
    """Sample maker for the excise and repaint approach.

    An excisor extract atomic environments with high uncertainties and a diffusion model is used to repaint around
    them.
    """

    def __init__(
        self,
        sample_maker_arguments: ExciseAndRepaintSampleMakerArguments,
        atom_selector: BaseAtomSelector,
        environment_excisor: BaseEnvironmentExcision,
        noise_parameters: NoiseParameters,
        sampling_parameters: SamplingParameters,
        diffusion_model: ScoreNetwork,
        device: str = "cpu",
    ):
        """Init method.

        Args:
            sample_maker_arguments: arguments for the excise and repaint sample maker
            atom_selector: an atom selector.
            environment_excisor: atomic environment excisor
            noise_parameters: noise parameters used for the diffusion model
            sampling_parameters: sampling parameters used for the diffusion model
            diffusion_model: score network used for constrained generation (repainting)
            device: torch device to use for the diffusion model. Defaults to cpu.
        """
        super().__init__(sample_maker_arguments=sample_maker_arguments,
                         atom_selector=atom_selector,
                         environment_excisor=environment_excisor)

        assert sample_maker_arguments.number_of_samples_per_substructure == sampling_parameters.number_of_samples, \
            ("ExciseAndRepaint uses a generative model to generates samples. The number of samples requested in "
             "the sampling_parameters (ie, 'number_of_samples') should be identical to the number of samples per "
             "substructure requested in the sample_maker configuration (ie 'number_of_samples_per_substructure'). "
             "The configuration currently asks for inconsistent things. Review input.")

        self.sample_noise_parameters = noise_parameters
        self.sampling_parameters = sampling_parameters
        self.diffusion_model = diffusion_model
        self.device = torch.device(device)

    def create_sampling_constraints(
        self,
        constrained_structure: AXL,
    ) -> SamplingConstraint:
        """Create a Sampling Constraint dataclass from an excised structure (an AXL of numpy arrays).

        Args:
            constrained_structure: fixed atomic positions and types determined from the excision algorithm

        Returns:
            sampling_constraint: data class usable by a ConstrainedLangevinGenerator
        """
        elements = self.arguments.element_list
        sampling_constraint = SamplingConstraint(
            elements=elements,
            constrained_relative_coordinates=torch.FloatTensor(constrained_structure.X),
            constrained_atom_types=torch.LongTensor(constrained_structure.A),
        )
        # a FloatTensor is used for the coordinates because torch will convert to float64 instead of float32
        return sampling_constraint

    @staticmethod
    def torch_batch_axl_to_list_of_numpy_axl(axl_structure: AXL):
        """Convert an AXL of torch.tensor with a batch_size dimension to a list of numpy AXL.

        Args:
            axl_structure: AXL with torch.tensor. The first dimension for A, X and L is the batch_size.

        Returns:
            list of batch_size AXL with numpy array without a batch dimension.
        """
        atoms_type = axl_structure.A.cpu().numpy()  # (batch, natom)
        relative_coordinates = (
            axl_structure.X.cpu().numpy()
        )  # (batch, natom, spatial dimension)
        lattice_parameters = (
            axl_structure.L.cpu().numpy()
        )  # (batch, num_lattice_parameters)
        list_of_np_axl = []
        for a, x, lp in zip(atoms_type, relative_coordinates, lattice_parameters):
            np_axl = AXL(A=a, X=x, L=lp)
            list_of_np_axl.append(np_axl)
        return list_of_np_axl

    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        active_atom_index: int,
        num_samples: int = 1,
    ) -> Tuple[List[AXL], List[int], List[Dict[str, Any]]]:
        """Create new samples using a constrained structure using a diffusion model to repaint non-constrained atoms.

        This method assumes the lattice parameters in the constrained structure are already rescaled
        (box size is reduced).

        Args:
            substructure: excised substructure
            active_atom_index: index of the "active atom" in the input substructure.
            num_samples: number of samples to generate with the substructure

        Returns:
            new_structures: list of generated candidates structure
            list_active_atom_indices: for each created sample, the index of the "active atom", ie the
                atom at the center of the excised region.
            list_info: list of samples additional information.
        """
        sampling_constraints = self.create_sampling_constraints(substructure)
        generator = ConstrainedLangevinGenerator(
            noise_parameters=self.sample_noise_parameters,
            sampling_parameters=self.sampling_parameters,
            axl_network=self.diffusion_model,
            sampling_constraints=sampling_constraints,
        )
        with torch.no_grad():
            generated_samples = create_batch_of_samples(
                generator=generator,
                sampling_parameters=self.sampling_parameters,
                device=self.device,
            )

        new_structures = self.torch_batch_axl_to_list_of_numpy_axl(
            generated_samples["original_axl"]
        )
        # TODO: verify that this is true!
        list_active_atom_indices = num_samples * [active_atom_index]

        # additional information on generated structures can be passed here
        additional_information_on_new_structures = [{}] * len(new_structures)

        return new_structures, list_active_atom_indices, additional_information_on_new_structures

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Return identical structures."""
        return structures
