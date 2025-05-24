from dataclasses import dataclass
from typing import List, Any

import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcision
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import \
    BaseExciseSampleMakerArguments, BaseExciseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import SamplingConstraint, \
    ConstrainedLangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling import create_batch_of_samples


@dataclass(kw_only=True)
class ExciseAndRepaintSampleMakerArguments(BaseExciseSampleMakerArguments):
    algorithm: str = "excise_and_repaint"


class ExciseAndRepaintSampleMaker(BaseExciseSampleMaker):
    def __init__(
        self,
        sample_maker_arguments: ExciseAndRepaintSampleMakerArguments,
        environment_excisor: BaseEnvironmentExcision,
        noise_parameters: NoiseParameters,
        sampling_parameters: SamplingParameters,
        diffusion_model: ScoreNetwork
    ):
        super().__init__(sample_maker_arguments, environment_excisor)
        self.sample_noise_parameters = noise_parameters
        self.sampling_parameters = sampling_parameters
        self.diffusion_model =diffusion_model

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
            constrained_relative_coordinates=torch.tensor(constrained_structure.X),
            constrained_atom_types=torch.LongTensor(constrained_structure.A)
        )
        return sampling_constraint

    @staticmethod
    def torch_batch_axl_to_list_of_numpy_axl(axl_structure: AXL):
        atoms_type = axl_structure.A.cpu().numpy()  # (batch, natom)
        reduced_coordinates = axl_structure.X.cpu().numpy()  # (batch, natom, spatial dimension)
        lattice_parameters = axl_structure.L.cpu().numpy()  # (batch, num_lattice_parameters)
        list_of_np_axl = []
        for a, x, lp in zip(atoms_type, reduced_coordinates, lattice_parameters):
            np_axl = AXL(A=a, X=x, L=lp)
            list_of_np_axl.append(np_axl)
        return list_of_np_axl

    def make_samples_from_constrained_substructure(
        self,
        constrained_structure: AXL,
        num_samples : int = 1,
    ) -> List[AXL]:
        """Create new samples using a constrained structure using a diffusion model to repaint non-constrained atoms.

        This method assumes the constrained structure the lattice parameters are already rescaled (box size is reduced).

        Args:
            constrained_structure: excised substructure
            num_samples: number of samples to generate with the substructure

        Returns:
            new_structures: list of generated candidates structure
        """
        sampling_constraints = self.create_sampling_constraints(constrained_structure)
        generator = ConstrainedLangevinGenerator(
            noise_parameters=self.sample_noise_parameters,
            sampling_parameters=self.sampling_parameters,
            axl_network=self.diffusion_model,
            sampling_constraints=sampling_constraints
        )
        with torch.no_grad():
            generated_samples = create_batch_of_samples(
                generator=generator,
                sampling_parameters=self.sampling_parameters,
                device=self.device
            )

        new_structures = self.torch_batch_axl_to_list_of_numpy_axl(generated_samples["original_axl"])
        return new_structures

    def filter_made_samples(
        self,
        structures: List[AXL]
    ) -> List[AXL]:
        return structures
