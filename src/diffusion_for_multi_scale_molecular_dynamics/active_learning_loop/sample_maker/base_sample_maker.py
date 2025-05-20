from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcision
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class BaseSampleMakerArguments:
    """Parameters controlling the sample maker method."""
    algorithm: str
    sample_box_strategy: str = "fixed"
    sample_box_size: Optional[float] = None

    def __post_init__(self):
        """Post init."""
        assert self.sample_box_strategy in ["fixed", "noop"], \
            f"Sample box making strategy {self.sample_box_strategy} is not implemented."
        if self.sample_box_strategy == "fixed":
            assert self.sample_box_size is not None
            self.new_box_lattice_parameters = None # TODO convert to lattice parameters


class BaseSampleMaker(ABC):
    """Base class for the method making new samples."""

    def __init__(self, sample_maker_arguments: BaseSampleMakerArguments):
        self.arguments = sample_maker_arguments

    @abstractmethod
    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        pass

    def make_new_lattice_parameters(
        self,
        structure: AXL
    ) -> np.array:
        match self.arguments.sample_box_strategy:
            case "noop":
                return structure.L
            case "fixed":
                return self.arguments.new_box_lattice_parameters
            case _:  # something went wrong, an invalid box making strategy is used
                return None



@dataclass(kw_only=True)
class NoOpSampleMakerArguments(BaseSampleMakerArguments):
    """Parameters for a trivial sample maker method."""
    algorithm = "NoOpSampleMaker"
    sample_box_strategy = "noop"


class NoOpSampleMaker(BaseSampleMaker):
    """Trivial sample maker that reproduces the excised environment without modifications."""

    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        return [structure]


@dataclass(kw_only=True)
class BaseExciseSampleMakerArguments(BaseSampleMakerArguments):
    """Parameters for a sample maker relying on an excision method to find the constrained atoms."""
    max_constrained_substructure : int = -1  # max number of problematic environment to consider. -1 means no limit.
    number_of_samples_per_substructure: int = 1  # number of samples to make for each sub-environment excised


class BaseExciseSampleMaker(BaseSampleMaker):
    """Base class for a sample maker relying on an excisor to extract atomic environments with high uncertainties."""
    def __init__(
        self,
        sample_maker_arguments: BaseExciseSampleMakerArguments,
        environment_excisor: BaseEnvironmentExcision
    ):
        super().__init__(sample_maker_arguments)
        self.environment_excisor = environment_excisor

    @abstractmethod
    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        num_samples : int = 1,
    ) -> List[AXL]:
        """Create new samples using a constrained structure.

        Args:
            substructure: constrained atoms described as an AXL
            num_samples: number of samples to make. Defaults to 1.

        Returns:
            list of samples created. The length of the list should match num_samples.
        """
        pass

    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        """Make new samples based on the excised substructures created using the uncertainties.

        Args:
            structure: crystal structure, including atomic species, relative coordinates and lattice parameters
            uncertainty_per_atom: uncertainty associated to each atom. The order is assumed to be the same as those in
                the structure variable.

        Returns:
            created_samples: list of generated samples as AXL structures
        """
        constrained_environments_after_excision = self.environment_excisor.excise_environments(
            structure, uncertainty_per_atom
        )
        if 0 < self.arguments.max_constrained_substructure < len(constrained_environments_after_excision):
            constrained_environments_after_excision = \
                constrained_environments_after_excision[:self.arguments.max_constrained_substructure]
        created_samples = []
        for constrained_environment in constrained_environments_after_excision:
            new_samples = self.make_samples_from_constrained_substructure(
                constrained_environment,
                self.arguments.number_of_samples_per_substructure
            )
            created_samples += new_samples
        return created_samples

