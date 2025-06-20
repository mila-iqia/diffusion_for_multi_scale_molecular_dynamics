from typing import Any, AnyStr, Dict, Optional

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.atom_selector_factory import \
    create_atom_selector
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import \
    BaseAtomSelectorParameters
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcisionArguments
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.excisor_factory import \
    create_excisor
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    BaseSampleMaker, BaseSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_noop_sample_maker import (  # noqa
    ExciseAndNoOpSampleMaker, ExciseAndNoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_random_sample_maker import (  # noqa
    ExciseAndRandomSampleMaker, ExciseAndRandomSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_repaint_sample_maker import (  # noqa
    ExciseAndRepaintSampleMaker, ExciseAndRepaintSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters

SAMPLE_MAKER_PARAMETERS_BY_NAME = dict(
    noop=NoOpSampleMakerArguments,
    excise_and_noop=ExciseAndNoOpSampleMakerArguments,
    excise_and_repaint=ExciseAndRepaintSampleMakerArguments,
    excise_and_random=ExciseAndRandomSampleMakerArguments,
)


def create_sample_maker_parameters(
    sample_maker_dictionary: Dict[AnyStr, Any],
) -> BaseSampleMakerArguments:
    """Create atomic excision method parameters.

    Args:
        sample_maker_dictionary : parsed configuration for the sample maker.

    Returns:
        sample_maker_parameters: a configuration object for a Sample Maker object.
    """
    algorithm = sample_maker_dictionary["algorithm"]

    assert algorithm in SAMPLE_MAKER_PARAMETERS_BY_NAME.keys(), (
        f"Sample maker method {algorithm} is not implemented. Possible choices are "
        f"{SAMPLE_MAKER_PARAMETERS_BY_NAME.keys()}"
    )

    sample_maker_parameters = SAMPLE_MAKER_PARAMETERS_BY_NAME[algorithm](
        **sample_maker_dictionary
    )
    return sample_maker_parameters


def create_sample_maker(
    sample_maker_parameters: BaseSampleMakerArguments,
    atom_selector_parameters: BaseAtomSelectorParameters,
    excisor_parameters: Optional[BaseEnvironmentExcisionArguments] = None,
    noise_parameters: Optional[NoiseParameters] = None,
    sampling_parameters: Optional[SamplingParameters] = None,
    diffusion_model: Optional[ScoreNetwork] = None,
    device: Optional[str] = "cpu",
) -> BaseSampleMaker:
    """Create a sample maker.

    This is a factory method responsible for instantiating the sample maker.
    """
    algorithm = sample_maker_parameters.algorithm
    assert algorithm in SAMPLE_MAKER_PARAMETERS_BY_NAME.keys(), (
        "Sample maker method {algorithm} is not implemented. Possible choices are "
        f"{SAMPLE_MAKER_PARAMETERS_BY_NAME.keys()}"
    )

    atom_selector = create_atom_selector(atom_selector_parameters)

    if excisor_parameters is not None:
        excisor = create_excisor(excisor_parameters)
    else:
        excisor = None

    match sample_maker_parameters.algorithm:
        case "noop":
            sample_maker = NoOpSampleMaker(sample_maker_parameters, atom_selector=atom_selector)

        case "excise_and_repaint":
            # TODO
            sample_maker = ExciseAndRepaintSampleMaker(
                sample_maker_arguments=sample_maker_parameters,
                environment_excisor=excisor,
                noise_parameters=noise_parameters,
                sampling_parameters=sampling_parameters,
                diffusion_model=diffusion_model,
                device=device,
            )
        case "excise_and_random":
            # TODO
            sample_maker = ExciseAndRandomSampleMaker(
                sample_maker_arguments=sample_maker_parameters,
                environment_excisor=excisor,
            )
        case "excise_and_noop":
            sample_maker = ExciseAndNoOpSampleMaker(
                sample_maker_arguments=sample_maker_parameters,
                atom_selector=atom_selector,
                environment_excisor=excisor,
            )
        case _:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")

    return sample_maker
