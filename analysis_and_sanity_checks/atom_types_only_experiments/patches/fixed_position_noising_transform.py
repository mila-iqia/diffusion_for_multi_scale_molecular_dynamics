from identity_noiser import IdentityRelativeCoordinatesNoiser

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noisers.lattice_noiser import (
    LatticeDataParameters, LatticeNoiser)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


class FixedPositionNoisingTransform(NoisingTransform):
    """Fixed Position Noising Transform."""

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        num_atom_types: int,
        spatial_dimension: int,
    ):
        """Init method."""
        super().__init__(
            noise_parameters=noise_parameters,
            num_atom_types=num_atom_types,
            spatial_dimension=spatial_dimension,
            use_fixed_lattice_parameters=True,
            use_optimal_transport=False,
        )

        # Overload the noisers with fixed atomic positions.
        self.noisers = AXL(
            A=AtomTypesNoiser(),
            X=IdentityRelativeCoordinatesNoiser(),
            L=LatticeNoiser(
                LatticeDataParameters(
                    spatial_dimension=spatial_dimension,
                    use_fixed_lattice_parameters=True,
                )
            ),
        )
