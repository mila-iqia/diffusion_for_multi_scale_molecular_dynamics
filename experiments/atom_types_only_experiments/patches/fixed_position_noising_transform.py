import sys

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import (
    LatticeDataParameters, LatticeNoiser)

sys.path.append(
    str(TOP_DIR / "experiments/atom_types_only_experiments/patches")
)  # noqa

from identity_noiser import IdentityRelativeCoordinatesNoiser  # noqa


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
