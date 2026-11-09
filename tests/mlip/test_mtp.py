import shutil
from pathlib import Path

import pytest

import diffusion_for_multi_scale_molecular_dynamics.mlip.mtp as mtp_package
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
    MtpConfiguration

# The level-6 template shipped with the package (a readable text header followed by binary data).
MTP_TEMPLATE_06 = Path(mtp_package.__file__).parent.parent / "MTP_templates" / "06.almtp"


class TestMtpConfiguration:

    def test_read_from_file(self):
        """Reading a level-6 template recovers the level-determined parameters from its text header."""
        configuration = MtpConfiguration(elements=["Si"], level=6, max_dist=4.0)
        configuration.read_from_file(MTP_TEMPLATE_06)

        assert configuration.species_count == 1
        assert configuration.radial_basis_size == 8
        assert configuration.radial_funcs_count == 2
        assert configuration.alpha_scalar_moments == 5
        assert configuration.number_of_adjustable_parameters == 14

    @pytest.mark.parametrize("level, elements, expected_descriptors", [
        # Basis sizes are level-determined (verified against the shipped templates); species_count = #elements.
        (6, ["Si"], dict(species_count=1, radial_basis_size=8, radial_funcs_count=2, alpha_scalar_moments=5)),
        (8, ["Si"], dict(species_count=1, radial_basis_size=8, radial_funcs_count=2, alpha_scalar_moments=9)),
        (16, ["Si"], dict(species_count=1, radial_basis_size=8, radial_funcs_count=4, alpha_scalar_moments=92)),
        (6, ["Si", "Ge"], dict(species_count=2, radial_basis_size=8, radial_funcs_count=2, alpha_scalar_moments=5)),
    ])
    def test_read_descriptors(self, level, elements, expected_descriptors):
        """read_descriptors reads the basis sizes from the level template but the species count from the elements."""
        template_path = Path(mtp_package.__file__).parent.parent / "MTP_templates" / f"{level:02d}.almtp"
        configuration = MtpConfiguration(elements=elements, level=level, max_dist=4.0)

        assert configuration.read_descriptors(template_path) == expected_descriptors

    def test_write_only_updates_max_dist(self, tmp_path):
        """Writing max_dist updates that line only, leaving the level-determined parameters readable."""
        template_copy = tmp_path / "06.almtp"
        shutil.copy(MTP_TEMPLATE_06, template_copy)

        MtpConfiguration(elements=["Si"], level=6, max_dist=6.5).write_to_file(template_copy)
        assert "max_dist = 6.5" in template_copy.read_text(errors="ignore")

        reread = MtpConfiguration(elements=["Si"], level=6, max_dist=1.0)
        reread.read_from_file(template_copy)
        assert (reread.species_count, reread.radial_basis_size, reread.alpha_scalar_moments) == (1, 8, 5)

    @pytest.mark.parametrize("invalid_kwargs", [
        dict(elements=[], level=6, max_dist=4.0),                       # empty element list
        dict(elements=["Si", "Si"], level=6, max_dist=4.0),            # duplicated element
        dict(elements=["Xx"], level=6, max_dist=4.0),                  # not a real element
        dict(elements=["Si"], level=0, max_dist=4.0),                  # non-positive level
        dict(elements=["Si"], level=6, max_dist=0.0),                  # non-positive cutoff
        dict(elements=["Si"], level=6, max_dist=4.0, energy_weight=-1.0),  # negative weight
    ])
    def test_invalid_configuration_raises(self, invalid_kwargs):
        """An inconsistent configuration is rejected at construction."""
        with pytest.raises(ValueError):
            MtpConfiguration(**invalid_kwargs)
