import shutil
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.build import bulk

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.flare import \
    FlarePotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.grace import \
    GracePotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.mtp import \
    MtpPotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator

REFERENCE_FILES_DIR = Path(__file__).parent.parent / "reference_files"
STRUCTURE_FILE = REFERENCE_FILES_DIR / "structure" / "Si8.in"
MLIP_DIR = REFERENCE_FILES_DIR / "mlip"
FLARE_SGP_FILE = MLIP_DIR / "flare_sgp.json"
FLARE_PAIR_COEFF_FILE = MLIP_DIR / "lammps_flare.flare"
FLARE_MAPPED_UNCERTAINTY_FILE = MLIP_DIR / "mapped_unc_flare.flare"
MTP_FILE = MLIP_DIR / "mtp10_model.mtp"
GRACE_MODEL_FILE = MLIP_DIR / "grace_model.yaml"
GRACE_ASI_FILE = MLIP_DIR / "grace_unc.asi"


def _stillinger_weber_potential():
    return StillingerWeberPotential(sw_coefficients_file_path=REFERENCE_FILES_DIR / "aSi.sw")


def _flare_potential():
    return FlarePotential(
        pair_coeff_file_path=FLARE_PAIR_COEFF_FILE,
        mapped_uncertainty_file_path=FLARE_MAPPED_UNCERTAINTY_FILE,
    )


def _mtp_potential():
    return MtpPotential(mtp_file_path=MTP_FILE)


def _grace_potential():
    return GracePotential(model_file_path=GRACE_MODEL_FILE, active_set_file_path=GRACE_ASI_FILE)


def _lammps_executable_path():
    return Path(shutil.which("lmp") or shutil.which("lammps"))


# (potential_id, potential_factory, with_uncertainty); each case requires its pair_style in LAMMPS.
POTENTIAL_CASES = [
    pytest.param("stillinger_weber", _stillinger_weber_potential, False,
                 marks=pytest.mark.requires_pair_style("sw"), id="stillinger_weber"),
    pytest.param("flare", _flare_potential, False,
                 marks=pytest.mark.requires_pair_style("flare"), id="flare-no-uncertainty"),
    pytest.param("flare", _flare_potential, True,
                 marks=pytest.mark.requires_pair_style("flare"), id="flare-with-uncertainty"),
    pytest.param("mtp", _mtp_potential, False,
                 marks=pytest.mark.requires_pair_style("mtp"), id="mtp-no-uncertainty"),
    pytest.param("mtp", _mtp_potential, True,
                 marks=pytest.mark.requires_pair_style("mtp/extrapolation"), id="mtp-with-uncertainty"),
    pytest.param("grace", _grace_potential, False,
                 marks=pytest.mark.requires_pair_style("grace/fs"), id="grace-no-uncertainty"),
    pytest.param("grace", _grace_potential, True,
                 marks=pytest.mark.requires_pair_style("grace/fs"), id="grace-with-uncertainty"),
]


class BaseTestLammpsSinglePointCalculator:

    @pytest.fixture()
    def structure(self):
        return ase.io.read(str(STRUCTURE_FILE), format="lammps-data", atom_style="atomic")

    @pytest.mark.parametrize("potential_id, potential_factory, with_uncertainty", POTENTIAL_CASES)
    def test_single_point(self, lammps_runner, structure, potential_id, potential_factory, with_uncertainty):
        """A single-point calculation returns finite energy and forces (and uncertainty when requested)."""
        calculator = LammpsSinglePointCalculator(
            potential_factory(), lammps_runner, with_uncertainty=with_uncertainty
        )
        result = calculator.calculate(structure)

        number_of_atoms = len(structure)
        assert np.ndim(result.energy) == 0 and np.isfinite(result.energy)
        assert result.forces.shape == (number_of_atoms, 3)
        assert np.all(np.isfinite(result.forces))

        if with_uncertainty:
            assert result.uncertainties.shape == (number_of_atoms,)
            assert np.all(np.isfinite(result.uncertainties))

    @pytest.mark.requires_pair_style("sw")
    def test_calculate_many_matches_single(self, lammps_runner, structure):
        """calculate_many (one looping run) returns one result per structure, matching individual calculate()."""
        structures = []
        for scale in (1.0, 1.03, 1.06):
            scaled_structure = structure.copy()
            scaled_structure.set_cell(scaled_structure.cell * scale ** (1 / 3), scale_atoms=True)
            structures.append(scaled_structure)
        calculator = LammpsSinglePointCalculator(_stillinger_weber_potential(), lammps_runner)

        batched_results = calculator.calculate_many(structures)
        assert len(batched_results) == len(structures)

        for single_structure, batched_result in zip(structures, batched_results):
            single_result = calculator.calculate(single_structure)
            np.testing.assert_allclose(batched_result.energy, single_result.energy, atol=1e-6)
            np.testing.assert_allclose(batched_result.forces, single_result.forces, atol=1e-6)

        # Distinct volumes give distinct energies: guards against every dump being parsed from one frame.
        energies = [result.energy for result in batched_results]
        assert len(set(np.round(energies, 6))) == len(energies)

    @pytest.mark.requires_pair_style("sw")
    def test_non_orthogonal_box_round_trips(self, lammps_runner):
        """A triclinic (non-orthogonal) configuration is read back from the oracle dump unchanged.

        This guards the LAMMPS-dump box parser: a tilted box must not be silently misread as orthogonal.
        """
        # ase's diamond primitive cell has 60-degree angles, so the repeated cell is genuinely triclinic.
        input_structure = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
        assert not np.allclose(input_structure.cell.angles(), 90.0)

        calculator = LammpsSinglePointCalculator(_stillinger_weber_potential(), lammps_runner)
        output_structure = calculator.calculate(input_structure).atoms

        # Same cell (rotation-invariant lattice parameters) and same atoms (fractional coordinates, mod 1).
        np.testing.assert_allclose(
            output_structure.cell.cellpar(), input_structure.cell.cellpar(), atol=1e-4
        )
        fractional_difference = (
            output_structure.get_scaled_positions() - input_structure.get_scaled_positions() + 0.5
        ) % 1.0 - 0.5
        np.testing.assert_allclose(fractional_difference, 0.0, atol=1e-4)

    @pytest.mark.requires_flare
    @pytest.mark.requires_pair_style("flare")
    def test_mapped_flare_matches_sgp(self, lammps_runner, structure):
        """The mapped FLARE potential run through LAMMPS agrees with its source SGP on energy and forces."""
        import json

        from flare.bffs.sgp import SGP_Wrapper

        from diffusion_for_multi_scale_molecular_dynamics.oracle.flare_single_point_calculator import \
            FlareSinglePointCalculator

        with open(FLARE_SGP_FILE) as file:
            sgp_model, kernels = SGP_Wrapper.from_dict(json.load(file))
        assert kernels  # keep the kernels alive to avoid a garbage-collection segfault in the C++ backend
        sgp_calculator = FlareSinglePointCalculator(sgp_model)

        lammps_calculator = LammpsSinglePointCalculator(_flare_potential(), lammps_runner)

        lammps_result = lammps_calculator.calculate(structure)
        sgp_result = sgp_calculator.calculate(structure)

        np.testing.assert_allclose(lammps_result.energy, sgp_result.energy, atol=1e-1)
        np.testing.assert_allclose(lammps_result.forces, sgp_result.forces, atol=1e-1)


@pytest.mark.not_on_github
@pytest.mark.requires_inprocess_lammps
class TestInProcess(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return InProcessLammpsRunner()


@pytest.mark.not_on_github
@pytest.mark.requires_lammps_bin
class TestSubprocessSerial(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return SubprocessLammpsRunner(lammps_executable_path=_lammps_executable_path(), mpi_processors=1)


@pytest.mark.not_on_github
@pytest.mark.requires_lammps_bin
@pytest.mark.requires_mpirun
class TestSubprocessMpi(BaseTestLammpsSinglePointCalculator):

    @pytest.fixture()
    def lammps_runner(self):
        return SubprocessLammpsRunner(lammps_executable_path=_lammps_executable_path(), mpi_processors=2)


def test_calculate_many_empty_returns_empty():
    """An empty batch short-circuits: no LAMMPS run, an empty list back."""
    calculator = LammpsSinglePointCalculator(_stillinger_weber_potential(), lammps_runner=object())
    assert calculator.calculate_many([]) == []


def test_stillinger_weber_potential_rejects_uncertainty():
    potential = StillingerWeberPotential(sw_coefficients_file_path=REFERENCE_FILES_DIR / "aSi.sw")
    with pytest.raises(ValueError):
        potential.interaction_commands("Si", with_uncertainty=True)
