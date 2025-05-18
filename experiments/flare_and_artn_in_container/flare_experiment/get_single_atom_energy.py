import numpy as np
from ase import Atoms
from matplotlib import pyplot as plt

from flare_experiment.utilities import PLOT_STYLE_PATH, PLEASANT_FIG_SIZE
from flare_experiment.utilities.single_point_calculator import StillingerWeberSinglePointCalculator

plt.style.use(PLOT_STYLE_PATH)


lammps_executable_path = "/home/user/sources/lammps/build/lmp"

sw_coefficients_file_path = "/home/user/experiments/potentials/Si.sw"




if __name__ == '__main__':


    sw_calculator = StillingerWeberSinglePointCalculator(lammps_executable_path, sw_coefficients_file_path)

    list_acell = np.linspace(2, 5, 20)
    list_energy = []

    symbols = ['Si']

    for idx, acell in enumerate(list_acell):
        print(idx)
        cell = np.array([acell, acell, acell])
        positions = 0.5 * np.array([[acell, acell, acell]])
        atoms = Atoms(positions=positions, symbols=symbols, cell=cell, pbc=[True, True, True])
        labelled_structure = sw_calculator.calculate(atoms)

        list_energy.append(labelled_structure.energy)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Energy as a function of acell")
    ax1 = fig.add_subplot(111)
    ax1.plot(list_acell, list_energy, 'go-', label='SW in Cube')
    ax1.set_ylabel("Energy (eV)")
    ax1.set_xlabel("acell ($\AA$)")
    ax1.hlines(0, *ax1.set_xlim(), linestyles='--', color='k', label='Zero')
    ax1.legend(loc=0)
    plt.show()