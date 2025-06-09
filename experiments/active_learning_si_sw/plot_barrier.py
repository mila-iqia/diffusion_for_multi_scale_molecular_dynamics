import yaml
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import \
    get_saddle_energy
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


experiment_dir = TOP_DIR / "experiments/active_learning_si_sw"
reference_artn_output_file = experiment_dir / "Si-vac_sw_potential/artn.out"

list_campaign_ids = [1, 2, 3]


if __name__ == "__main__":
    with open(reference_artn_output_file, "r") as fd:
        artn_output = fd.read()
        E_gt = get_saddle_energy(artn_output)

    list_thresholds = []
    list_E = []

    for campaign_id in list_campaign_ids:
        campaign_dir = experiment_dir / f"active_learning_campaign_{campaign_id}"

        with open(campaign_dir / "campaign_details.yaml", "r") as fd:
            campaign_details = yaml.load(fd, Loader=yaml.FullLoader)

        final_round = campaign_details['final_round']
        threshold = campaign_details['uncertainty_threshold']

        final_round_dir = campaign_dir / f"round_{final_round}"
        artn_output_file = final_round_dir / "lammps_artn/artn.out"
        try:
            with open(artn_output_file, "r") as fd:
                artn_output = fd.read()
                E = get_saddle_energy(artn_output)
            list_E.append(E)
            list_thresholds.append(threshold)
        except Exception:
            print("Failed to Extract Saddle Energy")

    error = E_gt - list_E[-1]

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Saddle Point Energy From Active Learning\n Best Result Error : {1000 * error: 4.1f} meV")
    ax1 = fig.add_subplot(111)

    ax1.semilogx(list_thresholds, list_E, 'ro-', ms=10, label='FLARE + Active Learning')

    ax1.set_ylabel('Saddle Point Energy (eV)')
    ax1.set_xlabel('Uncertainty Threshold (unitless)')

    ax1.hlines(E_gt, *ax1.set_xlim(), linestyles='-', colors='green', label='SW value')

    ax1.legend(loc=3)
    fig.tight_layout()

    plt.show()
