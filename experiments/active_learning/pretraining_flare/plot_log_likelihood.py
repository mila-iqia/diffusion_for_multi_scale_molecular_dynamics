import numpy as np
from flare.bffs.sgp.sparse_gp import compute_negative_likelihood_grad_stable
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)

experiment_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/"
checkpoint_dir = experiment_dir / "flare_checkpoints" / "number_of_structures_16"

images_dir = experiment_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

sigma = 1.0

default_sigma_e = 1e-1
default_sigma_f = 1e-3

if __name__ == "__main__":

    checkpoint_path = checkpoint_dir / "flare_model_preloaded.json"
    flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)

    flare_trainer.sgp_model.sparse_gp.precompute_KnK()

    list_sigma_e = np.exp(np.linspace(np.log(0.0001), np.log(1), 81))
    list_sigma_f = np.exp(np.linspace(np.log(0.0001), np.log(1), 81))

    list_log_likelihoods_vs_sigma_e = []
    list_log_likelihoods_vs_sigma_f = []

    for sigma_e in list_sigma_e:
        hyperparameters = np.array([sigma, default_sigma_e, default_sigma_f, 1.0])
        hyperparameters[1] = sigma_e

        nll, grads = compute_negative_likelihood_grad_stable(hyperparameters,
                                                             flare_trainer.sgp_model.sparse_gp,
                                                             precomputed=True)
        list_log_likelihoods_vs_sigma_e.append(-nll)
    list_log_likelihoods_vs_sigma_e = np.array(list_log_likelihoods_vs_sigma_e)

    for sigma_f in list_sigma_f:
        hyperparameters = np.array([sigma, default_sigma_e, default_sigma_f, 1.0])
        hyperparameters[2] = sigma_f

        nll, grads = compute_negative_likelihood_grad_stable(hyperparameters,
                                                             flare_trainer.sgp_model.sparse_gp,
                                                             precomputed=True)
        list_log_likelihoods_vs_sigma_f.append(-nll)
    list_log_likelihoods_vs_sigma_f = np.array(list_log_likelihoods_vs_sigma_f)

    figsize = (1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle("FLARE on Si 2x2x2: Log Likelihood\n FLARE trained on 16 structures")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title(rf"$\sigma$ = {sigma}, $\sigma_f$ = {default_sigma_f}")
    ax2.set_title(rf", $\sigma$ = {sigma}, $\sigma_e$ = {default_sigma_e}")

    ax1.loglog(list_sigma_e, list_log_likelihoods_vs_sigma_e, '-', color='k')
    ax2.loglog(list_sigma_f, list_log_likelihoods_vs_sigma_f, '-', color='k')

    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()

    ax1.vlines(default_sigma_e, ymin1, ymax1, color='red', label=rf'$\sigma_e$ = {default_sigma_e:3.2e}')
    ax2.vlines(default_sigma_f, ymin2, ymax2, color='red', label=rf'$\sigma_f$ = {default_sigma_f: 3.2e}')

    ax1.set_xlabel(r"$\sigma_e$")
    ax2.set_xlabel(r"$\sigma_f$")

    ax1.set_ylim(ymin1, ymax1)
    ax2.set_ylim(ymin2, ymax2)

    for ax in [ax1, ax2]:
        ax.set_ylabel("Log Likelihood")
        ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(images_dir / "log_likelihood.png")
