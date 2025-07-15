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
checkpoint_dir = experiment_dir / "flare_checkpoints" / "sigma_1000.0_n_10"

images_dir = experiment_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    checkpoint_path = checkpoint_dir / "flare_model_pretrained.json"
    flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)

    flare_trainer.sgp_model.sparse_gp.precompute_KnK()

    starting_hyperparameters = 1.0 * flare_trainer.sgp_model.sparse_gp.hyperparameters
    initial_sigma, initial_sigma_e, initial_sigma_f, _ = starting_hyperparameters

    list_sigma = np.exp(np.linspace(np.log(0.1), np.log(10000), 81))
    list_sigma_e = np.exp(np.linspace(np.log(0.0001), np.log(500), 81))
    list_sigma_f = np.exp(np.linspace(np.log(0.001), np.log(5), 81))

    list_log_likelihoods_vs_sigma = []
    list_log_likelihoods_vs_sigma_e = []
    list_log_likelihoods_vs_sigma_f = []

    for sigma in list_sigma:
        hyperparameters = 1.0 * starting_hyperparameters
        hyperparameters[0] = sigma

        nll, grads = compute_negative_likelihood_grad_stable(hyperparameters,
                                                             flare_trainer.sgp_model.sparse_gp,
                                                             precomputed=True)
        list_log_likelihoods_vs_sigma.append(-nll)
    list_log_likelihoods_vs_sigma = np.array(list_log_likelihoods_vs_sigma)

    for sigma_e in list_sigma_e:
        hyperparameters = 1.0 * starting_hyperparameters
        hyperparameters[1] = sigma_e

        nll, grads = compute_negative_likelihood_grad_stable(hyperparameters,
                                                             flare_trainer.sgp_model.sparse_gp,
                                                             precomputed=True)
        list_log_likelihoods_vs_sigma_e.append(-nll)
    list_log_likelihoods_vs_sigma_e = np.array(list_log_likelihoods_vs_sigma_e)

    for sigma_f in list_sigma_f:
        hyperparameters = 1.0 * starting_hyperparameters
        hyperparameters[2] = sigma_f

        nll, grads = compute_negative_likelihood_grad_stable(hyperparameters,
                                                             flare_trainer.sgp_model.sparse_gp,
                                                             precomputed=True)
        list_log_likelihoods_vs_sigma_f.append(-nll)
    list_log_likelihoods_vs_sigma_f = np.array(list_log_likelihoods_vs_sigma_f)

    figsize = (1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle("FLARE on Si 2x2x2: Log Likelihood\n FLARE trained on 10 structures")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.set_title(rf"$\sigma_e$ = {initial_sigma_e}, $\sigma_f$ = {initial_sigma_f}")
    ax2.set_title(rf"$\sigma$ = {initial_sigma}, $\sigma_f$ = {initial_sigma_f}")
    ax3.set_title(rf", $\sigma$ = {initial_sigma}, $\sigma_e$ = {initial_sigma_e}")

    ax1.loglog(list_sigma, list_log_likelihoods_vs_sigma, '-', color='k')
    ax2.loglog(list_sigma_e, list_log_likelihoods_vs_sigma_e, '-', color='k')
    ax3.loglog(list_sigma_f, list_log_likelihoods_vs_sigma_f, '-', color='k')

    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ymin3, ymax3 = ax3.get_ylim()

    ax1.vlines(1000.0, ymin1, ymax1, color='red', label=r'$\sigma$ = 1000.')
    ax2.vlines(1.0, ymin2, ymax2, color='red', label=r'$\sigma_e$ = 1.0')
    ax3.vlines(0.05, ymin3, ymax3, color='red', label=r'$\sigma_f$ = 0.05')

    ax1.set_xlabel(r"$\sigma$")
    ax2.set_xlabel(r"$\sigma_e$")
    ax3.set_xlabel(r"$\sigma_f$")

    ax1.set_ylim(ymin1, ymax1)
    ax2.set_ylim(ymin2, ymax2)
    ax3.set_ylim(ymin3, ymax3)

    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel("Log Likelihood")
        ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(images_dir / "log_likelihood.png")
