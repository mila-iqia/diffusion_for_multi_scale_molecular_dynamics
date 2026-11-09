from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


def plot_distributions(dataset1, dataset2, label1="Dataset 1", label2="Dataset 2", xlabel="Value", title=""):
    """Plot the overlaid distributions of two datasets, for visual comparison.

    Args:
        dataset1: first dataset.
        dataset2: second dataset.
        label1: legend label for dataset1.
        label2: legend label for dataset2.
        xlabel: x-axis label.
        title: plot title.

    Returns:
        fig: the created matplotlib figure.
    """
    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    if title:
        fig.suptitle(title)
    ax = fig.add_subplot(111)

    common_params = dict(density=True, bins=100, histtype="stepfilled", alpha=0.25)
    ax.hist(dataset1, **common_params, label=label1, color="red")
    ax.hist(dataset2, **common_params, label=label2, color="green")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig.tight_layout()
    return fig
