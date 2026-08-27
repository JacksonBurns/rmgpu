from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    *,
    model_name: str = "Chemprop",
    error_band: float = 0.5,
    save_path: str | Path | None = None,
    gridsize: int = 70,
    figsize: tuple[float, float] = (5.0, 4.0),
) -> tuple[plt.Figure, Axes]:
    """Generates a styled parity hexbin plot with an error distribution pie chart inset."""
    # Filter NaNs (e.g. from partially sparse multi-task datasets)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) == 0:
        raise ValueError(f"No valid non-NaN samples for target '{target_name}'.")

    # Dynamic plot limits based on data range
    val_min = min(yt.min(), yp.min())
    val_max = max(yt.max(), yp.max())
    padding = (val_max - val_min) * 0.08
    lim_min = val_min - padding
    lim_max = val_max + padding

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax: Axes
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)

    # Parity hexbin
    hb = ax.hexbin(
        x=yt,
        y=yp,
        gridsize=gridsize,
        cmap="viridis",
        mincnt=1,
        extent=(lim_min, lim_max, lim_min, lim_max),
    )

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Sample count")

    # Guide lines: y = x, y = x +/- error_band
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r", linewidth=1.0)
    ax.plot(
        [lim_min, lim_max],
        [lim_min + error_band, lim_max + error_band],
        "r--",
        linewidth=0.5,
    )
    ax.plot(
        [lim_min, lim_max],
        [lim_min - error_band, lim_max - error_band],
        "r--",
        linewidth=0.5,
    )

    clean_target = target_name.replace("_", " ")
    ax.set_title(rf"$\tt{{{model_name}}}$: {clean_target}")
    ax.set_xlabel(f"True {clean_target}")
    ax.set_ylabel(f"Predicted {clean_target}")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Metric annotations (transAxes ensures proper placement across all scales)
    r, _ = spearmanr(yt, yp)
    mse = mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    textstr = "\n".join(
        (
            f"$\\bf{{Spearman \\rho}}:$ {r:.2f}",
            f"$\\bf{{MAE}}:$ {mae:.2f}",
            f"$\\bf{{MSE}}:$ {mse:.2f}",
        )
    )
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    return fig, ax


def plot_multitask_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] | list[str],
    *,
    output_dir: str | Path,
    model_name: str = "Chemprop",
    error_band: float = 1.0,
) -> None:
    """Iterates through targets and outputs individual styled parity plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, target in enumerate(target_names):
        save_file = out / f"{target}_parity.png"
        parity_plot(
            y_true=y_true[:, i],
            y_pred=y_pred[:, i],
            target_name=target,
            model_name=model_name,
            error_band=error_band,
            save_path=save_file,
        )
