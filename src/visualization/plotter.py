import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

def save_plot(fig, filename, out_dir="outputs/plots"):
    """
    Save matplotlib figure to out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {filepath}")
    
def plot_error_law(horizons, errors, theoretical_fit, label="E(h)", title="Error Law"):
    """
    Plot the error vs horizon (Prediction 1)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(horizons, errors, marker='o', label=label)
    if theoretical_fit is not None:
        ax.plot(horizons, theoretical_fit, linestyle='--', color='red', label="Integrator Law (theoretical)")
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Error E(h)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    return fig

def plot_frame_freedom_vs_ood(orthogonality, ood_ratio, labels):
    """
    Scatter plot for E2: Frame Freedom vs OOD Generalization
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=orthogonality, y=ood_ratio, ax=ax, hue=labels, s=100)
    ax.set_xlabel("Frame Freedom (Orthogonality δ)")
    ax.set_ylabel("OOD Generalization Ratio")
    ax.set_title("Frame Freedom vs OOD Generalization")
    return fig

def plot_trajectory_divergence(time_steps, divergences, labels, title="Trajectory Divergence (Plan Transfer)"):
    """
    Line plot for E3: Trajectory divergence across time
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for div, label in zip(divergences, labels):
        ax.plot(time_steps, div, marker='x', label=label)
    ax.set_xlabel("Time step (t)")
    ax.set_ylabel("Latent Trajectory Divergence")
    ax.set_title(title)
    ax.legend()
    return fig
