import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Nature-Figure Python Quick-Start Configuration
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",     # editable text in SVG
    "pdf.fonttype": 42,         # editable TrueType text in PDF
    "font.size": 7,             # use 15-24 only for large slide-sized panels
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

def save_plot(fig, filename, out_dir="outputs/plots", dpi=600):
    """
    Save matplotlib figure to out_dir in PDF, SVG, and TIFF/PNG formats (Nature standards).
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Strip extension if provided
    base_name = os.path.splitext(filename)[0]
    
    filepath_svg = os.path.join(out_dir, f"{base_name}.svg")
    filepath_pdf = os.path.join(out_dir, f"{base_name}.pdf")
    filepath_png = os.path.join(out_dir, f"{base_name}.png") # Using PNG instead of TIFF for easier viewing
    
    fig.savefig(filepath_svg, bbox_inches='tight')
    fig.savefig(filepath_pdf, bbox_inches='tight')
    fig.savefig(filepath_png, dpi=dpi, bbox_inches='tight')
    
    print(f"Plot saved to {out_dir}/{base_name}.[svg, pdf, png]")
    
def plot_error_law(horizons, errors, theoretical_fit, label="E(h)", title="Error Law"):
    """
    Plot the error vs horizon (Prediction 1)
    """
    # Use standard 1-column Nature size (e.g., 3.5 inches width)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(horizons, errors, marker='o', markersize=3, label=label, color="#2c7fb8") # Neutral/signal color
    
    if theoretical_fit is not None:
        ax.plot(horizons, theoretical_fit, linestyle='--', color='#e34a33', linewidth=1.2, label="Integrator Law")
        
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
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    scatter = ax.scatter(x=orthogonality, y=ood_ratio, s=20, alpha=0.8)
    
    # Simple manual labeling as fallback if needed, usually direct labels are preferred over legend
    for i, txt in enumerate(labels):
        ax.annotate(txt, (orthogonality[i], ood_ratio[i]), fontsize=6, xytext=(3, 3), textcoords='offset points')
        
    ax.set_xlabel("Frame Freedom (Orthogonality δ)")
    ax.set_ylabel("OOD Generalization Ratio")
    ax.set_title("Frame Freedom vs OOD")
    return fig

def plot_trajectory_divergence(time_steps, divergences, labels, title="Trajectory Divergence"):
    """
    Line plot for E3: Trajectory divergence across time
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Palette of neutral + accent
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
    for i, (div, label) in enumerate(zip(divergences, labels)):
        color = colors[i % len(colors)]
        ax.plot(time_steps, div, marker='x', markersize=3, linewidth=1.0, label=label, color=color)
        
    ax.set_xlabel("Time step (t)")
    ax.set_ylabel("Latent Trajectory Divergence")
    ax.set_title(title)
    ax.legend(fontsize=6)
    return fig
