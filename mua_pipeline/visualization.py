# mua_pipeline/visualization.py
"""
Visualization utilities for MUA Pipeline
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec

def plot_results(predictions, actual, title=None):
    """
    Plot prediction results with scatter plot and error distribution.
    """
    if predictions is None:
        return

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })

    fig = plt.figure(figsize=(8.27, 3.5))

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        wspace=0.38,
        left=0.12,
        right=0.95,
        top=0.78,
        bottom=0.20
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ── Scatter ──────────────────────────────────────────────────────────
    ax1.scatter(actual, predictions, alpha=0.6, s=22,
                edgecolors='k', linewidth=0.3,
                color='#4472C4', zorder=3)

    r_val, p_val = pearsonr(actual, predictions)
    slope = r_val * (np.std(predictions) / np.std(actual))
    intercept = np.mean(predictions) - slope * np.mean(actual)
    x_lo, x_hi = actual.min(), actual.max()
    ax1.plot([x_lo, x_hi],
             [slope * v + intercept for v in [x_lo, x_hi]],
             'r-', lw=2, alpha=0.85, zorder=2)

    ax1.set_xlabel('Actual Values', fontsize=11, labelpad=5)
    ax1.set_ylabel('Predicted Values', fontsize=11, labelpad=5)

    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax1.text(0.05, 0.95, f'r = {r_val:.3f}\n{p_text}',
             transform=ax1.transAxes, fontsize=9.5, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#bbbbbb', alpha=0.9))

    ax1.grid(True, alpha=0.25, linewidth=0.4)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Histogram ────────────────────────────────────────────────────────
    errors = predictions - actual

    bins = np.linspace(errors.min() - 0.5, errors.max() + 0.5, 25)
    ax2.hist(errors, bins=bins, density=True, alpha=0.85,
             color='#70AD47', edgecolor='k', linewidth=0.4)
    ax2.axvline(0, color='k', linestyle='--', linewidth=1.3, alpha=0.8)

    mu, std = stats.norm.fit(errors)
    x_kde = np.linspace(errors.min() - 0.5, errors.max() + 0.5, 300)
    ax2.plot(x_kde, stats.norm.pdf(x_kde, mu, std),
             'k-', lw=1.8, alpha=0.85)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ax2.set_xlabel('Prediction Error', fontsize=11, labelpad=5)
    ax2.set_ylabel('Density', fontsize=11, labelpad=5)
    ax2.text(0.97, 0.95, f'MAE = {mae:.3f}\nRMSE = {rmse:.3f}',
             transform=ax2.transAxes, fontsize=9.5,
             va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#bbbbbb', alpha=0.9))

    ax2.grid(True, alpha=0.25, linewidth=0.4, axis='y')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel letters ────────────────────────────────────────────────────
    fig.canvas.draw()

    for ax, label in zip([ax1, ax2], ['A', 'B']):
        bbox = ax.get_position()
        fig.text(bbox.x0 - 0.07, bbox.y1 + 0.020, label,
                 fontsize=14, fontweight='bold',
                 va='bottom', ha='left',
                 transform=fig.transFigure)

    # ── Suptitle ─────────────────────────────────────────────────────────
    if title:
        fig.text(0.5, 0.97, title,
                 fontsize=13, fontweight='bold',
                 va='top', ha='center',
                 transform=fig.transFigure)
    
    plt.show()
