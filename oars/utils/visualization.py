import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from matplotlib.colors import CenteredNorm

# Figure 4 helper
def cplot(Z, title=None):
    # vmin = np.floor(np.min(np.minimum(W, Z)))
    cm = plt.cm.coolwarm
    n = Z.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    x = -np.arange(n)
    y = np.arange(n)

    pc = ax.pcolormesh(x, y, Z, norm=CenteredNorm(), cmap=cm)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    ax.set_aspect('equal')

    # Add colorbar
    plt.tight_layout()
    
    fig.colorbar(pc, ax=ax.ravel().tolist())
    return fig