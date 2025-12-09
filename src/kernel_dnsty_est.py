import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branin import branin
from scipy.stats import gaussian_kde

lda = pd.read_csv('data/lda.csv', header=None)
svm = pd.read_csv('data/svm.csv', header=None)

lda_y = lda[3].to_numpy()
svm_y = svm[3].to_numpy()

def plot_kde(y, ax, title):
    kde = gaussian_kde(y)
    xs = np.linspace(y.min(), y.max(), 200)
    ax.plot(xs, kde(xs))
    ax.set_title(title)
    ax.set_xlabel("objective value")
    ax.set_ylabel("density")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_kde(lda_y, axes[0], "LDA")
plot_kde(svm_y, axes[1], "SVM")
plt.savefig('Figures/kde_raw.png', dpi=150, bbox_inches='tight')
plt.tight_layout()
plt.show()
