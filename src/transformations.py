import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branin import branin
from kernel_dnsty_est import plot_kde

def log_transform(x):
    return np.log(x)

def exp_transform(x):
    return np.exp(x)

def sqrt_transform(x):
    return np.sqrt(x)

def inverse_transform(x):
    return 1/x

def local_means_and_vars(values, n_blocks=10):
    ny, nx = values.shape
    by, bx = ny // n_blocks, nx // n_blocks
    means, vars_ = [], []
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = values[i*by:(i+1)*by, j*bx:(j+1)*bx]
            means.append(block.mean())
            vars_.append(block.var())
    return np.array(means), np.array(vars_)

x1 = np.linspace(-5, 10, 1000)
x2 = np.linspace(0,15,1000)
X1, X2 = np.meshgrid(x1, x2)
grid = np.stack([X1, X2], axis=-1)

branin_values = branin(grid)
log_values = log_transform(branin_values)
exp_values = exp_transform(branin_values)
sqrt_values = sqrt_transform(branin_values)
inverse_values = inverse_transform(branin_values)

plt.imshow(log_values, cmap='viridis', extent=[x1.min(), x1.max(), x2.min(), x2.max()], aspect='auto', origin='lower')
plt.colorbar(label='Log transformed Branin function value')
plt.savefig('Figures/log_transformed_branin_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

m_raw, v_raw = local_means_and_vars(branin_values)
m_log, v_log = local_means_and_vars(log_values)
m_exp, v_exp = local_means_and_vars(exp_values)
m_sqrt, v_sqrt = local_means_and_vars(sqrt_values)
m_inverse, v_inverse = local_means_and_vars(inverse_values)

print("std of local means (raw):", m_raw.std())
print("std of local means (log):", m_log.std())
#print("std of local means (exp):", m_exp.std())
#print("std of local means (sqrt):", m_sqrt.std())
print("std of local means (inverse):", m_inverse.std())

print("std of local vars (raw):", v_raw.std())
print("std of local vars (log):", v_log.std())
#print("std of local vars (exp):", v_exp.std())
#print("std of local vars (sqrt):", v_sqrt.std())
print("std of local vars (inverse):", v_inverse.std())

lda = pd.read_csv('data/lda.csv', header=None)
svm = pd.read_csv('data/svm.csv', header=None)

lda_y = np.log(lda[3].to_numpy())
svm_y = np.log(svm[3].to_numpy())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_kde(lda_y, axes[0], "Log(LDA)")
plot_kde(svm_y, axes[1], "Log(SVM)")
plt.savefig('Figures/kde_transformed.png', dpi=150, bbox_inches='tight')
plt.tight_layout()
plt.show()