import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branin import branin

# defining a 2d domain
x1 = np.linspace(-5, 10, 1000)
x2 = np.linspace(0,15,1000)
X1, X2 = np.meshgrid(x1, x2)
grid = np.stack([X1, X2], axis=-1)

branin_values = branin(grid)

plt.imshow(branin_values, 
    cmap='viridis', 
    extent=[x1.min(), x1.max(), x2.min(), x2.max()], 
    aspect='auto',
    origin='lower',
)
plt.colorbar(label='Branin function value')
os.makedirs('Figures', exist_ok=True)
plt.savefig('Figures/branin_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()