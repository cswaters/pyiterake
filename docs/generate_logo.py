import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Set up the figure with a transparent background
fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
ax.set_aspect('equal')
ax.axis('off')

# Create a colorful background
rect = Rectangle((0, 0), 1, 1, facecolor='#2A2D34', edgecolor='none', alpha=0.9)
ax.add_patch(rect)

# Add some graphical elements representing "iterative raking"
# 1. Create some data points
np.random.seed(42)
x = np.random.normal(0.5, 0.15, 30)
y = np.random.normal(0.5, 0.15, 30)

# 2. Create multiple "layers" of points with different colors to represent iterations
colors = ['#F7E733', '#26C99E', '#F24C4E', '#FFFFFF']
sizes = [80, 70, 60, 50]

for i, (color, size) in enumerate(zip(colors, sizes)):
    offset = 0.03 * i
    ax.scatter(x + offset, y + offset, c=color, s=size, alpha=0.8, edgecolors='none')

# 3. Add horizontal and vertical lines to represent "raking"
for i in range(5):
    pos = 0.2 + i * 0.15
    ax.plot([0, 1], [pos, pos], color='white', alpha=0.4, linewidth=1)
    ax.plot([pos, pos], [0, 1], color='white', alpha=0.4, linewidth=1)

# Add the name "PyIterake" 
ax.text(0.5, 0.12, "PyIterake", fontsize=28, fontweight='bold', color='white', 
        ha='center', va='center', family='sans-serif')

# Set the axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Save the logo with a transparent background
plt.savefig('images/logo.png', dpi=300, bbox_inches='tight', transparent=True)
print("Logo generated at 'images/logo.png'")

# Create a smaller version for favicon
plt.savefig('images/favicon.png', dpi=100, bbox_inches='tight', transparent=True)
print("Favicon generated at 'images/favicon.png'")

plt.close() 