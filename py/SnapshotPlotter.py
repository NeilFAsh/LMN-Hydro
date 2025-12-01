import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
plt.style.use('./mpl_style.txt')

#  Script that plots and saves a snapshot from a simulation output file.
#Usage example:
# python SnapshotPlotter.py --snapshot_number 10 --streamlines

#Get snapshot number from command line arguments
parser = argparse.ArgumentParser(description='Script that plots and saves a snapshot.')
parser.add_argument('--snapshot_number', '-s', type=int, help='Number of the Snapshot file to plot',required=True)
parser.add_argument("--streamlines", "-l", action="store_true", help="Enable streamline plotting")


args = parser.parse_args()
N = args.snapshot_number
filename = f"output/snapshot_{N}.npz"

# Load file safely
try:
    snap = np.load(filename)
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    sys.exit(1)

# Extract arrays
rho = snap["rho"]
vx  = snap["vx"]
vy  = snap["vy"]
# P   = snap["P"]


# Plot density
fig = plt.figure(figsize=(8,6))
bkg = plt.imshow(rho.T, origin='lower',cmap='Spectral')

# Add time annotation
props = dict(facecolor='w',alpha=0.75, edgecolor = 'gray',boxstyle='round,pad=0.25')
plt.text(0.05, 0.97, f'Snap# {N}', transform=plt.gca().transAxes,fontsize=15, bbox=props,verticalalignment='top')


if args.streamlines:
    # Create a grid for streamlines
    Y, X = np.mgrid[0:rho.shape[0], 0:rho.shape[1]]
    plt.streamplot(X, Y, vx.T, vy.T, color='k', density=1.5, linewidth=0.5, arrowsize=1)

plt.clim(0.8, 2.2)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
    
cbar = fig.colorbar(bkg, cax=cax)
cbar.set_label('Density') # Label the colorbar

plt.savefig(f"output/frame_{N:04d}.png", dpi=300, bbox_inches='tight')

plt.show()