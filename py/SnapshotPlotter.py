import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import sys
import os

#  Script that plots and saves a snapshot from a simulation output file.
# Usage example:
# python SnapshotPlotter.py --snapshot_number 10 
# python SnapshotPlotter.py --snapshot_number 10 --property rho --streamlines

#Get snapshot number from command line arguments
parser = argparse.ArgumentParser(description='Script that plots and saves a snapshot.')
parser.add_argument('--snapshot_number', '-n', type=int, help='Number of the Snapshot file to plot',required=True)
parser.add_argument('--property', '-p', type=str, help='Property to plot (e.g., rho, vx, vy)', default='rho')
parser.add_argument("--streamlines", "-sl", action="store_true", help="Enable streamline plotting")


args = parser.parse_args()
N = args.snapshot_number
property = args.property

#Create figs directory if it doesn't exist
fig_dir = "./figs/" 
os.makedirs(fig_dir, exist_ok=True)

# Load snapshot file
snap_dir = "./output/"
os.makedirs(snap_dir, exist_ok=True)
filename = snap_dir+f"snapshot_{N:04d}.npz"

# Check if property is valid
if property not in ['rho', 'vx', 'vy']:
    print(f"Error: Property '{property}' not found in snapshot. Choose from 'rho', 'vx', 'vy'.")
    sys.exit(1)

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

if property == 'rho':
    data = rho
    vmin = 0.8
    vmax = 2.2
    lbl = 'Density'
elif property == 'vx':
    data = vx
    vmin = -0.7
    vmax = 0.7
    lbl = 'Velocity X'
elif property == 'vy':
    data = vy
    vmin = -0.5
    vmax = 0.5
    lbl = 'Velocity Y'

# Plot density
fig = plt.figure(figsize=(8,6))

bkg = plt.imshow(data.T, origin='lower',cmap='Spectral_r',vmin=vmin,vmax=vmax)

# Add time annotation
props = dict(facecolor='w',alpha=0.75, edgecolor = 'gray',boxstyle='round,pad=0.25')
time = snap['conserved'][0][0]

# plt.text(0.05, 0.97, f'Snap# {N}', transform=plt.gca().transAxes,fontsize=15, bbox=props,verticalalignment='top')
plt.text(0.05, 0.93, f'Time: {time:.3f}', transform=plt.gca().transAxes,fontsize=12, bbox=props,verticalalignment='top')

if args.streamlines:
    # Create a grid for streamlines
    Y, X = np.mgrid[0:rho.shape[0], 0:rho.shape[1]]
    plt.streamplot(X, Y, vx.T, vy.T, color='k', density=1.5, linewidth=0.5, arrowsize=1)

plt.clim(vmin, vmax)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
    
cbar = fig.colorbar(bkg, cax=cax)
cbar.set_label(lbl) # Label the colorbar

# Save figure
plt.savefig(fig_dir+f"frame_{property}_{N:04d}.png", dpi=300, bbox_inches='tight')

plt.show()