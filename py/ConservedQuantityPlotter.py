import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import glob

# Create directories for images to go, if not already made
fig_dir = "./figs/" 
os.makedirs(fig_dir, exist_ok=True)

movie_dir = fig_dir+"conservation_movie/"
os.makedirs(movie_dir, exist_ok=True)

snap_dir = "./output/"

all_entries = os.listdir(snap_dir)

# Filter for only files
files_only = np.sort([entry for entry in all_entries if os.path.isfile(os.path.join(snap_dir, entry))])

# Initialize arrays
time          = np.zeros(len(files_only))
totalMass     = np.zeros(len(files_only))
totalEnergy   = np.zeros(len(files_only))
totalMomentum = np.zeros(len(files_only))
totalMomX     = np.zeros(len(files_only))
totalMomY     = np.zeros(len(files_only))

for it, filename in enumerate(files_only):
    print(filename, time[it])

    # Load file safely
    try:
        snap = np.load(snap_dir+filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)
    
    # Read in conserved quantities
    conserved = snap['conserved'][0]

    time[it]        = conserved[0]
    totalMass[it]   = conserved[1]
    totalEnergy[it] = conserved[2]
    totalMomX[it]   = conserved[3]
    totalMomY[it]   = conserved[4]
    #totalMass[it]   = conserved[0]
    #totalEnergy[it] = conserved[1]
    #totalMomX[it]   = conserved[2]
    #totalMomY[it]   = conserved[3]

normMass = np.abs(totalMass/totalMass[0] - 1)
normEnergy = np.abs(totalEnergy/totalEnergy[0] - 1)
totalMomentum = np.sqrt(totalMomX**2 + totalMomY**2)
normMomentum = np.abs(totalMomentum/totalMomentum[0] - 1)

plotmin = np.min( [np.mean(normMass) - 3*np.std(normMass),
                  np.mean(normEnergy) - 3*np.std(normEnergy),
                  np.mean(normMomentum) - 3*np.std(normMomentum)]
)
plotmax = np.max( [np.mean(normMass) + 3*np.std(normMass),
                  np.mean(normEnergy) + 3*np.std(normEnergy),
                  np.mean(normMomentum) + 3*np.std(normMomentum)]
)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.zeros_like(time[it]), np.zeros_like(normMass[it]), color='cornflowerblue', linestyle='-',label='Mass')
ax.plot(np.zeros_like(time[it]), np.zeros_like(normEnergy[it]), color='tomato', linestyle='-',label='Energy')
ax.plot(np.zeros_like(time[it]), np.zeros_like(normMomentum[it]), color='mediumseagreen', linestyle='-',label='Momentum')

for it, filename in enumerate(files_only):

    if (it % 50 != 0) and (it != 1999):
        continue
    else:
        ax.plot(time[:it], normMass[:it], color='cornflowerblue', linestyle='-', linewidth=1,)
        ax.plot(time[:it], normEnergy[:it], color='tomato', linestyle='-', linewidth=1,)
        ax.plot(time[:it], normMomentum[:it], color='mediumseagreen', linestyle='-', linewidth=1,)

    
    ax.set_xlim(time[0],time[-1])
    ax.set_ylim(5e-14, 4e-12)
    ax.set_yscale('log')
    ax.set_xlabel(r'time',fontsize=14)
    ax.set_ylabel(r'|Fractional error|',fontsize=14)
    
    plt.legend(title='Conserved quantity', loc='lower right')
    plt.savefig(movie_dir+f"frame_{it:04d}.png", dpi=300, bbox_inches='tight')