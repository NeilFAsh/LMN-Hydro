#!/bin/bash

# This script generates a movie from a series of snapshot images
# produced by the LMN-Hydro simulation. It uses ffmpeg to compile
# the images into a video file. 

# Allows for optional use of command line argument to enable streamline plotting.

# Usaege: ./MovieMaker.sh output_name

#Read property from command line argument
PROPERTY=${1}
#property must be rho, vx,vy
#validate property
if [[ "$PROPERTY" != "rho" && "$PROPERTY" != "vx" && "$PROPERTY" != "vy" ]]; then
    echo "Invalid property: $PROPERTY"
    echo "Valid properties are: rho, vx, vy"
    exit 1
fi

STREAM_FLAG="-sl" #Optional flag for streamline plotting. If needed, set to "--streamlines" or "-sl". Leave "" for no streamlines.

#Find LMN-Hydro root directory
ROOT_DIR=$(git rev-parse --show-toplevel)

#Create output directory if it doesn't exist
mkdir -p figs/movies/

echo "Generating frames..."

for file in output/snapshot_*.npz; do
    num=$(echo $file | sed -E 's/.*snapshot_([0-9]+)\.npz/\1/')
    printf "Plotting snapshot %s\n" "$num"
    python3 $ROOT_DIR/py/SnapshotPlotter.py --snapshot_number $num $STREAM_FLAG
done

echo "Creating movie with ffmpeg..."
ffmpeg -framerate 30 -pattern_type glob -i "figs/frame_${PROPERTY}_*.png" -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ./figs/movies/${PROPERTY}_movie.mp4


echo "Creating plots of conserved quantities and packaging it into movie..."
python3 ../py/ConservedQuantityPlotter.py
ffmpeg -framerate 30 -pattern_type glob -i "figs/frame_*.png" -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ./figs/movies/conservation_movie.mp4