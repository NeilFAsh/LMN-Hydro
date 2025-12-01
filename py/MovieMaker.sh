#!/bin/bash

# This script generates a movie from a series of snapshot images
# produced by the LMN-Hydro simulation. It uses ffmpeg to compile
# the images into a video file. 

# Allows for optional use of command line argument to enable streamline plotting.

# Usaege: ./MovieMaker.sh output_name

#Read output name from command line argument
OUTPUT_NAME=${1:-movie.mp4}

STREAM_FLAG="" #Optional flag for streamline plotting. If needed, set to "--streamlines" or "-l"

# Finds all simulation files in the output directory, sorts and generates the frames for the movie.
#!/bin/bash

mkdir -p output

echo "Generating frames..."

for file in output/snapshot_*.npz; do
    num=$(echo $file | sed -E 's/.*snapshot_([0-9]+)\.npz/\1/')
    printf "Plotting snapshot %s\n" "$num"
    python3 SnapshotPlot.py --snapshot_number $num $STREAM_FLAG
done

echo "Creating movie with ffmpeg..."
ffmpeg -framerate 30 -i output/frame_%04d.png -pix_fmt yuv420p output/${OUTPUT_NAME}.mp4

echo "Done! Saved as output/${OUTPUT_NAME}.mp4"