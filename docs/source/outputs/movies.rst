Movie Maker
==============================

LMN-Hydro includes a bash script to create movies from a series of snapshot plots. This script is located in the `py` directory and is named `MakeMovie.sh`.

How it works:
------------
1. Scrip parses through the output folders where each snapshot output data is stored
2. For each snapshot, script generates individual plots using the provided python script 'SnapshotPlotter.py' using the specified property (e.g., density, pressure, velocityMagnitude), 
3. Compiles these plots into a movie file using FFmpeg.

How to use:
------------
1. Ensure you have Python and FFmpeg installed on your system.
2. Run the script from the command line,

    .. code-block:: bash

         bash MakeMovie.sh <property> 

    Replace `<number_of_snapshots>` with the total number of snapshots you have, and `<property>` with the variable you want to visualize (e.g., Density, Pressure, VelocityMagnitude). You can also specify additional options such as frame rate and output file name.