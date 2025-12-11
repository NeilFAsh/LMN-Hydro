Snapshot Plotter
==============================

LMN-Hydro includes a ready to use Python script to generate single snapshot plots of simulation data. This script is located in the `py` directory and is named `SnapshotPlotter.py`.
To use the Snapshot Plotter, follow these steps:

1. Ensure you have Python installed on your system, along with the necessary libraries such as Matplotlib and NumPy.
2. Run the script from the command line, 

    .. code-block:: bash
    
         python3 SnapshotPlotter.py -n <snapshot_number> [options]
    
    Replace `<snapshot_number>` with the number of the snapshot you want to plot. You can also specify additional options to customize the output, such as the variables to plot, or overlayed velocity streamlines. Default property is density.

