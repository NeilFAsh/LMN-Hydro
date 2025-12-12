# LMN-Hydro

LMN-Hydro is a simple but extensible 2D hydrodynamics simulation code written in C++. It is designed for simulating fluid dynamics problems in astrophysics and engineering applications.

<img width="841" height="743" alt="image" src="https://github.com/user-attachments/assets/b1b46dcd-9c69-4656-977f-eda1e647fe8e" />


## How to install

CMake is our primary build system. To install LMN-Hydro with a serial (CPU) backend, follow these steps:

    cmake -B build -DUSE_KOKKOS=OFF
    cd build
    make 
    
To build with Kokkos (GPU) support enabled, run the above with:

    cmake -B build -DUSE_KOKKOS=ON

Note that CUDA is required for Kokkos support. LMN-Hydro assumes SKX and VOLTA70 architectures (passed to Kokkos) by default, this can be changed by modifying CMakeLists.txt.

## Documentation

Find more details on how to use our package here: https://lmn-hydro.readthedocs.io/en/latest/  
