#ifndef FLUID_KHI_HPP
#define FLUID_KHI_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "fluid.hpp"
#include "SnapshotWriter.hpp"

using namespace fluid;

class Fluid_KHI : public Fluid {
    public:
        Fluid_KHI(int Nx, int Ny, double boxSizeX, double boxSizeY)
            : Fluid(Nx, Ny, boxSizeX, boxSizeY) {};

        void initialize_KHI();
};


#endif // FLUID_HPP