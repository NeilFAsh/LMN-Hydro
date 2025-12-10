#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>

#include "fluid_kokkos.hpp"
#include "SnapshotWriter.hpp"


using namespace fluid_kokkos;

#ifndef FLUID_KHI_HPP
#define FLUID_KHI_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "fluid.hpp"
#include "SnapshotWriter.hpp"

using namespace fluid_kokkos;

class Fluid_KHI : public Fluid {
    public:
        Fluid_KHI(int Nx, int Ny, double boxSizeX, double boxSizeY)
            : Fluid(Nx, Ny, boxSizeX, boxSizeY) {};

        void initialize_KHI();
};


#endif // FLUID_HPP


void Fluid_KHI::initialize_KHI(){
    // Initialize a Kelvin-Helmholtz Instability setup
    initialize();

    // Retrieve view objects on the device
    auto vx = this->vx.view<Kokkos::DefaultExecutionSpace>();
    auto vy = this->vy.view<Kokkos::DefaultExecutionSpace>();
    auto density = this->density.view<Kokkos::DefaultExecutionSpace>();
    auto pressure = this->pressure.view<Kokkos::DefaultExecutionSpace>();

    // Mark the device as modified
    this->vx.modify<Kokkos::DefaultExecutionSpace>();
    this->vy.modify<Kokkos::DefaultExecutionSpace>();
    this->density.modify<Kokkos::DefaultExecutionSpace>();
    this->pressure.modify<Kokkos::DefaultExecutionSpace>();

    // set up initial conditions on the device
    Kokkos::parallel_for("initialize_KHI",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j) {

            double y = (j + 0.5) * dy;
            double x = (i + 0.5) * dx;
            if(y < BoxSizeY * 0.75 && y > BoxSizeY * 0.25){
                vx(i,j) = 0.5;
                density(i,j) = 1.0;
            } else {
                vx(i,j) = -0.5;
                density(i,j) = 2.0;
            }
            // Add perturbation
            vy(i,j) = w0 * sin(4 * M_PI * x / BoxSizeX) * ( exp(-0.5*pow((y - BoxSizeY / 4) / sigma, 2))
                                                + exp(-0.5*pow((y - 3 * BoxSizeY / 4) / sigma, 2)) );
            pressure(i,j) = 2.5; // uniform pressure
        }
    );
    Kokkos::fence();

    // sync changes back to host
    this->vx.sync<Kokkos::HostSpace>();
    this->vy.sync<Kokkos::HostSpace>();
    this->density.sync<Kokkos::HostSpace>();
    this->pressure.sync<Kokkos::HostSpace>();
};

int main(){
    // Simulation parameters
    int Nx = 400;
    int Ny = 400;
    double boxSizeX = 1.0;
    double boxSizeY = 1.0;
    double tFinal = 2.0; //2.0;
    double tOut = 0.001;

    Kokkos::initialize();

    cout << "Starting Kelvin-Helmholtz Instability Simulation" << endl;
    Fluid_KHI fluid(Nx, Ny, boxSizeX, boxSizeY);
    fluid.initialize_KHI();

    try {
        fluid.runSimulation(tFinal,tOut);
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
        cerr << "Simulation terminated prematurely at time t = " << fluid.t << endl;
        Kokkos::finalize();
        return -1;
    }
    Kokkos::finalize();
    return 0;
}