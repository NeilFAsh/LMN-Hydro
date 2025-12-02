#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>

#include "fluid.hpp"
#include "SnapshotWriter.hpp"

#include "fluid_KHI.hpp"

using namespace fluid;


fluid_KHI::initialize_KHI(){
    // Initialize a Kelvin-Helmholtz Instability setup
    initialize();

    for(int i = 0; i < Nx; i++){
        for(int j = 0; j < Ny; j++){
            double y = (j + 0.5) * dy;
            double x = (i + 0.5) * dx;
            if(y < BoxSizeY * 0.75 && y > BoxSizeY * 0.25){
                vx[i][j] = 0.5;
                density[i][j] = 1.0;
            } else {
                vx[i][j] = -0.5;
                density[i][j] = 2.0;
            }
            // Add perturbation
            vy[i][j] = w0 * sin(4 * M_PI * x / BoxSizeX) * ( exp(-0.5*pow((y - BoxSizeY / 4) / sigma, 2))
                                                + exp(-0.5*pow((y - 3 * BoxSizeY / 4) / sigma, 2)) );
            pressure[i][j] = 2.5; // uniform pressure
        }
    }
};

int main(){
    // Simulation parameters
    int Nx = 100;
    int Ny = 100;
    double boxSizeX = 1.0;
    double boxSizeY = 1.0;
    double tFinal = 1.0; //2.0;
    double tOut = 0.01;

    cout << "Starting Kelvin-Helmholtz Instability Simulation" << endl;
    Fluid_KHI fluid(Nx, Ny, boxSizeX, boxSizeY);
    fluid.initialize_KHI();

    try {
        fluid.runSimulation(tFinal,tOut);
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
        cerr << "Simulation terminated prematurely at time t = " << fluid.t << endl;
        return -1;
    }
    return 0;

    return 0;
}