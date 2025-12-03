#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <time.h>

#include "fluid.hpp"
#include "SnapshotWriter.hpp"


using namespace fluid;

void timeStepBenchmark(int Nx, int Ny, int useSlopeLimiter){
    // Placeholder for timestep benchmarking function
    double boxSizeX = 1.0;
    double boxSizeY = 1.0;
    double tFinal = 1.0; //2.0;
    double tOut = 0.01;

    struct timespec start, end;

    Fluid fluid(Nx, Ny, boxSizeX, boxSizeY);
    fluid.useSlopeLimiter = useSlopeLimiter; // enable slope limiter
    fluid.initialize();

    // set null conditions for benchmarking
    for (int i = 0; i < Nx; i++){
        for (int j = 0; j < Ny; j++){
            fluid.density[i][j] = 1.0;
            fluid.pressure[i][j] = 2.5;
            fluid.vx[i][j] = 0.5;
            fluid.vy[i][j] = 0.0;
        }
    }

    // time execution for 10 timesteps
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int j = 0; j < 10; j++){
        fluid.runTimeStep();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);   

    double time_taken = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    cout << "Avg time per step for Nx = " << Nx << ", Ny = " << Ny << " : " 
            << time_taken / 10.0 << " seconds." << endl;

    //fluid.~Fluid();

};


int main(){
    // Simulation parameters

    for (int power = 6; power <= 11; power++){; 

        int Nx = pow(2, power); //1024;
        int Ny = pow(2, power); //1024;
        

        cout << "Starting Fluid Solver Benchmarking with Nx = " << Nx << ", Ny = " << Ny << endl;
        timeStepBenchmark(Nx, Ny, 0); // without slope limiter
        //timeStepBenchmark(Nx, Ny, 1); // with slope limiter
    }

    return 0;
}
