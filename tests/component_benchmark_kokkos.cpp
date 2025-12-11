#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <time.h>

#include "fluid_kokkos.hpp"
#include "SnapshotWriter.hpp"


using namespace fluid_kokkos;

void componentBenchmark(int Nx, int Ny, int useSlopeLimiter){
    // Placeholder for timestep benchmarking function
    double boxSizeX = 1.0;
    double boxSizeY = 1.0;
    double tFinal = 1.0; //2.0;
    double tOut = 0.01;

    struct timespec start, end;

    Fluid fluid(Nx, Ny, boxSizeX, boxSizeY);
    fluid.useSlopeLimiter = useSlopeLimiter; // enable slope limiter
    fluid.initialize();
    fluid.assemble();

    // retrieve view objects on device
    auto vx = fluid.vx.view<Kokkos::DefaultExecutionSpace>();
    auto vy = fluid.vy.view<Kokkos::DefaultExecutionSpace>();
    auto density = fluid.density.view<Kokkos::DefaultExecutionSpace>();
    auto pressure = fluid.pressure.view<Kokkos::DefaultExecutionSpace>();

    // set null conditions for benchmarking
    Kokkos::parallel_for("initialize_benchmark_conditions",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j) {
            density(i,j) = 1.0;
            pressure(i,j) = 2.5;
            vx(i,j) = 0.5;
            vy(i,j) = 0.0;
        }
    );
    Kokkos::fence();

    // mark as modified on device
    fluid.density.modify<Kokkos::DefaultExecutionSpace>();
    fluid.vx.modify<Kokkos::DefaultExecutionSpace>();
    fluid.vy.modify<Kokkos::DefaultExecutionSpace>();
    fluid.pressure.modify<Kokkos::DefaultExecutionSpace>();

    // synce device to host 
    {
        fluid.density.sync<Kokkos::DefaultExecutionSpace>();
        fluid.vx.sync<Kokkos::DefaultExecutionSpace>();
        fluid.vy.sync<Kokkos::DefaultExecutionSpace>();
        fluid.pressure.sync<Kokkos::DefaultExecutionSpace>();
    }
    
    // time dt calc for 10 timesteps
    double dt;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int j = 0; j < 10; j++){
        dt = fluid.calculateTimeStep();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);   

    double calc_dt_time = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;

    // time extrapolateToFaces
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int j = 0; j < 10; j++){
        fluid.extrapolateToFaces(dt);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);   

    double extrapolate_time = (end.tv_sec - start.tv_sec) + 
                                (end.tv_nsec - start.tv_nsec) / 1e9;

    // time Riemann solver
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int j = 0; j < 10; j++){
        fluid.RiemannSolver();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);   

    double riemann_time = (end.tv_sec - start.tv_sec) + 
                                (end.tv_nsec - start.tv_nsec) / 1e9;

    // time updateStates
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int j = 0; j < 10; j++){
        fluid.updateStates(dt);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);   

    double update_time = (end.tv_sec - start.tv_sec) + 
                                (end.tv_nsec - start.tv_nsec) / 1e9;

    calc_dt_time /= 10;
    extrapolate_time /= 10;
    riemann_time /= 10; 
    update_time /= 10;

    cout << "Avg time for Nx = " << Nx << ", Ny = " << Ny << " : "  << endl;
    cout << "dt: " << calc_dt_time << ", extrapolation: " << extrapolate_time << ", riemann solver: " << riemann_time
            << ", update states: " << update_time << endl;
    cout << endl;


};


int main(){
    // Simulation parameters

    Kokkos::initialize();

    for (int power = 6; power <= 11; power++){; 
        int Nx = pow(2, power); //1024;
        int Ny = pow(2, power); //1024;
        
        cout << "Starting Fluid Solver Benchmarking with Nx = " << Nx << ", Ny = " << Ny << endl;
        componentBenchmark(Nx, Ny, 0); // without slope limiter
        //timeStepBenchmark(Nx, Ny, 1); // with slope limiter
    }

    Kokkos::finalize();
    
    return 0;
}
