#ifndef FLUID_KOKKOS_HPP
#define FLUID_KOKKOS_HPP
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

using namespace std;

namespace fluid_kokkos {

    enum fluidState {undefined, allocated, initialized, assembled};
    //enum device {Kokkos::Cuda, Kokkos::OpenMP, Kokkos::Serial};

    using Kmat = Kokkos::DualView<double**,Kokkos::LayoutRight>;
    using Kdouble = Kokkos::DualView<double>;


class Fluid {
    protected:
        fluidState _state = undefined;

    public: 
        Kmat vx;
        Kmat vy;
        Kmat pressure;
        Kmat density;
        double totalEnergy;
        double totalMass;
        double totalMomentumX;
        double totalMomentumY;
        double t;
        double tFinal;
        double tOut;
        int useSlopeLimiter = 0;
        // fluid params
        double courant_fac = 0.4;
        double w0 = 0.1;
        double sigma = 0.05/sqrt(2.);
        double gamma = 5./3.;
        int Nx;
        int Ny;
        double BoxSizeX;
        double BoxSizeY;
        double dx;
        double dy;
        
    private:
        
        double cellVol;

        Kmat isNotBoundary;

        Kmat  rho_XL, rho_XR, rho_YB, rho_YT;
        Kmat  vx_XL, vx_XR, vx_YB, vx_YT;
        Kmat  vy_XL, vy_XR, vy_YB, vy_YT;
        Kmat  P_XL, P_XR, P_YB, P_YT;

        Kmat flux_rho_X, flux_rho_Y;
        Kmat flux_momx_X, flux_momx_Y;
        Kmat flux_momy_X, flux_momy_Y;
        Kmat flux_E_X, flux_E_Y;
    
    // function declarations
    public:
        Fluid(int Nx, int Ny, double boxSizeX, double boxSizeY);
        ~Fluid();
        void initialize();
        void runSimulation(double tFinal, double tOut);
        void runTimeStep();
        double calculateTimeStep();

    private:
        void extrapolateToFaces(double dt);
        void updateStates(double dt);
        void RiemannSolver();
        void slopeLimiter(double &gradx, double &grady, const Kokkos::View<double**, Kokkos::LayoutRight> field,
                            int i, int j, int Ri, int Li, int Ti, int Bi);
        void printState(int step);
};
}

#endif // FLUID_HPP