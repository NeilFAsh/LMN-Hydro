#ifndef FLUID_HPP
#define FLUID_HPP
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

namespace fluid {

    enum fluidState {undefined, allocated, initialized, assembled};


class Fluid {
    protected:
        fluidState _state = undefined;

    public: 
        vector<vector<double>> vx;
        vector<vector<double>> vy;
        vector<vector<double>> pressure;
        vector<vector<double>> density;
        double totalEnergy;
        double totalMass;
        double totalMomentumX;
        double totalMomentumY;
        float t;
        float tFinal;
        float tOut;
        int useSlopeLimiter = 0;
        // fluid params
        float courant_fac = 0.4;
        float w0 = 0.1;
        float sigma = 0.05/sqrt(2.);
        float gamma = 5./3.;
        int Nx;
        int Ny;
        double BoxSizeX;
        double BoxSizeY;
        double dx;
        double dy;
        
    private:
        
        double cellVol;

        vector<vector<double>> isNotBoundary;

        vector<vector<double>>  rho_XL, rho_XR, rho_YB, rho_YT;
        vector<vector<double>>  vx_XL, vx_XR, vx_YB, vx_YT;
        vector<vector<double>>  vy_XL, vy_XR, vy_YB, vy_YT;
        vector<vector<double>>  P_XL, P_XR, P_YB, P_YT;

        vector<vector<double>> flux_rho_X, flux_rho_Y;
        vector<vector<double>> flux_momx_X, flux_momx_Y;
        vector<vector<double>> flux_momy_X, flux_momy_Y;
        vector<vector<double>> flux_E_X, flux_E_Y;
    
    // function declarations
    public:
        Fluid(int Nx, int Ny, double boxSizeX, double boxSizeY);
        ~Fluid();
        void initialize();
        void assemble();
        void runSimulation(double tFinal, double tOut);
        void runTimeStep();
        double calculateTimeStep();
        void extrapolateToFaces(double dt);
        void updateStates(double dt);
        void RiemannSolver();
        void slopeLimiter(double &gradx, double &grady, vector<vector<double>> field,
                            int i, int j, int Ri, int Li, int Ti, int Bi);
        void printState(int step);
};
}

#endif // FLUID_HPP