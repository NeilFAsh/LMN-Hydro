#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <time.h>

#include "fluid_kokkos.hpp"
#include "SnapshotWriter.hpp"
#include <Kokkos_Core.hpp>


using namespace fluid_kokkos;


// Constructor implementation
Fluid::Fluid(int Nx, int Ny, double boxSizeX, double boxSizeY){

    assert(this->_state == undefined);

    assert(Kokkos::is_initialized());

    // Set up grid params and send to device

    this->Nx = Nx; 
    this->Ny = Ny; 

    this->BoxSizeX = boxSizeX;
    this->BoxSizeY = boxSizeY;

    this->dx = boxSizeX / Nx;
    this->dy = boxSizeY / Ny;
    this->cellVol = dx * dy;

    // allocate memory for the fluid properties

    Kmat vx("vx", this->Nx, this->Ny); this->vx = vx;
    Kmat vy("vy", this->Nx, this->Ny); this->vy = vy;
    Kmat pressure("pressure", this->Nx, this->Ny); this->pressure = pressure;
    Kmat density("density", this->Nx, this->Ny); this->density = density;

    Kmat rho_XL("rho_XL", this->Nx, this->Ny); this->rho_XL = rho_XL;
    Kmat rho_XR("rho_XR", this->Nx, this->Ny); this->rho_XR = rho_XR;
    Kmat rho_YB("rho_YB", this->Nx, this->Ny); this->rho_YB = rho_YB;
    Kmat rho_YT("rho_YT", this->Nx, this->Ny); this->rho_YT = rho_YT;

    Kmat vx_XL("vx_XL", this->Nx, this->Ny); this->vx_XL = vx_XL;
    Kmat vx_XR("vx_XR", this->Nx, this->Ny); this->vx_XR = vx_XR;
    Kmat vx_YB("vx_YB", this->Nx, this->Ny); this->vx_YB = vx_YB;
    Kmat vx_YT("vx_TY", this->Nx, this->Ny); this->vx_YT = vx_YT;

    Kmat vy_XL("vy_XL", this->Nx, this->Ny); this->vy_XL = vy_XL;
    Kmat vy_XR("vy_XR", this->Nx, this->Ny); this->vy_XR = vy_XR;
    Kmat vy_YB("vy_YB", this->Nx, this->Ny); this->vy_YB = vy_YB;
    Kmat vy_YT("vy_YT", this->Nx, this->Ny); this->vy_YT = vy_YT;
    
    Kmat P_XL("P_XL", this->Nx, this->Ny); this->P_XL = P_XL;
    Kmat P_XR("P_XR", this->Nx, this->Ny); this->P_XR = P_XR;
    Kmat P_YB("P_YB", this->Nx, this->Ny); this->P_YB = P_YB;
    Kmat P_YT("P_YT", this->Nx, this->Ny); this->P_YT = P_YT;
    
    Kmat flux_rho_X("flux_rho_X", this->Nx, this->Ny); this->flux_rho_X = flux_rho_X;
    Kmat flux_rho_Y("flux_rho_Y", this->Nx, this->Ny); this->flux_rho_Y = flux_rho_Y;
    Kmat flux_momx_X("flux_momx_X", this->Nx, this->Ny); this->flux_momx_X = flux_momx_X;
    Kmat flux_momx_Y("flux_momx_Y", this->Nx, this->Ny); this->flux_momx_Y = flux_momx_Y;
    Kmat flux_momy_X("flux_momy_X", this->Nx, this->Ny); this->flux_momy_X = flux_momy_X;
    Kmat flux_momy_Y("flux_momy_Y", this->Nx, this->Ny); this->flux_momy_Y = flux_momy_Y;
    Kmat flux_E_X("flux_E_X", this->Nx, this->Ny); this->flux_E_X = flux_E_X;
    Kmat flux_E_Y("flux_E_Y", this->Nx, this->Ny); this->flux_E_Y = flux_E_Y;
    
    this->_state = allocated;
};

Fluid::~Fluid(){
    // Destructor
    this->_state = undefined;
};

void Fluid::initialize(){

    assert(this->_state == allocated);
    this->_state = initialized;

};

void Fluid::assemble(){

    assert(this->_state == assembled || this->_state == initialized);
    this->_state = assembled;

}
        
void Fluid::printState(int step){

    assert(this->_state == initialized || this->_state == assembled);

    totalEnergy = 0.0;
    totalMass = 0.0;
    totalMomentumX = 0.0;
    totalMomentumY = 0.0;

    // retrieve view objects on device
    auto vx = this->vx.view<Kokkos::DefaultExecutionSpace>();
    auto vy = this->vy.view<Kokkos::DefaultExecutionSpace>();
    auto pressure = this->pressure.view<Kokkos::DefaultExecutionSpace>();
    auto density = this->density.view<Kokkos::DefaultExecutionSpace>();

    // make local copies of the class attributes here
    // do this because of issues with implicit capture in Kokkos
    double gamma = this->gamma;
    double cellVol = this->cellVol;

    Kokkos::parallel_reduce("compute_totals",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j, double& local_totalEnergy,
                                        double& local_totalMass,
                                        double& local_totalMomentumX,
                                        double& local_totalMomentumY) {
            local_totalEnergy += (pressure(i,j)/(gamma - 1) + 0.5 * density(i,j) * 
                            (vx(i,j)*vx(i,j) + vy(i,j)*vy(i,j))) * cellVol;
            local_totalMass += density(i,j) * cellVol;
            local_totalMomentumX += density(i,j) * vx(i,j) * cellVol;
            local_totalMomentumY += density(i,j) * vy(i,j) * cellVol;
        }, 
        Kokkos::Sum<double>(totalEnergy),
        Kokkos::Sum<double>(totalMass),
        Kokkos::Sum<double>(totalMomentumX),
        Kokkos::Sum<double>(totalMomentumY)
    );
    
    cout << "t = " << t << ", E = " << totalEnergy << ", Mass = " << totalMass 
                << ", Px = " << totalMomentumX << ", Py = " << totalMomentumY << endl;
};

void Fluid::runSimulation(double tFinal, double tOut){

    assert(this->_state == initialized);
    this->_state = assembled;

    assert(Kokkos::is_initialized());
    
    SnapshotWriter writer("output");   // creates "output/" folder

    t = 0.0;
    this->tFinal = tFinal;
    this->tOut = tOut;

    int numOutputs = 0;
    // Ensure device has up-to-date data before starting the time loop
    this->density.sync<Kokkos::DefaultExecutionSpace>();
    this->vx.sync<Kokkos::DefaultExecutionSpace>();
    this->vy.sync<Kokkos::DefaultExecutionSpace>();
    this->pressure.sync<Kokkos::DefaultExecutionSpace>();
    while (t < tFinal){

        if (t >= numOutputs * tOut){
            printState(numOutputs);

            // sync device to host for output
            vector<vector<double>> vec_rho, vec_vx, vec_vy, vec_P;
            
            vec_rho = vector<vector<double>>(this->Nx, vector<double>(this->Ny, 0.0));
            vec_vx = vector<vector<double>>(this->Nx, vector<double>(this->Ny, 0.0));
            vec_vy = vector<vector<double>>(this->Nx, vector<double>(this->Ny, 0.0));
            vec_P = vector<vector<double>>(this->Nx, vector<double>(this->Ny, 0.0));

            // sync and retrive host view
            this->density.sync<Kokkos::HostSpace>();
            this->vx.sync<Kokkos::HostSpace>();
            this->vy.sync<Kokkos::HostSpace>();
            this->pressure.sync<Kokkos::HostSpace>();

            auto density = this->density.view<Kokkos::HostSpace>();
            auto vx = this->vx.view<Kokkos::HostSpace>();
            auto vy = this->vy.view<Kokkos::HostSpace>();
            auto pressure = this->pressure.view<Kokkos::HostSpace>();
            
            for (int i = 0; i < Nx; i++){
                for (int j = 0; j < Ny; j++){
                    vec_rho[i][j] = density(i,j);
                    vec_vx[i][j] = vx(i,j);
                    vec_vy[i][j] = vy(i,j);
                    vec_P[i][j] = pressure(i,j);
                }
            }
            
            // Format filename with leading zeros
            std::ostringstream oss;
            oss << std::setw(4) << std::setfill('0') << numOutputs; // "0000"
            std::string filenum = oss.str();
            // save
            writer.save_npz_snapshot("snapshot_" + filenum + ".npz", 
                                        vec_rho, vec_vx, vec_vy, vec_P, Nx, Ny,
                                        totalMass, totalEnergy, totalMomentumX, totalMomentumY);
            
            numOutputs++;

            // After creating host-side snapshot, ensure device views are up-to-date
            // before continuing with device kernels.
            this->density.sync<Kokkos::DefaultExecutionSpace>();
            this->vx.sync<Kokkos::DefaultExecutionSpace>();
            this->vy.sync<Kokkos::DefaultExecutionSpace>();
            this->pressure.sync<Kokkos::DefaultExecutionSpace>();
        }

        runTimeStep();
    }
};
        
void Fluid::runTimeStep(){

    assert(this->_state == assembled || this->_state == initialized);
    this->_state = assembled;
    //cout << "calculating dt" << endl;
    double dt = calculateTimeStep();
    //cout << "dt = " << dt << endl;
    //cout << "extrapolate to faces" << endl;
    extrapolateToFaces(dt);
    //cout << "riemann solver" << endl;
    RiemannSolver();
    //cout << "update states" << endl;
    updateStates(dt);
    //cout << "Finished timestep, current dt =  " << dt << endl;
    t += dt;
};

void Fluid::updateStates(double dt){

    assert(this->_state == assembled);

    // retrieve view objects on device
    auto density = this->density.view<Kokkos::DefaultExecutionSpace>();
    auto vx = this->vx.view<Kokkos::DefaultExecutionSpace>();
    auto vy = this->vy.view<Kokkos::DefaultExecutionSpace>();
    auto pressure = this->pressure.view<Kokkos::DefaultExecutionSpace>();

    auto flux_rho_X = this->flux_rho_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_rho_Y = this->flux_rho_Y.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momx_X = this->flux_momx_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momx_Y = this->flux_momx_Y.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momy_X = this->flux_momy_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momy_Y = this->flux_momy_Y.view<Kokkos::DefaultExecutionSpace>();
    auto flux_E_X = this->flux_E_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_E_Y = this->flux_E_Y.view<Kokkos::DefaultExecutionSpace>();

    // update cell-centered states using computed fluxes

    // Mark DualViews as modified on the device since we'll write to their device views.
    this->density.modify<Kokkos::DefaultExecutionSpace>();
    this->vx.modify<Kokkos::DefaultExecutionSpace>();
    this->vy.modify<Kokkos::DefaultExecutionSpace>();
    this->pressure.modify<Kokkos::DefaultExecutionSpace>();

    // Make local copies of class attributes used by Kokkos
    double cellVol = this->cellVol;
    double gamma = this->gamma;
    double dx = this->dx;
    double dy = this->dy;
    int Nx = this->Nx;
    int Ny = this->Ny;

    Kokkos::parallel_for("update_states",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j) {

            // Periodic boundary conditions
            int Li = (i - 1 + Nx) % Nx; //left 
            int Bi = (j - 1 + Ny) % Ny; // bottom

            // Get conserved quantities
            double mass = density(i,j) * cellVol;
            double momx = mass * vx(i,j);
            double momy = mass * vy(i,j);
            double E = cellVol * pressure(i,j)/(gamma - 1) + 0.5 * mass * 
                            (vx(i,j)*vx(i,j) + vy(i,j)*vy(i,j));

            // update density
            // subtract fluxes leaving the cell (right and top)
            // and add fluxes entering the cell (left and bottom)
            mass -= dt * dy * (flux_rho_X(i,j) - flux_rho_X(Li,j))
                                + dt * dx * (flux_rho_Y(i,j) - flux_rho_Y(i,Bi));

            // update momentum in x
            momx -= dt * dy * (flux_momx_X(i,j) - flux_momx_X(Li,j))
                                + dt * dx * (flux_momx_Y(i,j) - flux_momx_Y(i,Bi));

            // update momentum in y
            momy -= dt * dy * (flux_momy_X(i,j) - flux_momy_X(Li,j))
                                + dt * dx * (flux_momy_Y(i,j) - flux_momy_Y(i,Bi));

            // update energy
            E -= dt * dy * (flux_E_X(i,j) - flux_E_X(Li,j))
                        + dt * dx * (flux_E_Y(i,j) - flux_E_Y(i,Bi));

            double e_cell = E / cellVol;
            double ke = 0.5 * mass / cellVol * 
                            ( (momx / mass)*(momx / mass) + (momy / mass)*(momy / mass) );
            double e_internal = e_cell - ke;

            // Note: this causes non-conservation of energy
            //if (e_internal < 1e-12) e_internal = 1e-12;
            // update primitive variables
            density(i,j) = mass / cellVol;
            vx(i,j) = momx / mass;
            vy(i,j) = momy / mass;
            pressure(i,j) = (gamma - 1.0) * e_internal;
        }
    );

    Kokkos::fence();
    // Compute minimum density and pressure on device to detect non-physical states.
    //cout << "beginning parallel reduce" << endl;
    double minDensity = 1e300;
    double minPressure = 1e300;
    Kokkos::parallel_reduce("check_physical_states",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j, double& local_minDensity, double& local_minPressure) {
            local_minDensity = fmin(local_minDensity, density(i,j));
            local_minPressure = fmin(local_minPressure, pressure(i,j));
        },
        Kokkos::Min<double>(minDensity),
        Kokkos::Min<double>(minPressure)
    );
    // If the above fails just use serial implementation:
    // for (int i = 0; i < Nx; i++){
    //     for (int j = 0; j < Ny; j++){
    //         if (density(i,j) <= 0.0) {
    //             cout << "Found density with value " << density(i,j) << " at (" << i << ", " << j << ")\n";
    //             throwException = 1;
    //         };
    //         if (pressure(i,j) <= 0.0) {
    //             cout << "Found pressure with value " << pressure(i,j) << " at (" << i << ", " << j << ")\n";
    //             throwException = 1;
    //         };
    //     }
    // }

    Kokkos::fence();
    //cout << "ended parallel for" << endl;
    if (minDensity <= 0.0 || minPressure <= 0.0) {
        // Sync to host and print offending cell values for debugging
        cout << "syncing to host b/c problem" << endl;
        this->density.sync<Kokkos::HostSpace>();
        this->pressure.sync<Kokkos::HostSpace>();
        auto density_h = this->density.view<Kokkos::HostSpace>();
        auto pressure_h = this->pressure.view<Kokkos::HostSpace>();
        bool found = false;
        for (int ii = 0; ii < Nx && !found; ++ii){
            for (int jj = 0; jj < Ny && !found; ++jj){
                if (density_h(ii,jj) <= 0.0) {
                    std::cerr << "Found density with value " << density_h(ii,jj) << " at (" << ii << ", " << jj << ")\n";
                    found = true;
                    break;
                }
                if (pressure_h(ii,jj) <= 0.0) {
                    std::cerr << "Found pressure with value " << pressure_h(ii,jj) << " at (" << ii << ", " << jj << ")\n";
                    found = true;
                    break;
                }
            }
        }
        throw runtime_error("Simulation encountered non-physical state (negative or zero density or pressure).");
    }
    //cout << "Finished updateState" << endl;
};  

void Fluid::RiemannSolver() {

    assert(this->_state == assembled);

    // retrieve view objects on device
    auto rho_XL = this->rho_XL.view<Kokkos::DefaultExecutionSpace>();
    auto rho_XR = this->rho_XR.view<Kokkos::DefaultExecutionSpace>();
    auto rho_YB = this->rho_YB.view<Kokkos::DefaultExecutionSpace>();
    auto rho_YT = this->rho_YT.view<Kokkos::DefaultExecutionSpace>();

    auto vx_XL = this->vx_XL.view<Kokkos::DefaultExecutionSpace>();
    auto vx_XR = this->vx_XR.view<Kokkos::DefaultExecutionSpace>();
    auto vx_YB = this->vx_YB.view<Kokkos::DefaultExecutionSpace>();
    auto vx_YT = this->vx_YT.view<Kokkos::DefaultExecutionSpace>();

    auto vy_XL = this->vy_XL.view<Kokkos::DefaultExecutionSpace>();
    auto vy_XR = this->vy_XR.view<Kokkos::DefaultExecutionSpace>();
    auto vy_YB = this->vy_YB.view<Kokkos::DefaultExecutionSpace>();
    auto vy_YT = this->vy_YT.view<Kokkos::DefaultExecutionSpace>();

    auto P_XL = this->P_XL.view<Kokkos::DefaultExecutionSpace>();
    auto P_XR = this->P_XR.view<Kokkos::DefaultExecutionSpace>();
    auto P_YB = this->P_YB.view<Kokkos::DefaultExecutionSpace>();
    auto P_YT = this->P_YT.view<Kokkos::DefaultExecutionSpace>();

    auto flux_rho_X = this->flux_rho_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_rho_Y = this->flux_rho_Y.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momx_X = this->flux_momx_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momx_Y = this->flux_momx_Y.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momy_X = this->flux_momy_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_momy_Y = this->flux_momy_Y.view<Kokkos::DefaultExecutionSpace>();
    auto flux_E_X = this->flux_E_X.view<Kokkos::DefaultExecutionSpace>();
    auto flux_E_Y = this->flux_E_Y.view<Kokkos::DefaultExecutionSpace>();

    // Make local copies of class attributes used by Kokkos
    double gamma = this->gamma;
    int Nx = this->Nx;
    int Ny = this->Ny;

    // Solve Riemann problems at each right/top face of cell (i,j).
    // We'll assemble conserved left and right states from the face-extrapolated primitives
    // and apply the Rusanov flux formula.
    Kokkos::parallel_for("Riemann_solver",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j) {
    
            // neighbor indices (periodic)
            const int Ri = (i + 1) % Nx;
            const int Li = (i - 1 + Nx) % Nx;
            const int Ti = (j + 1) % Ny;
            const int Bi = (j - 1 + Ny) % Ny;

            //
            // --- X-face between cell i and Ri (right face of cell i)
            // left state = right-extrapolation of cell i  -> (rho_XR[i][j], vx_XR[i][j], vy_XR[i][j], P_XR[i][j])
            // right state = left-extrapolation of cell Ri -> (rho_XL[Ri][j], vx_XL[Ri][j], vy_XL[Ri][j], P_XL[Ri][j])
            //
            {
                // primitives L (cell i right face)
                double rhoL = rho_XR(i,j);
                double uL = vx_XR(i,j);
                double vL = vy_XR(i,j);
                double pL = P_XR(i,j);

                // primitives R (cell Ri left face)
                double rhoR = rho_XL(Ri,j);
                double uR = vx_XL(Ri,j);
                double vR = vy_XL(Ri,j);
                double pR = P_XL(Ri,j);

                // conserved variables U = [rho, rho*u, rho*v, E]
                double UL_rho = rhoL;
                double UL_mx = rhoL * uL;
                double UL_my = rhoL * vL;
                double UL_E  = pL / (gamma - 1.0) + 0.5 * rhoL * (uL*uL + vL*vL);

                double UR_rho = rhoR;
                double UR_mx = rhoR * uR;
                double UR_my = rhoR * vR;
                double UR_E  = pR / (gamma - 1.0) + 0.5 * rhoR * (uR*uR + vR*vR);

                // physical flux in x-direction: F(U) = [rho*u, rho*u^2 + p, rho*u*v, (E + p)*u]
                double FL_rho = UL_mx;
                double FL_mx  = UL_mx * uL + pL;
                double FL_my  = UL_my * uL;
                double FL_E   = (UL_E + pL) * uL;

                double FR_rho = UR_mx;
                double FR_mx  = UR_mx * uR + pR;
                double FR_my  = UR_my * uR;
                double FR_E   = (UR_E + pR) * uR;

                // compute local max signal speed in x (Rusanov): s = max(|u| + c) over L,R
                double cL = sqrt(max(0.0, gamma * pL / rhoL));
                double cR = sqrt(max(0.0, gamma * pR / rhoR));
                double smax = std::max(fabs(uL) + cL, fabs(uR) + cR);

                // Rusanov flux: 0.5*(F_L + F_R) - 0.5 * smax * (U_R - U_L)
                flux_rho_X(i,j) = 0.5 * (FL_rho + FR_rho) - 0.5 * smax * (UR_rho - UL_rho);
                flux_momx_X(i,j) = 0.5 * (FL_mx + FR_mx) - 0.5 * smax * (UR_mx - UL_mx);
                flux_momy_X(i,j) = 0.5 * (FL_my + FR_my) - 0.5 * smax * (UR_my - UL_my);
                flux_E_X(i,j)    = 0.5 * (FL_E + FR_E) - 0.5 * smax * (UR_E - UL_E);
            }

            //
            // --- Y-face between cell j and Ti (top face of cell (i,j))
            // bottom state = top-extrapolation of cell j -> (rho_YT[i][j] is top of cell (i,j)?)
            // careful indexing: we'll define:
            // left/bottom = cell (i,j) top face is YT[i][j] (that's the top value in cell i,j)
            // right/top = cell (i,Ti) bottom face is YB[i][Ti]
            //
            {
                // primitives L (bottom side of top-face) = YT of (i,j) is the state just below the face
                // but consistent with your notation: we treat "left" as the lower side (cell (i,j) top extrapolation)
                double rhoL = rho_YT(i,j);
                double uL = vx_YT(i,j);
                double vL = vy_YT(i,j);
                double pL = P_YT(i,j);

                // primitives R (upper side of top-face) = YB of (i,Ti) is the state just above the face
                double rhoR = rho_YB(i,Ti);
                double uR = vx_YB(i,Ti);
                double vR = vy_YB(i,Ti);
                double pR = P_YB(i,Ti);

                // conserved variables
                double UL_rho = rhoL;
                double UL_mx  = rhoL * uL;
                double UL_my  = rhoL * vL;
                double UL_E   = pL / (gamma - 1.0) + 0.5 * rhoL * (uL*uL + vL*vL);

                double UR_rho = rhoR;
                double UR_mx  = rhoR * uR;
                double UR_my  = rhoR * vR;
                double UR_E   = pR / (gamma - 1.0) + 0.5 * rhoR * (uR*uR + vR*vR);

                // physical flux in y-direction: G(U) = [rho*v, rho*u*v, rho*v^2 + p, (E + p)*v]
                double GL_rho = UL_my;
                double GL_mx  = UL_mx * vL;
                double GL_my  = UL_my * vL + pL;
                double GL_E   = (UL_E + pL) * vL;

                double GR_rho = UR_my;
                double GR_mx  = UR_mx * vR;
                double GR_my  = UR_my * vR + pR;
                double GR_E   = (UR_E + pR) * vR;

                // compute local max signal speed in y
                double cL = sqrt(max(0.0, gamma * pL / rhoL));
                double cR = sqrt(max(0.0, gamma * pR / rhoR));
                double smax = std::max(fabs(vL) + cL, fabs(vR) + cR);

                // Rusanov flux in y-direction
                flux_rho_Y(i,j)  = 0.5 * (GL_rho + GR_rho) - 0.5 * smax * (UR_rho - UL_rho);
                flux_momx_Y(i,j) = 0.5 * (GL_mx + GR_mx)   - 0.5 * smax * (UR_mx  - UL_mx);
                flux_momy_Y(i,j) = 0.5 * (GL_my + GR_my)   - 0.5 * smax * (UR_my  - UL_my);
                flux_E_Y(i,j)    = 0.5 * (GL_E + GR_E)     - 0.5 * smax * (UR_E   - UL_E);
            }
        }
    );
    Kokkos::fence();
}

void Fluid::extrapolateToFaces(double dt){

    assert(this->_state == assembled);

    // retrieve view objects on device
    auto vx = this->vx.view<Kokkos::DefaultExecutionSpace>();
    auto vy = this->vy.view<Kokkos::DefaultExecutionSpace>();
    auto pressure = this->pressure.view<Kokkos::DefaultExecutionSpace>();
    auto density = this->density.view<Kokkos::DefaultExecutionSpace>();

    auto rho_XL = this->rho_XL.view<Kokkos::DefaultExecutionSpace>();
    auto rho_XR = this->rho_XR.view<Kokkos::DefaultExecutionSpace>();
    auto rho_YB = this->rho_YB.view<Kokkos::DefaultExecutionSpace>();
    auto rho_YT = this->rho_YT.view<Kokkos::DefaultExecutionSpace>();

    auto vx_XL = this->vx_XL.view<Kokkos::DefaultExecutionSpace>();
    auto vx_XR = this->vx_XR.view<Kokkos::DefaultExecutionSpace>();
    auto vx_YB = this->vx_YB.view<Kokkos::DefaultExecutionSpace>();
    auto vx_YT = this->vx_YT.view<Kokkos::DefaultExecutionSpace>();

    auto vy_XL = this->vy_XL.view<Kokkos::DefaultExecutionSpace>();
    auto vy_XR = this->vy_XR.view<Kokkos::DefaultExecutionSpace>();
    auto vy_YB = this->vy_YB.view<Kokkos::DefaultExecutionSpace>();
    auto vy_YT = this->vy_YT.view<Kokkos::DefaultExecutionSpace>();

    auto P_XL = this->P_XL.view<Kokkos::DefaultExecutionSpace>();
    auto P_XR = this->P_XR.view<Kokkos::DefaultExecutionSpace>();
    auto P_YB = this->P_YB.view<Kokkos::DefaultExecutionSpace>();
    auto P_YT = this->P_YT.view<Kokkos::DefaultExecutionSpace>();

    // Make local copies of class attributes used by Kokkos
    double gamma = this->gamma;
    double dx = this->dx;
    double dy = this->dy;
    int Nx = this->Nx;
    int Ny = this->Ny;

    // extrapolate cell-centered values to faces
    Kokkos::parallel_for("extrapolate_to_faces",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j) {
 
            // Periodic boundary conditions
            const int Ri = (i + 1) % Nx; // right
            const int Li = (i - 1 + Nx) % Nx; //left 

            const int Ti = (j + 1) % Ny; // top
            const int Bi = (j - 1 + Ny) % Ny; // bottom

            // compute gradients using central differences
            double gradX_Density = (density(Ri,j) - density(Li,j)) / (2.0*dx);
            double gradY_Density = (density(i,Ti) - density(i,Bi)) / (2.0*dy);

            double gradX_Pressure = (pressure(Ri,j) - pressure(Li,j)) / (2.0*dx);
            double gradY_Pressure = (pressure(i,Ti) - pressure(i,Bi)) / (2.0*dy);

            double gradX_Vx = (vx(Ri,j) - vx(Li,j)) / (2.0*dx);
            double gradY_Vx = (vx(i,Ti) - vx(i,Bi)) / (2.0*dy);
            double gradX_Vy = (vy(Ri,j) - vy(Li,j)) / (2.0*dx);
            double gradY_Vy = (vy(i,Ti) - vy(i,Bi)) / (2.0*dy);

            // apply slope limiter if enabled
            // if(useSlopeLimiter){
            //     slopeLimiter(gradX_Density, gradY_Density, density,
            //                     i, j, Ri, Li, Ti, Bi);
            //     slopeLimiter(gradX_Pressure, gradY_Pressure, pressure,
            //                     i, j, Ri, Li, Ti, Bi);
            //     slopeLimiter(gradX_Vx, gradY_Vx, vx,
            //                     i, j, Ri, Li, Ti, Bi);
            //     slopeLimiter(gradX_Vy, gradY_Vy, vy,
            //                     i, j, Ri, Li, Ti, Bi);
            // };

            // extrapolate cell-centered values to faces
            double rho_prime = density(i,j) - 0.5 * dt * (gradX_Density * vx(i,j) + gradY_Density * vy(i,j) +
                                                        density(i,j) * (gradX_Vx + gradY_Vy));
            double vx_prime = vx(i,j) - 0.5 * dt * (gradX_Vx * vx(i,j) + gradY_Vx * vy(i,j) +
                                                (1.0 / density(i,j)) * gradX_Pressure);
            double vy_prime = vy(i,j) - 0.5 * dt * (gradX_Vy * vx(i,j) + gradY_Vy * vy(i,j) +
                                                (1.0 / density(i,j)) * gradY_Pressure);

            double pressure_prime = pressure(i,j) - 0.5 * dt * (gradX_Pressure * vx(i,j) + gradY_Pressure * vy(i,j) +
                                                        gamma * pressure(i,j) * (gradX_Vx + gradY_Vy));

            // Store the extrapolated values at faces
            // Will later use to calculate fluxes via Riemann solver
            rho_XL(i,j) = rho_prime - gradX_Density * dx / 2.0;
            rho_XR(i,j) = rho_prime + gradX_Density * dx / 2.0;

            rho_YB(i,j) = rho_prime - gradY_Density * dy / 2.0;
            rho_YT(i,j) = rho_prime + gradY_Density * dy / 2.0;

            vx_XL(i,j) = vx_prime - gradX_Vx * dx / 2.0;
            vx_XR(i,j) = vx_prime + gradX_Vx * dx / 2.0;
            vx_YB(i,j) = vx_prime - gradY_Vx * dy / 2.0;
            vx_YT(i,j) = vx_prime + gradY_Vx * dy / 2.0; 

            vy_XL(i,j) = vy_prime - gradX_Vy * dx / 2.0;
            vy_XR(i,j) = vy_prime + gradX_Vy * dx / 2.0;
            vy_YB(i,j) = vy_prime - gradY_Vy * dy / 2.0;
            vy_YT(i,j) = vy_prime + gradY_Vy * dy / 2.0;

            P_XL(i,j) = pressure_prime - gradX_Pressure * dx / 2.0;
            P_XR(i,j) = pressure_prime + gradX_Pressure * dx / 2.0;
            P_YB(i,j) = pressure_prime - gradY_Pressure * dy / 2.0;
            P_YT(i,j) = pressure_prime + gradY_Pressure * dy / 2.0;
        }
    );
    //cout << "Finished parallel for" << endl;
    Kokkos::fence();
};

double Fluid::calculateTimeStep(){
    double maxP = 0.0;
    double minRho = 1e10;
    double maxV = 0.0;

    assert(this->_state == assembled);

    // retrieve view objects on device
    auto vx = this->vx.view<Kokkos::DefaultExecutionSpace>();
    auto vy = this->vy.view<Kokkos::DefaultExecutionSpace>();   
    auto pressure = this->pressure.view<Kokkos::DefaultExecutionSpace>();
    auto density = this->density.view<Kokkos::DefaultExecutionSpace>();

    Kokkos::parallel_reduce("compute_dt_params",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {Nx,Ny}),
        KOKKOS_LAMBDA (const int i, const int j, double& local_maxP,
                                        double& local_minRho,
                                        double& local_maxV) {
            local_maxP = max(local_maxP, pressure(i,j));
            local_minRho = min(local_minRho, density(i,j));
            local_maxV = max(local_maxV, sqrt(vx(i,j)*vx(i,j) + vy(i,j)*vy(i,j)));
        }, 
        Kokkos::Max<double>(maxP),
        Kokkos::Min<double>(minRho),
        Kokkos::Max<double>(maxV)
    );

    Kokkos::fence();

    double cs = sqrt(gamma * maxP / minRho);
    //cout << "Sound speed: " << cs << ", Max Velocity: " << maxV << ", Max pressure: " << maxP << ", min density: " << minRho << endl;
    double dt = courant_fac * min(dx, dy) / (cs + maxV);

    return dt;
};

void Fluid::slopeLimiter(double &gradx, double &grady, const Kokkos::View<double**> field,
                    int i, int j, int Ri, int Li, int Ti, int Bi){

    double floor_x, floor_y;
    assert(this->_state == assembled);

    if (gradx == 0.0){
        floor_x = 1e-8;
    } else {
        floor_x = 0.0;
    }
    if (grady == 0.0){
        floor_y = 1e-8;
    } else {
        floor_y = 0.0;
    }
    // Minmod slope limiter
    gradx = max(0.0, min(1.0, ((field(i,j) - field(Li,j)) / dx) / (gradx + floor_x))) * gradx;
    gradx = max(0.0, min(1.0, ((field(Ri,j) - field(i,j)) / dx) / (gradx + floor_x))) * gradx;
    grady = max(0.0, min(1.0, ((field(i,j) - field(i,Bi)) / dy) / (grady + floor_y))) * grady;
    grady = max(0.0, min(1.0, ((field(i,Ti) - field(i,j)) / dy) / (grady + floor_y))) * grady;
};

