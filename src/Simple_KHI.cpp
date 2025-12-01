#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

using namespace std;

class Fluid {
    public: 
        vector<vector<double>> vx;
        vector<vector<double>> vy;
        vector<vector<double>> pressure;
        vector<vector<double>> density;
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
        
    private:
        
        double dx;
        double dy;
        double BoxSizeX;
        double BoxSizeY;
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

        // Constructor implementation
        public:
        Fluid(int Nx, int Ny, double boxSizeX, double boxSizeY){

            this->Nx = Nx; 
            this->Ny = Ny; 

            this->BoxSizeX = boxSizeX;
            this->BoxSizeY = boxSizeY;

            this->dx = boxSizeX / Nx; 
            this->dy = boxSizeY / Ny; 
            this->cellVol = dx * dy;

            // allocate memory for the fluid properties

            this->vx = vector<vector<double>>(this->Nx, 
                        vector<double>(this->Ny, 0.0));
            this->vy = vector<vector<double>>(this->Nx, 
                        vector<double>(this->Ny, 0.0));
            this->pressure = vector<vector<double>>(this->Nx, 
                        vector<double>(this->Ny, 0.0));
            this->density = vector<vector<double>>(this->Nx, 
                        vector<double>(this->Ny, 0.0));

            this->isNotBoundary = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 1.0));

            this->rho_XL = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0)); 
            this->rho_XR = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->rho_YB = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->rho_YT = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0)); 
            this->vx_XL = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vx_XR = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vx_YB = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vx_YT = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vy_XL = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vy_XR = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vy_YB = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->vy_YT = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->P_XL = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->P_XR = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->P_YB = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->P_YT = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));

            this->flux_rho_X = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_rho_Y = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_momx_X = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_momx_Y = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_momy_X = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_momy_Y = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_E_X = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
            this->flux_E_Y = vector<vector<double>>(this->Nx, 
                            vector<double>(this->Ny, 0.0));
        };

        public:
        void initializeKHI(){
            // Initialize a Kelvin-Helmholtz Instability setup
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
        
        private:
        void printState(int step){
            double totalEnergy = 0.0;
            double totalMass = 0.0;
            double totalMomentumX = 0.0;
            for(int j = Ny - 1; j >= 0; j--){
                for(int i = 0; i < Nx; i++){
                    totalEnergy += (pressure[i][j]/(gamma - 1) + 0.5 * density[i][j] * 
                                    (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j])) * cellVol;
                    totalMass += density[i][j] * cellVol;
                    totalMomentumX += density[i][j] * vx[i][j] * cellVol;
                }
            }
            cout << "step = " << step << "t = " << t << ", E = " << totalEnergy << ", Mass = " << totalMass 
                     << ", Px = " << totalMomentumX << endl;
        };

        public:
        void runSimulation(double tFinal, double tOut){

            initializeKHI();
            t = 0.0;
            this->tFinal = tFinal;
            this->tOut = tOut;

            int numOutputs = 0;
            while (t < tFinal){

                if (t >= numOutputs * tOut){
                    printState(numOutputs);
                    numOutputs++;
                }

                runTimeStep();
            }
        };
        public:
        void runTimeStep(){
            double dt = calculateTimeStep();
            extrapolateToFaces(dt);
            RiemannSolver();
            updateStates(dt);
            //cout << "Finished timestep, current dt =  " << dt << endl;
            t += dt;
        };
        private:
        void updateStates(double dt){
            // update cell-centered states using computed fluxes
            int throwException = 0;
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {

                    // Periodic boundary conditions
                    int Ri = (i + 1) % Nx; // right
                    int Li = (i - 1 + Nx) % Nx; //left 

                    int Ti = (j + 1) % Ny; // top
                    int Bi = (j - 1 + Ny) % Ny; // bottom

                    // Get conserved quantities
                    double mass = density[i][j] * cellVol;
                    double momx = mass * vx[i][j];
                    double momy = mass * vy[i][j];
                    double E = cellVol * pressure[i][j]/(gamma - 1) + 0.5 * mass * 
                                    (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]);

                    // update density
                    // subtract fluxes leaving the cell (right and top)
                    // and add fluxes entering the cell (left and bottom)
                    mass -= dt * dy * (flux_rho_X[i][j] - flux_rho_X[Li][j])
                                     + dt * dx * (flux_rho_Y[i][j] - flux_rho_Y[i][Bi]);

                    // update momentum in x
                    momx -= dt * dy * (flux_momx_X[i][j] - flux_momx_X[Li][j])
                                     + dt * dx * (flux_momx_Y[i][j] - flux_momx_Y[i][Bi]);

                    // update momentum in y
                    momy -= dt * dy * (flux_momy_X[i][j] - flux_momy_X[Li][j])
                                     + dt * dx * (flux_momy_Y[i][j] - flux_momy_Y[i][Bi]);

                    // update energy
                    E -= dt * dy * (flux_E_X[i][j] - flux_E_X[Li][j])
                             + dt * dx * (flux_E_Y[i][j] - flux_E_Y[i][Bi]);

                    // update primitive variables
                    density[i][j] = mass / cellVol;
                    vx[i][j] = momx / mass;
                    vy[i][j] = momy / mass;
                    pressure[i][j] = (E - 0.5 * mass * (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j])) / cellVol * (gamma - 1);
                    
                    // This really shouldn't be done, ideally we would modify
                    // the Riemann solver to prevent negative pressures/densities
                    // if (pressure[i][j] <= 0.0) { 
                    //     pressure[i][j] = 1e-8;
                    // }
                    
                    if (density[i][j] <= 0.0) {
                        cout << "Negative density encountered at (" << i << "," << j << "): " << density[i][j] << endl;
                        throwException = 1;
                    };
                    if (pressure[i][j] <= 0.0) {
                        cout << "Negative pressure encountered at (" << i << "," << j << "): " << pressure[i][j] << endl;
                        throwException = 1;
                    };
                }
            }
            if (throwException){
                throw runtime_error("Simulation encountered non-physical state (negative or zero density or pressure).");
            }
        };  
        private:
        void RiemannSolver() {
            // Solve Riemann problems at each right/top face of cell (i,j).
            // We'll assemble conserved left and right states from the face-extrapolated primitives
            // and apply the Rusanov flux formula.
            for (int i = 0; i < Nx; ++i) {
                for (int j = 0; j < Ny; ++j) {

                    // neighbor indices (periodic)
                    int Ri = (i + 1) % Nx;
                    int Li = (i - 1 + Nx) % Nx;
                    int Ti = (j + 1) % Ny;
                    int Bi = (j - 1 + Ny) % Ny;

                    //
                    // --- X-face between cell i and Ri (right face of cell i)
                    // left state = right-extrapolation of cell i  -> (rho_XR[i][j], vx_XR[i][j], vy_XR[i][j], P_XR[i][j])
                    // right state = left-extrapolation of cell Ri -> (rho_XL[Ri][j], vx_XL[Ri][j], vy_XL[Ri][j], P_XL[Ri][j])
                    //
                    {
                        // primitives L (cell i right face)
                        double rhoL = rho_XR[i][j];
                        double uL = vx_XR[i][j];
                        double vL = vy_XR[i][j];
                        double pL = P_XR[i][j];

                        // primitives R (cell Ri left face)
                        double rhoR = rho_XL[Ri][j];
                        double uR = vx_XL[Ri][j];
                        double vR = vy_XL[Ri][j];
                        double pR = P_XL[Ri][j];

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
                        this->flux_rho_X[i][j] = 0.5 * (FL_rho + FR_rho) - 0.5 * smax * (UR_rho - UL_rho);
                        this->flux_momx_X[i][j] = 0.5 * (FL_mx + FR_mx) - 0.5 * smax * (UR_mx - UL_mx);
                        this->flux_momy_X[i][j] = 0.5 * (FL_my + FR_my) - 0.5 * smax * (UR_my - UL_my);
                        this->flux_E_X[i][j]    = 0.5 * (FL_E + FR_E) - 0.5 * smax * (UR_E - UL_E);
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
                        double rhoL = rho_YT[i][j];
                        double uL = vx_YT[i][j];
                        double vL = vy_YT[i][j];
                        double pL = P_YT[i][j];

                        // primitives R (upper side of top-face) = YB of (i,Ti) is the state just above the face
                        double rhoR = rho_YB[i][Ti];
                        double uR = vx_YB[i][Ti];
                        double vR = vy_YB[i][Ti];
                        double pR = P_YB[i][Ti];

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
                        this->flux_rho_Y[i][j] = 0.5 * (GL_rho + GR_rho) - 0.5 * smax * (UR_rho - UL_rho);
                        this->flux_momx_Y[i][j] = 0.5 * (GL_mx + GR_mx) - 0.5 * smax * (UR_mx - UL_mx);
                        this->flux_momy_Y[i][j] = 0.5 * (GL_my + GR_my) - 0.5 * smax * (UR_my - UL_my);
                        this->flux_E_Y[i][j]    = 0.5 * (GL_E + GR_E) - 0.5 * smax * (UR_E - UL_E);
                    }
                }
            }
        }
        private:
        void extrapolateToFaces(float dt){
            // extrapolate cell-centered values to faces
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    
                    // Periodic boundary conditions
                    int Ri = (i + 1) % Nx; // right
                    int Li = (i - 1 + Nx) % Nx; //left 

                    int Ti = (j + 1) % Ny; // top
                    int Bi = (j - 1 + Ny) % Ny; // bottom

                    // compute gradients using central differences
                    double gradX_Density = (density[Ri][j] - density[Li][j]) / (2.0*dx);
                    double gradY_Density = (density[i][Ti] - density[i][Bi]) / (2.0*dy);

                    double gradX_Pressure = (pressure[Ri][j] - pressure[Li][j]) / (2.0*dx);
                    double gradY_Pressure = (pressure[i][Ti] - pressure[i][Bi]) / (2.0*dy);

                    double gradX_Vx = (vx[Ri][j] - vx[Li][j]) / (2.0*dx);
                    double gradY_Vx = (vx[i][Ti] - vx[i][Bi]) / (2.0*dy);
                    double gradX_Vy = (vy[Ri][j] - vy[Li][j]) / (2.0*dx);
                    double gradY_Vy = (vy[i][Ti] - vy[i][Bi]) / (2.0*dy);

                    // apply slope limiter if enabled
                    if(useSlopeLimiter){
                        slopeLimiter(gradX_Density, gradY_Density, density,
                                        i, j, Ri, Li, Ti, Bi);
                        slopeLimiter(gradX_Pressure, gradY_Pressure, pressure,
                                        i, j, Ri, Li, Ti, Bi);
                        slopeLimiter(gradX_Vx, gradY_Vx, vx,
                                        i, j, Ri, Li, Ti, Bi);
                        slopeLimiter(gradX_Vy, gradY_Vy, vy,
                                        i, j, Ri, Li, Ti, Bi);
                    }

                    // extrapolate cell-centered values to faces
                    double rho_prime = density[i][j] - 0.5 * dt * (gradX_Density * vx[i][j] + gradY_Density * vy[i][j] +
                                                                density[i][j] * (gradX_Vx + gradY_Vy));
                    double vx_prime = vx[i][j] - 0.5 * dt * (gradX_Vx * vx[i][j] + gradY_Vx * vy[i][j] +
                                                        (1.0 / density[i][j]) * gradX_Pressure);
                    double vy_prime = vy[i][j] - 0.5 * dt * (gradX_Vy * vx[i][j] + gradY_Vy * vy[i][j] +
                                                        (1.0 / density[i][j]) * gradY_Pressure);

                    double pressure_prime = pressure[i][j] - 0.5 * dt * (gradX_Pressure * vx[i][j] + gradY_Pressure * vy[i][j] +
                                                                gamma * pressure[i][j] * (gradX_Vx + gradY_Vy));

                    // Store the extrapolated values at faces
                    // Will later use to calculate fluxes via Riemann solver
                    this->rho_XL[i][j] = rho_prime - gradX_Density * dx / 2.0;
                    this->rho_XR[i][j] = rho_prime + gradX_Density * dx / 2.0;

                    this->rho_YB[i][j] = rho_prime - gradY_Density * dy / 2.0;
                    this->rho_YT[i][j] = rho_prime + gradY_Density * dy / 2.0;

                    this->vx_XL[i][j] = vx_prime - gradX_Vx * dx / 2.0;
                    this->vx_XR[i][j] = vx_prime + gradX_Vx * dx / 2.0;
                    this->vx_YB[i][j] = vx_prime - gradY_Vx * dy / 2.0;
                    this->vx_YT[i][j] = vx_prime + gradY_Vx * dy / 2.0; 

                    this->vy_XL[i][j] = vy_prime - gradX_Vy * dx / 2.0;
                    this->vy_XR[i][j] = vy_prime + gradX_Vy * dx / 2.0;
                    this->vy_YB[i][j] = vy_prime - gradY_Vy * dy / 2.0;
                    this->vy_YT[i][j] = vy_prime + gradY_Vy * dy / 2.0;

                    this->P_XL[i][j] = pressure_prime - gradX_Pressure * dx / 2.0;
                    this->P_XR[i][j] = pressure_prime + gradX_Pressure * dx / 2.0;
                    this->P_YB[i][j] = pressure_prime - gradY_Pressure * dy / 2.0;
                    this->P_YT[i][j] = pressure_prime + gradY_Pressure * dy / 2.0;
                }
            }
        };
        public:
        double calculateTimeStep(){
            double maxP = 0.0;
            double minRho = 1e10;
            double maxV = 0.0;

            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    maxP = max(maxP, pressure[i][j]);
                    minRho = min(minRho, density[i][j]);
                    maxV = max(maxV, sqrt(vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]));
                }
            }

            double cs = sqrt(gamma * maxP / minRho);
            //cout << "Sound speed: " << cs << ", Max Velocity: " << maxV << ", Max pressure: " << maxP << ", min density: " << minRho << endl;
            double dt = courant_fac * min(dx, dy) / (cs + maxV);

            if (minRho == 1e10){
                for (int i = 0; i < 10; i++){
                    cout << density[i][i] << endl;
                }
            }

            return dt;
        };
        private:
        void slopeLimiter(double &gradx, double &grady, vector<vector<double>> field,
                            int i, int j, int Ri, int Li, int Ti, int Bi){

            double floor_x, floor_y;
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
            gradx = max(0.0, min(1.0, ((field[i][j] - field[Li][j]) / dx) / (gradx + floor_x))) * gradx;
            gradx = max(0.0, min(1.0, ((field[Ri][j] - field[i][j]) / dx) / (gradx + floor_x))) * gradx;
            grady = max(0.0, min(1.0, ((field[i][j] - field[i][Bi]) / dy) / (grady + floor_y))) * grady;
            grady = max(0.0, min(1.0, ((field[i][Ti] - field[i][j]) / dy) / (grady + floor_y))) * grady;
        };
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
    Fluid fluid(Nx, Ny, boxSizeX, boxSizeY);
    cout << "Initialized fluid domain of size " << Nx << " x " << Ny << endl;

    fluid.useSlopeLimiter = 0; // enable slope limiter
    //fluid.initializeKHI();
    //fluid.runTimeStep();
    try {
        fluid.runSimulation(tFinal,tOut);
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
        cerr << "Simulation terminated prematurely at time t = " << fluid.t << endl;
        return -1;
    }
    return 0;
}