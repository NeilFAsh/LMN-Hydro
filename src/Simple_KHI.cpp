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
            cout << "t = " << t << ", E = " << totalEnergy << ", Mass = " << totalMass 
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
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {

                    // Periodic boundary conditions
                    int Ri = (i + 1) % Nx; // right
                    int Li = (i - 1 + Nx) % Nx; //left 

                    int Ti = (j + 1) % Ny; // top
                    int Bi = (j - 1 + Ny) % Ny; // bottom

                    // update density
                    // subtract fluxes leaving the cell (right and top)
                    // and add fluxes entering the cell (left and bottom)
                    density[i][j] -= (dt / dx) * (flux_rho_X[i][j] - flux_rho_X[Li][j])
                                    + (dt / dy) * (flux_rho_Y[i][j] - flux_rho_Y[i][Bi]);

                    if (density[i][j] <= 0.0) {
                        cout << "Negative density encountered at (" << i << "," << j << "): " << density[i][j] << endl;
                        throw runtime_error("Simulation aborted due to negative or zero density.");
                    };

                    // update momentum in x
                    vx[i][j] -= (dt / dx) * (flux_momx_X[i][j]/density[i][j] - flux_momx_X[Li][j]/density[Li][j]) 
                                + (dt / dy) * (flux_momx_Y[i][j]/density[i][j] - flux_momx_Y[i][Bi]/density[i][Bi]);

                    // update momentum in y
                    vy[i][j] -= (dt / dx) * (flux_momy_X[i][j]/density[i][j] - flux_momy_X[Li][j]/density[Li][j])
                                + (dt / dy) * (flux_momy_Y[i][j]/density[i][j] - flux_momy_Y[i][Bi]/density[i][Bi]);

                    // update energy and pressure
                    // we may need to use old velocities here
                    double E = pressure[i][j]/(gamma - 1) + 0.5 * density[i][j] * 
                                (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]);

                    E -= (dt / dx) * (flux_E_X[i][j] - flux_E_X[Li][j])
                         + (dt / dy) * (flux_E_Y[i][j] - flux_E_Y[i][Bi]);

                    pressure[i][j] = (gamma - 1) * (E - 0.5 * density[i][j] * 
                                        (vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]));
                    if (pressure[i][j] <= 0.0) {
                        cout << "Negative pressure encountered at (" << i << "," << j << "): " << pressure[i][j] << endl;
                        throw runtime_error("Simulation aborted due to negative or zero pressure.");
                    };
                }
            }
        };  
        private:
        void RiemannSolver(){
            // solve the Riemann problems at each face
            // will populate the matrices containing fluxes at each face
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {

                    // Periodic boundary conditions
                    int Ri = (i + 1) % Nx; // right
                    int Li = (i - 1 + Nx) % Nx; //left 

                    int Ti = (j + 1) % Ny; // top
                    int Bi = (j - 1 + Ny) % Ny; // bottom


                    // Simple Riemann solver: average states 
                    // Average the contributions from neighboring cells
                    // Take the left edge from the cell to the right,
                    // and the bottom edge from the cell above
                    double rho_X_star = 0.5 * (rho_XL[Ri][j] + rho_XR[i][j]);
                    double rho_Y_star = 0.5 * (rho_YB[i][Ti] + rho_YT[i][j]);

                    double momx_X_star = 0.5 * (rho_XL[Ri][j] * vx_XL[Ri][j] + rho_XR[i][j] * vx_XR[i][j]);
                    double momx_Y_star = 0.5 * (rho_YB[i][Ti] * vx_YB[i][Ti] + rho_YT[i][j] * vx_YT[i][j]);
                    double momy_X_star = 0.5 * (rho_XL[Ri][j] * vy_XL[Ri][j] + rho_XR[i][j] * vy_XR[i][j]);
                    double momy_Y_star = 0.5 * (rho_YB[i][Ti] * vy_YB[i][Ti] + rho_YT[i][j] * vy_YT[i][j]);

                    double E_X_star = 0.5 * (P_XL[Ri][j]/(gamma - 1) + 0.5 * rho_XL[Ri][j] * 
                                        (vx_XL[Ri][j]*vx_XL[Ri][j] + vy_XL[Ri][j]*vy_XL[Ri][j]) +
                                        P_XR[i][j]/(gamma - 1) + 0.5 * rho_XR[i][j] * 
                                        (vx_XR[i][j]*vx_XR[i][j] + vy_XR[i][j]*vy_XR[i][j]));
                    double E_Y_star = 0.5 * (P_YB[i][Ti]/(gamma - 1) + 0.5 * rho_YB[i][Ti] * 
                                        (vx_YB[i][Ti]*vx_YB[i][Ti] + vy_YB[i][Ti]*vy_YB[i][Ti]) +
                                        P_YT[i][j]/(gamma - 1) + 0.5 * rho_YT[i][j] * 
                                        (vx_YT[i][j]*vx_YT[i][j] + vy_YT[i][j]*vy_YT[i][j]));

                    double P_X_star = (gamma - 1) * (E_X_star - 0.5 * (momx_X_star*momx_X_star + momy_X_star*momy_X_star) / rho_X_star);
                    double P_Y_star = (gamma - 1) * (E_Y_star - 0.5 * (momx_Y_star*momx_Y_star + momy_Y_star*momy_Y_star) / rho_Y_star);

                    // compute fluxes 
                    double C = sqrt(gamma * P_XL[Ri][j] / rho_XL[Ri][j]) + abs(vx_XL[Ri][j]);
                    C = max(C, sqrt(gamma * P_XR[i][j] / rho_XR[i][j]) + abs(vx_XR[i][j]));
                    C = max(C, sqrt(gamma * P_YB[i][Ti] / rho_YB[i][Ti]) + abs(vy_YB[i][Ti]));
                    C = max(C, sqrt(gamma * P_YT[i][j] / rho_YT[i][j]) + abs(vy_YT[i][j]));

                    double flux_rho_X = momx_X_star - C * 0.5 * (rho_XL[Ri][j] - rho_XR[i][j]);
                    double flux_rho_Y = momx_Y_star - C * 0.5 * (rho_YB[i][Ti] - rho_YT[i][j]);
                    
                    double flux_momx_X;
                    flux_momx_X = momx_X_star * momx_X_star / rho_X_star + P_X_star;
                    flux_momx_X = flux_momx_X - C * 0.5 * (rho_XL[Ri][j] * vx_XL[Ri][j] - rho_XR[i][j] * vx_XR[i][j]);
                    
                    double flux_momx_Y; 
                    flux_momx_Y = momy_Y_star * momx_Y_star / rho_Y_star;
                    flux_momx_Y = flux_momx_Y - C * 0.5 * (rho_YB[i][Ti] * vx_YB[i][Ti] - rho_YT[i][j] * vx_YT[i][j]);
                    
                    double flux_momy_X;
                    flux_momy_X = momy_X_star * momx_X_star / rho_X_star; 
                    flux_momy_X = flux_momy_X - C * 0.5 * (rho_XL[Ri][j] * vy_XL[Ri][j] - rho_XR[i][j] * vy_XR[i][j]);
                    
                    double flux_momy_Y;
                    flux_momy_Y = momy_Y_star * momx_Y_star / rho_Y_star + P_X_star;
                    flux_momy_Y = flux_momy_Y - C * 0.5 * (rho_YB[i][Ti] * vy_YB[i][Ti] - rho_YT[i][j] * vy_YT[i][j]);

                    double flux_E_X;
                    flux_E_X = (E_X_star + P_X_star) * momx_X_star / rho_X_star;
                    flux_E_X = flux_E_X - C * 0.5 * ( (P_XL[Ri][j]/(gamma - 1) + 0.5 * rho_XL[Ri][j] * 
                                        (vx_XL[Ri][j]*vx_XL[Ri][j] + vy_XL[Ri][j]*vy_XL[Ri][j]) ) - 
                                        (P_XR[i][j]/(gamma - 1) + 0.5 * rho_XR[i][j] * 
                                        (vx_XR[i][j]*vx_XR[i][j] + vy_XR[i][j]*vy_XR[i][j]) ) );

                    double flux_E_Y;
                    flux_E_Y = (E_Y_star + P_Y_star) * momx_Y_star / rho_Y_star;
                    flux_E_Y = flux_E_Y - C * 0.5 * ( (P_YB[i][Ti]/(gamma - 1) + 0.5 * rho_YB[i][Ti] * 
                                        (vx_YB[i][Ti]*vx_YB[i][Ti] + vy_YB[i][Ti]*vy_YB[i][Ti]) ) - 
                                        (P_YT[i][j]/(gamma - 1) + 0.5 * rho_YT[i][j] * 
                                        (vx_YT[i][j]*vx_YT[i][j] + vy_YT[i][j]*vy_YT[i][j]) ) );
                    
                    // These are the fluxes through the right and top faces of cell (i,j)
                    this->flux_rho_X[i][j] = flux_rho_X;
                    this->flux_rho_Y[i][j] = flux_rho_Y;
                    this->flux_momx_X[i][j] = flux_momx_X;
                    this->flux_momx_Y[i][j] = flux_momx_Y;
                    this->flux_momy_X[i][j] = flux_momy_X;
                    this->flux_momy_Y[i][j] = flux_momy_Y;
                    this->flux_E_X[i][j] = flux_E_X;
                    this->flux_E_Y[i][j] = flux_E_Y;
                }
            }
        };
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

    fluid.useSlopeLimiter = 1; // enable slope limiter
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
