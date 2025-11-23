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
        float tFinal = 2;
        float tOut = 0.01;
        int useSlopeLimiter = 0;

        // fluid params
        float courant_fac = 0.4;
        float w0 = 0.1;
        float sigma = 0.05/sqrt(2.);
        float gamma = 5./3.;


    private:
        int Nx;
        int Ny;
        double dx;
        double dy;
        double BoxSizeX;
        double BoxSizeY;
        double cellVol;

        vector<vector<double>> vx_new;
        vector<vector<double>> vy_new;
        vector<vector<double>> isNotBoundary;

        Fluid(int Nx, int Ny, double boxSizeX, double boxSizeY){

            this->Nx = Nx; 
            this->Ny = Ny; 

            this->BoxSizeX = boxSizeX;
            this->BoxSizeY = boxSizeY;

            this->dx = boxSizeX / Nx; 
            this->dy = boxSizeY / Ny; 
            this->cellVol = dx * dy;

            // allocate memory for the fluid properties

            this->vx = std::vector<std::vector<double>>(this->Nx, 
                        std::vector<double>(this->Ny, 0.0));
            this->vy = std::vector<std::vector<double>>(this->Nx, 
                        std::vector<double>(this->Ny, 0.0));
            this->pressure = std::vector<std::vector<double>>(this->Nx, 
                        std::vector<double>(this->Ny, 0.0));

            this->vx_new = std::vector<std::vector<double>>(this->Nx, 
                            std::vector<double>(this->Ny, 0.0));
            this->vy_new = std::vector<std::vector<double>>(this->Nx, 
                            std::vector<double>(this->Ny, 0.0));
            this->isNotBoundary = std::vector<std::vector<double>>(this->Nx, 
                            std::vector<double>(this->Ny, 1.0));

            }

        void initializeKHI(){
            // Initialize a Kelvin-Helmholtz Instability setup
            for(int i = 0; i <Nx; i++){
                for(int j = 0; j <Ny; j++){
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
        }
        
        void printState(int step){
            cout << "Step: " << step << ", Time: " << t << endl;
            for(int j = Ny - 1; j >= 0; j--){
                for(int i = 0; i < Nx; i++){
                    cout << "(" << vx[i][j] << ", " << vy[i][j] << ") ";
                }
                cout << endl;
            }
        }

        void runStep(){

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
            double dt = courant_fac * min(dx, dy) / (cs + maxV);

            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    
                    // Periodic boundary conditions
                    int Ri = (i + 1) % Nx; // right
                    int Li = (i - 1 + Nx) % Nx; //left 

                    int Ti = (j + 1) % Ny; // top
                    int Bi = (j - 1 + Ny) % Ny; // bottom

                    double gradX_Density = (density[Ri][j] - density[Li][j]) / 2.0*dx;
                    double gradY_Density = (density[i][Ti] - density[i][Bi]) / 2.0*dy;

                    double gradX_Pressure = (pressure[Ri][j] - pressure[Li][j]) / 2.0*dx;
                    double gradY_Pressure = (pressure[i][Ti] - pressure[i][Bi]) / 2.0*dy;

                    double gradX_Vx = (vx[Ri][j] - vx[Li][j]) / 2.0*dx;
                    double gradY_Vx = (vx[i][Ti] - vx[i][Bi]) / 2.0*dy;
                    double gradX_Vy = (vy[Ri][j] - vy[Li][j]) / 2.0*dx;
                    double gradY_Vy = (vy[i][Ti] - vy[i][Bi]) / 2.0*dy;

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
                                                        (1.0 / pressure[i][j]) * gradX_Pressure);
                    double vy_prime = vy[i][j] - 0.5 * dt * (gradX_Vy * vx[i][j] + gradY_Vy * vy[i][j] +
                                                        (1.0 / pressure[i][j]) * gradY_Pressure);

                    double pressure_prime = pressure[i][j] - 0.5 * dt * (gradX_Pressure * vx[i][j] + gradY_Pressure * vy[i][j] +
                                                                gamma * pressure[i][j] * (gradX_Vx + gradY_Vy));

                    // NOTE: the "left" edge values will need to be shifted when we calculate fluxes through a face
                    double rho_XL = rho_prime - gradX_Density * dx / 2.0;
                    double rho_XR = rho_prime + gradX_Density * dx / 2.0;

                    double rho_YB = rho_prime - gradY_Density * dy / 2.0;
                    double rho_YT = rho_prime + gradY_Density * dy / 2.0;

                    double vx_XL = vx_prime - gradX_Vx * dx / 2.0;
                    double vx_XR = vx_prime + gradX_Vx * dx / 2.0;




                

                }
            }
        }
        void slopeLimiter(double &gradx, double &grady, vector<vector<double>> field,
                            int i, int j, int Ri, int Li, int Ti, int Bi){
            // Minmod slope limiter
            gradx = max(0.0, min(1.0, ((field[i][j] - field[Li][j]) / dx) / gradx)) * gradx;
            gradx = max(0.0, min(1.0, ((field[Ri][j] - field[i][j]) / dx) / gradx)) * gradx;
            grady = max(0.0, min(1.0, ((field[i][j] - field[i][Bi]) / dy) / grady)) * grady;
            grady = max(0.0, min(1.0, ((field[i][Ti] - field[i][j]) / dy) / grady)) * grady;
        }

};
