#include "SnapshotWriter.hpp"
#include "miniz.h"

#include <cstring>
#include <filesystem>
#include <iostream>


// =========================================================
// Constructor
// =========================================================
SnapshotWriter::SnapshotWriter(const std::string& outputDir)
    : outputDir(outputDir)
{
    std::filesystem::create_directories(outputDir);
}



// =========================================================
// Helper: Flatten vector<vector<double>>
// =========================================================
std::vector<double> SnapshotWriter::flatten(
    const std::vector<std::vector<double>>& v2d)
{
    std::vector<double> flat;
    flat.reserve(v2d.size() * v2d[0].size());

    for (const auto& row : v2d)
        flat.insert(flat.end(), row.begin(), row.end());

    return flat;
}



// =========================================================
// Write a .npy into the zip archive
// =========================================================
void SnapshotWriter::add_npy_to_zip(
    void* zipPtr,
    const std::string& name,
    const std::vector<double>& flatData,
    size_t Nx,
    size_t Ny)
{
    mz_zip_archive* zip = reinterpret_cast<mz_zip_archive*>(zipPtr);

    // -------- Build header dictionary --------
    // Use shape (Ny, Nx) → NumPy standard row-major
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': ("
                         + std::to_string(Ny) + ", " 
                         + std::to_string(Nx) +
                         "), }";

    // -------- Pad header to 16-byte boundary --------
    size_t header_len = header.size() + 1; // +1 newline
    size_t pad = 16 - ((10 + header_len) % 16);
    if (pad == 16) pad = 0;  // exact alignment

    header.append(pad, ' ');
    header.push_back('\n');

    header_len = header.size();

    // -------- Build .npy file buffer --------
    std::vector<unsigned char> buffer(10 + header_len + flatData.size() * sizeof(double));

    unsigned char* p = buffer.data();

    // Magic bytes
    p[0] = 0x93; p[1] = 'N'; p[2] = 'U'; p[3] = 'M';
    p[4] = 'P';  p[5] = 'Y';
    p[6] = 1;    // major
    p[7] = 0;    // minor

    uint16_t hlen_u16 = static_cast<uint16_t>(header_len);

    memcpy(p + 8, &hlen_u16, 2);
    memcpy(p + 10, header.c_str(), header_len);

    memcpy(p + 10 + header_len,
           flatData.data(),
           flatData.size() * sizeof(double));

    // -------- Add to ZIP --------
    mz_zip_writer_add_mem(
        zip,
        name.c_str(),      // do NOT append .npy here
        buffer.data(),
        buffer.size(),
        MZ_BEST_SPEED
    );
}



// =========================================================
// Save snapshot as NPZ
// =========================================================
void SnapshotWriter::save_npz_snapshot(
    const std::string& filename,
    const std::vector<std::vector<double>>& rho,
    const std::vector<std::vector<double>>& vx,
    const std::vector<std::vector<double>>& vy,
    const std::vector<std::vector<double>>& P,
    size_t Nx,
    size_t Ny, 
    double totalMass,
    double totalEnergy,
    double totalMomentumX,
    double totalMomentumY)
{
    std::string path = outputDir + "/" + filename;

    mz_zip_archive zip;
    memset(&zip, 0, sizeof(zip));

    mz_zip_writer_init_file(&zip, path.c_str(), 0);

    // Add .npy with conserved quantities before other values
    std::vector<double> summary = { totalMass, totalEnergy, totalMomentumX, totalMomentumY };
    add_npy_to_zip(&zip, "conserved.npy", summary, 4, 1);
    
    auto rho_f = flatten(rho);
    auto vx_f  = flatten(vx);
    auto vy_f  = flatten(vy);
    auto P_f   = flatten(P);

    add_npy_to_zip(&zip, "rho.npy", rho_f, Nx, Ny);
    add_npy_to_zip(&zip, "vx.npy",  vx_f,  Nx, Ny);
    add_npy_to_zip(&zip, "vy.npy",  vy_f,  Nx, Ny);
    add_npy_to_zip(&zip, "P.npy",   P_f,   Nx, Ny);

    mz_zip_writer_finalize_archive(&zip);
    mz_zip_writer_end(&zip);
}
