#include "SnapshotWriter.hpp"
#include "miniz.h"

#include <cstring>
#include <filesystem>
#include <iostream>

// Creates output directory if it doesn't exist
SnapshotWriter::SnapshotWriter(const std::string& outputDir)
    : outputDir(outputDir)
{
    std::filesystem::create_directories(outputDir);
}


// ================= PRIVATE ===================
void SnapshotWriter::add_npy_to_zip(
    void* zipArchive,
    const std::string& name,
    const double* data,
    size_t Nx,
    size_t Ny
) {

    // Goal: Convert a raw C++ double* array into a valid NumPy .npy file and stick it into the .npz archive.

    mz_zip_archive* zip = reinterpret_cast<mz_zip_archive*>(zipArchive); // cast to proper type (from void*)

    // Build NumPy header
    // 'descr': '<f8' --> matches C++ double ; 'fortran_order': False --> row-major layout (C)
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': ("
                         + std::to_string(Nx) + ", " + std::to_string(Ny) +
                         "), }";
    
    // Pad header to 16-byte alignment (NumPy requires that the header ends at a 16-byte boundary)
    size_t total_header = 10 + header.size() + 1;
    size_t padding = 16 - (total_header % 16);
    header.append(padding, ' ');

    // Magic + version + header length (this writes the .npy file preamble) (called magic prefix in NumPy docs)
    // NumPy requires \x93NUMPY at the start of the file to know is it a .npy file
    unsigned char npy_header[10];
    npy_header[0] = 0x93;
    npy_header[1] = 'N';
    npy_header[2] = 'U';
    npy_header[3] = 'M';
    npy_header[4] = 'P';
    npy_header[5] = 'Y';
    npy_header[6] = 1;
    npy_header[7] = 0;

    // insert header length. Note: header length does not include the first 10 bytes
    uint16_t header_len = header.size() + 1;
    memcpy(&npy_header[8], &header_len, sizeof(uint16_t));

    // Combine header + data

    // Allocate buffer for entire .npy file
    std::vector<unsigned char> buffer(10 + header_len + Nx*Ny*sizeof(double));
    // Copy header and data into buffer
    memcpy(buffer.data(), npy_header, 10);
    memcpy(buffer.data() + 10, header.c_str(), header_len);
    memcpy(buffer.data() + 10 + header_len, data, Nx*Ny*sizeof(double));

    // Write to archive
    std::string fname = name + ".npy";
    mz_zip_writer_add_mem(zip, fname.c_str(), buffer.data(), buffer.size(), MZ_BEST_COMPRESSION);
}



// ================= PUBLIC ===================
void SnapshotWriter::save_npz_snapshot(
    const std::string& filename,
    const std::vector<std::vector<double>>& rho,
    const std::vector<std::vector<double>>& vx,
    const std::vector<std::vector<double>>& vy,
    const std::vector<std::vector<double>>& P,
    size_t Nx,
    size_t Ny
) {
    std::string path = outputDir + "/" + filename; // path to the output .npz file

    mz_zip_archive zip; //struct from miniz that represents a zip file, must be initialized
    memset(&zip, 0, sizeof(zip)); // initialize to 0

    mz_zip_writer_init_file(&zip, path.c_str(), 0); // creates the zip file with the given name

    // Add each file as a .npy files to the zip archive
    add_npy_to_zip(&zip, "rho", rho.data(), Nx, Ny);
    add_npy_to_zip(&zip, "vx",  vx.data(), Nx, Ny);
    add_npy_to_zip(&zip, "vy",  vy.data(), Nx, Ny);
    add_npy_to_zip(&zip, "P",   P.data(),  Nx, Ny);

    mz_zip_writer_finalize_archive(&zip); // finalizes the archive
    mz_zip_writer_end(&zip); // cleans up the mz_zip_archive structure
}
