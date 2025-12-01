#ifndef SNAPSHOT_WRITER_HPP
#define SNAPSHOT_WRITER_HPP

#include <string>
#include <vector>

class SnapshotWriter {
public:
    SnapshotWriter(const std::string& outputDir);

    void save_npz_snapshot(
        const std::string& filename,
        const std::vector<std::vector<double>>& rho,
        const std::vector<std::vector<double>>& vx,
        const std::vector<std::vector<double>>& vy,
        const std::vector<std::vector<double>>& P,
        size_t Nx,
        size_t Ny
    );

private:

    std::vector<double> flatten(
        const std::vector<std::vector<double>>& field,
        size_t Nx,
        size_t Ny
    );

    void add_npy_to_zip(
        void* zipArchive,
        const std::string& name,
        const double* data,
        size_t Nx,
        size_t Ny
    );

    std::string outputDir;
};

#endif
