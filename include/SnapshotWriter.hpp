#ifndef SNAPSHOT_WRITER_HPP
#define SNAPSHOT_WRITER_HPP

#include <string>
#include <vector>


class SnapshotWriter {
public:

    // ----------------------------------------------------
    // Constructor: provide the directory to write snapshots into
    // ----------------------------------------------------
    explicit SnapshotWriter(const std::string& outputDir);

    // ----------------------------------------------------
    // Save a snapshot to an .npz file
    // ----------------------------------------------------
    void save_npz_snapshot(
        const std::string& filename,
        const std::vector<std::vector<double>>& rho,
        const std::vector<std::vector<double>>& vx,
        const std::vector<std::vector<double>>& vy,
        const std::vector<std::vector<double>>& P,
        size_t Nx,
        size_t Ny,
        double t,
        double totalMass,
        double totalEnergy,
        double totalMomentumX,
        double totalMomentumY
    );

private:
    std::string outputDir;

    // ----------------------------------------------------
    // Flatten vector<vector<double>> → 1D contiguous array
    // ----------------------------------------------------
    static std::vector<double> flatten(
        const std::vector<std::vector<double>>& v2d
    );

    // ----------------------------------------------------
    // Writes a single NumPy .npy file into the ZIP archive
    // ----------------------------------------------------
    void add_npy_to_zip(
        void* zipPtr,
        const std::string& name,
        const std::vector<double>& flatData,
        size_t Nx,
        size_t Ny
    );
};

#endif
