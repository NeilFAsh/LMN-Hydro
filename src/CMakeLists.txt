cmake_minimum_required(VERSION 3.10)
project(KHI_Solver)

set(CMAKE_CXX_STANDARD 17)

set(Kokkos_ROOT "/home/nfash/NERS570/Lab12/kokkos/build/")
# set(Kokkos_DIR "${DKokkos_ROOT}/build/cmake_packages/Kokkos/")

find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(fluid_khi
    src/fluid.cpp
    examples/Fluid_KHI.cpp
    src/SnapshotWriter.cpp
)

add_executable(fluid_khi_kokkos
    src/fluid_kokkos.cpp
    examples/Fluid_KHI_Kokkos.cpp
    src/SnapshotWriter.cpp

)
target_link_libraries(fluid_khi_kokkos Kokkos::kokkos)

# Might need to fetch miniz in the future, but for now we include it directly in the source tree.
#  include(FetchContent)

# FetchContent_Declare(
#     miniz
#     GIT_REPOSITORY https://github.com/richgel999/miniz
#     GIT_TAG master
# )

# FetchContent_MakeAvailable(miniz)