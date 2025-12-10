cmake_minimum_required(VERSION 3.10)
project(KHI_Solver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER /sw/pkgs/arc/gcc/13.2.0/bin/g++)

set(DKokkos_ENABLE_CUDA ON)
set(DKokkos_ENABLE_OPENMP ON)
set(DKokkos_ENABLE_SERIAL ON)

#find_package(Kokkos REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)


include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(fluid_khi)
target_sources(fluid_khi
    PRIVATE 
        ./src/fluid.cpp
        ./examples/fluid_KHI.cpp
        ./src/SnapshotWriter.cpp
        ./src/miniz.c

    PUBLIC
        FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ./include
        FILES
            ./include/fluid.hpp
            ./include/SnapshotWriter.hpp
            ./include/fluid_KHI.hpp
            ./include/miniz.h
)

add_executable(fluid_khi_kokkos)
target_sources(fluid_khi_kokkos
    PRIVATE 
        ./src/fluid_kokkos.cpp
        ./examples/fluid_KHI_Kokkos.cpp
        ./src/SnapshotWriter.cpp
        ./src/miniz.c

    PUBLIC
        FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ./include
        FILES
            ./include/fluid_kokkos.hpp
            ./include/SnapshotWriter.hpp
            ./include/miniz.h
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