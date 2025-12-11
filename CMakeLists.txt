cmake_minimum_required(VERSION 3.10)
project(KHI_Solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER /sw/pkgs/arc/gcc/13.2.0/bin/g++)

set(Kokkos_ENABLE_CUDA ON)
set(Kokkos_ENABLE_OPENMP ON)
set(Kokkos_ENABLE_SERIAL ON)
set(Kokkos_ARCH_SKX ON)
set(Kokkos_ARCH_VOLTA70 ON)

#find_package(Kokkos REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/4.7.01.zip #alt 4.7.01, 5.0.0
)
FetchContent_MakeAvailable(Kokkos)


include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(fluid_khi)
target_link_options(fluid_khi PRIVATE -lstdc++fs)
target_compile_options(fluid_khi PRIVATE -O3 -march=native)
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
target_link_options(fluid_khi_kokkos PRIVATE -lstdc++fs)
target_compile_options(fluid_khi_kokkos PRIVATE -O3 -march=native)
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

add_executable(timestep_benchmark)
target_link_options(timestep_benchmark PRIVATE -lstdc++fs)
target_compile_options(timestep_benchmark PRIVATE -O3 -march=native)
target_sources(timestep_benchmark
    PRIVATE 
        ./src/fluid.cpp
        ./tests/timestep_benchmark.cpp
        ./src/SnapshotWriter.cpp
        ./src/miniz.c

    PUBLIC
        FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ./include
        FILES
            ./include/fluid.hpp
            ./include/SnapshotWriter.hpp
            ./include/miniz.h
)

add_executable(timestep_benchmark_kokkos)
target_link_options(timestep_benchmark_kokkos PRIVATE -lstdc++fs)
target_compile_options(timestep_benchmark_kokkos PRIVATE -O3 -march=native)
target_sources(timestep_benchmark_kokkos
    PRIVATE 
        ./src/fluid_kokkos.cpp
        ./tests/timestep_benchmark_kokkos.cpp
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
target_link_libraries(timestep_benchmark_kokkos Kokkos::kokkos)

add_executable(component_benchmark)
target_link_options(component_benchmark PRIVATE -lstdc++fs)
target_compile_options(component_benchmark PRIVATE -O3 -march=native)
target_sources(component_benchmark
    PRIVATE 
        ./src/fluid.cpp
        ./tests/component_benchmark.cpp
        ./src/SnapshotWriter.cpp
        ./src/miniz.c

    PUBLIC
        FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ./include
        FILES
            ./include/fluid.hpp
            ./include/SnapshotWriter.hpp
            ./include/miniz.h
)

add_executable(component_benchmark_kokkos)
target_link_options(component_benchmark_kokkos PRIVATE -lstdc++fs)
target_compile_options(component_benchmark_kokkos PRIVATE -O3 -march=native)
target_sources(component_benchmark_kokkos
    PRIVATE 
        ./src/fluid_kokkos.cpp
        ./tests/component_benchmark_kokkos.cpp
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
target_link_libraries(component_benchmark_kokkos Kokkos::kokkos)


# Might need to fetch miniz in the future, but for now we include it directly in the source tree.
#  include(FetchContent)

# FetchContent_Declare(
#     miniz
#     GIT_REPOSITORY https://github.com/richgel999/miniz
#     GIT_TAG master
# )

# FetchContent_MakeAvailable(miniz)