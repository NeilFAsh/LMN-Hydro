cmake_minimum_required(VERSION 3.10)
project(LMN_Hydro_Tests)

find_package(Kokkos REQUIRED)

add_executable(timestep_benchmark_kokkos timestep_benchmark_kokkos.cpp)
target_link_libraries(timestep_benchmark_kokkos Kokkos::kokkos)