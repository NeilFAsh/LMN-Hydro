cmake_minimum_required(VERSION 3.16)

# Make sure C is listed, otherwise miniz.c won't compile correctly
project(LMNHydro LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /sw/pkgs/arc/gcc/13.2.0/bin/g++) # potentially problematic

# Older GCC needs filesystem linking
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_link_options(-lstdc++fs)
endif()

set(Kokkos_ENABLE_CUDA ON)
set(Kokkos_ENABLE_OPENMP ON)
set(Kokkos_ENABLE_SERIAL ON)
set(Kokkos_ARCH_SKX ON)
set(Kokkos_ARCH_VOLTA70 ON)

#find_package(Kokkos REQUIRED)
add_compile_options(-Wno-deprecated-gpu-targets)
include(FetchContent)
FetchContent_Declare(
  Kokkos
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip #alt 4.7.01, 5.0.0
)
FetchContent_MakeAvailable(Kokkos)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
