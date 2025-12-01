cmake_minimum_required(VERSION 3.10)
project(KHI_Solver)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(khi
    src/Simple_KHI.cpp
    src/SnapshotWriter.cpp
)

# Might need to fetch miniz in the future, but for now we include it directly in the source tree.
#  include(FetchContent)

# FetchContent_Declare(
#     miniz
#     GIT_REPOSITORY https://github.com/richgel999/miniz
#     GIT_TAG master
# )

# FetchContent_MakeAvailable(miniz)