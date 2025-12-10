cmake_minimum_required(VERSION 3.16)

# Make sure C is listed, otherwise miniz.c won't compile correctly
project(LMNHydro LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Older GCC needs filesystem linking
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_link_options(-lstdc++fs)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)