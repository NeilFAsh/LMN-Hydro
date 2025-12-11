# Mark miniz.c as pure C
set_source_files_properties(miniz.c PROPERTIES LANGUAGE C)

# Source files for the hydro library
set(LMN_HYDRO_SOURCES
    fluid.cpp
    SnapshotWriter.cpp
    miniz.c
)

# Build static library containing hydro solver
add_library(lmnhydro STATIC ${LMN_HYDRO_SOURCES})

# Ensure miniz uses its internal API names (no zlib-compat layer)
target_compile_definitions(lmnhydro PRIVATE
    MINIZ_NO_ZLIB_COMPATIBLE_NAMES
)
target_link_libraries(fluid_khi_kokkos Kokkos::kokkos)

# Include headers for library consumers
target_include_directories(lmnhydro PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# For older GCC (< 9)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(lmnhydro stdc++fs)
endif()