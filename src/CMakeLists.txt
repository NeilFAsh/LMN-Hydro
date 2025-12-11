# Mark miniz.c as pure C
set_source_files_properties(miniz.c PROPERTIES LANGUAGE C)

# Pick the correct fluid implementation
if(USE_KOKKOS)
    message(STATUS "Compiling with Kokkos backend")
    set(FLUID_SOURCE fluid_kokkos.cpp)
else()
    message(STATUS "Compiling with normal CPU backend")
    set(FLUID_SOURCE fluid.cpp)
endif()

# Source files for the hydro library
set(LMN_HYDRO_SOURCES
    $(FLUID_SOURCE)
    SnapshotWriter.cpp
    miniz.c
)

# Build static library containing hydro solver
add_library(lmnhydro STATIC
    ${FLUID_SOURCE}
    SnapshotWriter.cpp
    miniz.c
)

if(USE_KOKKOS)
    #set(Kokkos_DIR /home/nfash/NERS570/LMN-Hydro/build/_deps/kokkos-build)
    #find_package(Kokkos REQUIRED)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(lmnhydro PRIVATE Kokkos::kokkos stdc++fs)
    else()
        target_link_libraries(lmnhydro PRIVATE Kokkos::kokkos)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(lmnhydro stdc++fs)
endif()

target_compile_definitions(lmnhydro PRIVATE
    MINIZ_NO_ZLIB_COMPATIBLE_NAMES
)

target_include_directories(lmnhydro PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# For older GCC (< 9)
#if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#    target_link_libraries(lmnhydro stdc++fs)
#endif()