
if(USE_KOKKOS)
    message(STATUS "Compiling KHI example with Kokkos backend")

    add_executable(fluid_KHI fluid_KHI_kokkos.cpp)
    target_link_libraries(fluid_KHI PRIVATE lmnhydro Kokkos::kokkos)
    target_compile_options(fluid_KHI PRIVATE -O3 -march=native)
    target_link_options(fluid_KHI PRIVATE -Wno-deprecated-gpu-targets)

else()
    message(STATUS "Compiling KHI example with CPU backend")

    add_executable(fluid_KHI fluid_KHI.cpp)
    target_link_libraries(fluid_KHI PRIVATE lmnhydro)
    target_compile_options(fluid_KHI PRIVATE -O3 -march=native)

endif()