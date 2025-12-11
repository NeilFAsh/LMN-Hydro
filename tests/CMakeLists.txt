
if(USE_KOKKOS)
    message(STATUS "Compiling benchmark tests with Kokkos backend")

    add_executable(test_timestep timestep_benchmark_kokkos.cpp)
    target_link_libraries(test_timestep PRIVATE lmnhydro Kokkos::kokkos)
    target_compile_options(test_timestep PRIVATE -O3 -march=native)
    target_link_options(test_timestep PRIVATE -Wno-deprecated-gpu-targets)

    add_executable(test_component component_benchmark_kokkos.cpp)
    target_link_libraries(test_component PRIVATE lmnhydro Kokkos::kokkos) 
    target_compile_options(test_component PRIVATE -O3 -march=native)
    target_link_options(test_component PRIVATE -Wno-deprecated-gpu-targets)

else()
    message(STATUS "Compiling benchmark tests with CPU backend")

    add_executable(test_timestep timestep_benchmark.cpp)
    target_link_libraries(test_timestep PRIVATE lmnhydro)
    target_compile_options(test_timestep PRIVATE -O3 -march=native)

    add_executable(test_component component_benchmark.cpp)
    target_link_libraries(test_component PRIVATE lmnhydro)  
    target_compile_options(test_component PRIVATE -O3 -march=native)
endif()
