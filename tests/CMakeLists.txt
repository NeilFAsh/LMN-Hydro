
if(USE_KOKKOS)
    message(STATUS "Compiling benchmark tests with Kokkos backend")

    add_executable(test_timestep timestep_benchmark_kokkos.cpp)
    target_link_libraries(test_timestep PRIVATE lmnhydro)

    add_executable(test_component component_benchmark_kokkos.cpp)
    target_link_libraries(test_component PRIVATE lmnhydro)  
else()
    message(STATUS "Compiling benchmark tests with CPU backend")

    add_executable(test_timestep timestep_benchmark.cpp)
    target_link_libraries(test_timestep PRIVATE lmnhydro)

    add_executable(test_component component_benchmark.cpp)
    target_link_libraries(test_component PRIVATE lmnhydro)  
endif()
