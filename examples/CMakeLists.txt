
if(USE_KOKKOS)
    message(STATUS "Compiling KHI example with Kokkos backend")

    add_executable(fluid_KHI fluid_KHI_kokkos.cpp)
    target_link_libraries(fluid_KHI PRIVATE lmnhydro)
else()
    message(STATUS "Compiling KHI example with CPU backend")

    add_executable(fluid_KHI fluid_KHI.cpp)
    target_link_libraries(fluid_KHI PRIVATE lmnhydro)
endif()