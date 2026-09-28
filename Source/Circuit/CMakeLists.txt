foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_link_libraries(lib_${SD} PRIVATE ${CMAKE_DL_LIBS})
    target_sources(lib_${SD}
      PRIVATE
        CircuitCoupler.cpp
    NativeCircuitTransaction.cpp
        CircuitCoupling.cpp
        DeviceCircuit.cpp
        DarwinDeviceCoupling.cpp
        DeviceCircuitRate.cpp
        NativeEndpointRateCoupling.cpp
        NativeSourceImpulseCoupling.cpp
    )
endforeach()

add_subdirectory(Coils)
