foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        CircuitCoupler.cpp
        CircuitCoupling.cpp
    )
endforeach()

add_subdirectory(Coils)
