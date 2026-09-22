foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    if(D STREQUAL "RZ")
        target_sources(lib_${SD} PRIVATE HybridPICRZOperator.cpp)
    endif()
    target_sources(lib_${SD}
      PRIVATE
        WarpX_PETSc.cpp
    )
endforeach()
