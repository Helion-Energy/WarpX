foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
      ElectronInertiaElliptic.cpp
      HybridPICModel.cpp
      EBJBoundary.cpp
      EBJBoundaryFold.cpp
      ExternalVectorPotential.cpp
    )
endforeach()
