# Include after a coherent full WarpX RZ target is available.
if(NOT TARGET WarpX::lib_rz OR NOT TARGET app_rz)
  message(FATAL_ERROR "Native Aitken fixtures require the full RZ application library")
endif()
foreach(name IN ITEMS test_native_longitudinal_aitken test_aitken_rollback)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE WarpX::lib_rz)
  target_compile_features(${name} PRIVATE cxx_std_20)
  target_include_directories(${name} PRIVATE $<TARGET_PROPERTY:app_rz,INCLUDE_DIRECTORIES>)
  target_compile_definitions(${name} PRIVATE $<TARGET_PROPERTY:app_rz,COMPILE_DEFINITIONS>)
  target_compile_options(${name} PRIVATE $<TARGET_PROPERTY:app_rz,COMPILE_OPTIONS>)
endforeach()
# The native fault is injected only after an actual successful Schur action.
# This diagnostic requires a static library and GNU-compatible --wrap linker.
target_link_options(test_aitken_rollback PRIVATE
  "-Wl,--wrap=_ZN5warpx6darwin31CorrectNativeVacuumLongitudinalER5WarpXRKN5amrex8MultiFabEidRKSt5arrayIPS4_Lm3EESB_RS4_SB_")
