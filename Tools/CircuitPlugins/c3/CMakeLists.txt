# Copyright 2026 The WarpX Community
#
# Standalone build of the c3 act-bank external-circuit plugin: no WarpX or
# AMReX dependency (the ABI headers are vendored in vendor/). Produces
#   libc3circuit.so   the dlopen'd plugin (extern "C" factory + ABI stamp)
#   c3_harness        the offline validation driver
#
#   cmake -S . -B build && cmake --build build -j
#
# PRECISION NOTE: the plugin is double-precision throughout and the ABI
# version stamp does NOT encode the amrex::Real width -- load only into a
# WarpX built with -DWarpX_PRECISION=DOUBLE, same toolchain family.
#
# License: BSD-3-Clause-LBNL

cmake_minimum_required(VERSION 3.20)
project(c3circuit CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Strict per-operation double rounding: no FMA contraction (a fused
# a*b+c rounds once, diverging from the Python reference's semantics --
# same-precision discipline beats the negligible speed win).
add_compile_options(-ffp-contract=off)
# host tuning is available but measured ~neutral for the dense LU here
option(C3_MARCH_NATIVE "compile with -march=native" OFF)
if(C3_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()

add_library(c3core STATIC
    src/C3BankNetwork.cpp
    src/C3Config.cpp
    src/C3ExternalCircuit.cpp
    src/C3Stepper.cpp
    src/ReferenceCircuit.cpp
    src/DenseLU.cpp
    src/YamlLite.cpp
)
target_include_directories(c3core PUBLIC src vendor)
set_target_properties(c3core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(c3core PRIVATE -Wall -Wextra)

add_library(c3circuit SHARED src/PluginFactory.cpp)
target_link_libraries(c3circuit PRIVATE c3core)

add_executable(c3_harness harness/c3_harness.cpp)
target_link_libraries(c3_harness PRIVATE c3core ${CMAKE_DL_LIBS})
target_compile_options(c3_harness PRIVATE -Wall -Wextra)
