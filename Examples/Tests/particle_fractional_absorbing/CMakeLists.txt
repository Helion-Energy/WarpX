# Native particle trajectories are checked in the PICMI driver; no RNG checksum.
if(WarpX_PYTHON)
    foreach(fraction IN ITEMS 0 0.5 1)
        string(REPLACE "." "p" suffix "${fraction}")
        add_warpx_test(
            test_rz_fractional_absorbing_${suffix} RZ 1
            "inputs_test_fractional_absorbing.py --fraction ${fraction}"
            OFF OFF OFF)
        if(WarpX_EB)
            add_warpx_test(
                test_3d_fractional_absorbing_eb_${suffix} 3 1
                "inputs_test_fractional_absorbing.py --dim 3 --wall eb --fraction ${fraction}"
                OFF OFF OFF)
        endif()
    endforeach()
    add_warpx_test(
        test_rz_fractional_absorbing_radial RZ 2
        "inputs_test_fractional_absorbing.py --wall rhi --subcycled --fraction 0.1"
        OFF OFF OFF)
    add_warpx_test(
        test_rz_fractional_absorbing_lower RZ 1
        "inputs_test_fractional_absorbing.py --wall zlo --fraction 0.9"
        OFF OFF OFF)
    add_warpx_test(
        test_rz_fractional_absorbing_hybrid RZ 2
        "inputs_test_fractional_absorbing.py --subcycled --coupled --fraction 0.5"
        OFF OFF OFF)
    add_warpx_test(
        test_rz_fractional_absorbing_corner RZ 1
        "inputs_test_fractional_absorbing.py --wall corner --fraction 0.5"
        OFF OFF OFF)
    add_warpx_test(
        test_3d_fractional_absorbing_domain 3 1
        "inputs_test_fractional_absorbing.py --dim 3 --fraction 0.5"
        OFF OFF OFF)
    add_warpx_test(
        test_rz_fractional_absorbing_repeated RZ 1
        "inputs_test_fractional_absorbing.py --repeat --steps 3 --fraction 0.25"
        OFF OFF OFF)
    if(WarpX_EB)
        add_warpx_test(
            test_3d_fractional_absorbing_subcycled 3 2
            "inputs_test_fractional_absorbing.py --dim 3 --wall eb --subcycled --fraction 0.5"
            OFF OFF OFF)
        add_warpx_test(
            test_3d_fractional_absorbing_curved 3 1
            "inputs_test_fractional_absorbing.py --dim 3 --wall curved --fraction 0.5"
            OFF OFF OFF)
    endif()
endif()
