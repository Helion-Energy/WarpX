# Native text-input tests exercise the temperature used by the hybrid solver.
foreach(dim IN ITEMS 3 RZ)
    if(dim STREQUAL "RZ")
        set(label rz)
    else()
        set(label 3d)
    endif()
    add_warpx_test(
        test_${label}_temperature_deposition
        ${dim}
        2
        inputs_test_${label}_temperature_deposition
        "analysis.py diags/diag1000002"
        "OFF"
        OFF
    )
endforeach()
