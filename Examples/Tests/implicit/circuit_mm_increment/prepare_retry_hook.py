import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
replacements = [
    (
        '#include "NativeEndpointField.H"',
        '#include "NativeEndpointField.H"\nextern void MMCompanionRetryHook(warpx::particles::NativeEndpointCurrentResponse*);\nextern void MMCircuitAfterCurrentIncrementHook();',
    ),
    (
        "        m_mass_matrices_calls_since_deposit = 1;",
        "        m_mass_matrices_calls_since_deposit = 1;\n        if(m_endpoint_current_response)MMCompanionRetryHook(m_endpoint_current_response.get());",
    ),
    (
        "            ComposeMassMatrixCurrentIncrement();",
        "            ComposeMassMatrixCurrentIncrement();\n            MMCircuitAfterCurrentIncrementHook();",
    ),
]
for before, after in replacements:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
Path(sys.argv[2]).write_text(source)
