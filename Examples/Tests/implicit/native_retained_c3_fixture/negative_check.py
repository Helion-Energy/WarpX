import re

_EXPECTED = {
    "provider": "Eulerian Darwin requires the selected native pre-accept or retained post-accept capability",
    "combined_vacuum": "Native vacuum/circuit endpoint initialization is available; a live step requires retained endpoint acceptance",
}


def negative_pass(text, returncode, kind):
    # WarpX wraps long assertion messages onto lines prefixed with '#'.
    normalized = " ".join(re.sub(r"\n#\s*", "\n", text).split())
    return (
        returncode != 0
        and _EXPECTED[kind] in normalized
        and all(
            marker not in text
            for marker in (
                "Eulerian accepted energy [J]:",
                "Device circuit accepted:",
                "RETAINED_MIDPOINT_DIAGNOSTIC",
                "RETAINED_C3_CLOSED_CANDIDATE",
            )
        )
    )
