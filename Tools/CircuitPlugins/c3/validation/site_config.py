#!/usr/bin/env python3
"""Site configuration for the c3 plugin validation scripts.

The validation cases are built from a formation-deck tree and a
machine-profile tree that live in SEPARATE, PRIVATE repositories.  Their
locations, python module names and asset filenames are site-specific, so
they are supplied from OUTSIDE this repository through a small JSON file
named by an environment variable:

    export C3_VALIDATION_SITE=~/.config/c3_validation_site.json

    {
      "deck_dir":       "<abs path to the formation deck directory>",
      "machine_dir":    "<abs path to the machine profile directory>",
      "circuit_module": "<name of the act-bank circuit python module>",
      "seg_mutual":     "<name of the segment-mutual function exported by
                         c3_bank_source>",
      "ref_shot_yaml":  "<circuit yaml filename for the reference shot>",
      "alt_case_yaml":  "<circuit yaml filename for the alternate case>",
      "alt_case_cache": "<directory holding the cached c2 M_ff npz>"
    }

Keeping these out of the tree is deliberate: this repository is a fork of
an open-source project and must not carry machine, shot or deck
identities.  See tools/check_code_names.sh.

License: BSD-3-Clause-LBNL
"""

import json
import os
import pathlib

REQUIRED = ("deck_dir", "machine_dir", "circuit_module", "seg_mutual",
            "ref_shot_yaml", "alt_case_yaml", "alt_case_cache")

_ENV = "C3_VALIDATION_SITE"


def load_site():
    """Return the site configuration dict, or exit with a usable message."""
    path = os.environ.get(_ENV)
    if not path:
        raise SystemExit(
            f"{_ENV} is not set.\n"
            "Point it at a JSON file describing the local deck and machine\n"
            "profile locations -- see the header of this file, or\n"
            "Tools/CircuitPlugins/c3/README.md.")
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        raise SystemExit(f"{_ENV} does not name a readable file: {p}")
    site = json.loads(p.read_text())
    missing = [k for k in REQUIRED if k not in site]
    if missing:
        raise SystemExit(f"{_ENV} file {p} is missing keys: {missing}")
    return site
