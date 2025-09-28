#!/usr/bin/env python3
"""Backward-compatible wrapper that runs GEPA with the JSON judge config."""

from __future__ import annotations

import sys

from run_gepa_generic import main as generic_main


if __name__ == "__main__":  # pragma: no cover
    argv = ["--config", "experiments.gepa_gemma3.configs.json_judge", *sys.argv[1:]]
    generic_main(argv)
