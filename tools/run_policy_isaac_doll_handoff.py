#!/usr/bin/env python3
"""Generic entry point for the camera-identical Policy-A/Policy-B Isaac runner."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("run_policy_b_isaac_doll_handoff.py")),
    run_name="__main__",
)
