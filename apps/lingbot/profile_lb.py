#!/usr/bin/env python3
"""Compatibility entry point; the profiler lives in scripts/profile_lingbot.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "profile_lingbot.py"),
               run_name="__main__")
