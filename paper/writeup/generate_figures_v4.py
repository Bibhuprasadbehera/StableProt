#!/usr/bin/env python3
"""Backward-compatible entry point → figures/generate_main.py (Figures 1–6 only).

Supplementary figures:
  python paper/writeup/figures/generate_supplement.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "figures" / "generate_main.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(SCRIPT)] + sys.argv[1:]))
