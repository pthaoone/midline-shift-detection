#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizer shortcut for Midline Shift (MLS) Detection & Analysis
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scripts.visualize_midline_shift import main

if __name__ == "__main__":
    main()
