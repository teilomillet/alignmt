#!/usr/bin/env python
"""
Run script for the feature interpretation pipeline.

This script provides a simple command-line interface to run the
feature interpretation pipeline.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to allow importing ops
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ops.feature_interpreter.main import main

if __name__ == "__main__":
    main() 