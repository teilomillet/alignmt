"""
Main entry point for feature-level model difference interpretation.

This module serves as a simple entry point for the feature interpretation
pipeline, delegating to the more modular implementation in the pipeline package.
"""

import sys

from .pipeline.cli import main

if __name__ == "__main__":
    main()
    sys.exit(0) 