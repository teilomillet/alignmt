"""
Main entry point for the integrated crosscoder + feature interpretation pipeline.

This module serves as a simple entry point for the integrated pipeline,
delegating to the more modular implementation in the CLI module.
"""

import sys

from .cli import main

if __name__ == "__main__":
    main()
    sys.exit(0) 