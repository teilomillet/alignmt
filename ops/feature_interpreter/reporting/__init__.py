"""
Report generation package.

This package contains modules for generating reports in different formats.
"""

from .markdown_report import generate_markdown_report
from .report_generator import generate_report

__all__ = ['generate_markdown_report', 'generate_report'] 