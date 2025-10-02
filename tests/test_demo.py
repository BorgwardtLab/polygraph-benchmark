#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the polygraph_demo.py script to ensure it works properly."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import polygraph_demo
sys.path.insert(0, str(Path(__file__).parent.parent))

import polygraph_demo


def test_demo_main():
    """Test that the main function runs without errors."""
    # This should run the full demo without crashing
    polygraph_demo.main()
