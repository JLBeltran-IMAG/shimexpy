#!/usr/bin/env python3
"""
Morphostructural Analysis Tool
Main entry point for the morphostructural analysis functionality.
"""
import sys
#from pathlib import Path

# Import the main function from morphostructural module
from src.post_shi.morphostructural import main

if __name__ == "__main__":
    sys.exit(main())
