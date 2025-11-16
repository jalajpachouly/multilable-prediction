"""
Simple launcher for configurable main experiments
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from configurable_main import main

if __name__ == "__main__":
    main()
