"""Allow ``python -m ensemble_gate ...``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
