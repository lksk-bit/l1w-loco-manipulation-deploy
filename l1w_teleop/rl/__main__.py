"""Command-line entry for the independent deployment runtime."""

import sys

from .runtime import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"RL deployment failed: {error}", file=sys.stderr)
        raise SystemExit(1)
