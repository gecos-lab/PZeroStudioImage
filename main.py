"""Compatibility launcher for DOMStudio 2.

The application implementation lives in the testable :mod:`domstudio` package.
"""

from domstudio.application import main


if __name__ == "__main__":
    raise SystemExit(main())
