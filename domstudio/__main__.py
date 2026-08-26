"""Allow ``python -m domstudio`` to launch the desktop application."""

from domstudio.application import main


raise SystemExit(main())
