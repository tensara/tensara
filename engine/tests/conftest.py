"""Pytest configuration for the engine test suite.

The engine modules (`utils`, `problem`, ...) live at the `engine/` root and are
imported by bare name (e.g. `import utils`). That resolves when pytest is invoked
from inside `engine/`, but not from the repository root. Prepend the engine root to
`sys.path` here so the tests import cleanly regardless of the working directory.
"""

import pathlib
import sys

ENGINE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))
