# Ensure sibling packages (parser/, utils/) are importable when this
# package is loaded as a submodule inside a parent repo.
import sys as _sys
from pathlib import Path as _Path

_pkg_dir = str(_Path(__file__).resolve().parent)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)
