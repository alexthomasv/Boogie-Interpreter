"""Compatibility package for the archived Python interpreter runtime.

The active interpreter package moved the legacy runtime under
``interpreter.archive.legacy_python.runtime.python``.  Some verification and
Anvil utilities still import ``interpreter.python.*`` directly, so keep this
package as a narrow import-path shim.
"""

from pathlib import Path

_legacy_runtime = (
    Path(__file__).resolve().parents[1]
    / "archive"
    / "legacy_python"
    / "runtime"
    / "python"
)
if _legacy_runtime.is_dir():
    __path__.append(str(_legacy_runtime))
