"""Isolated sign recognition package.

Importing this package makes matplotlib headless-safe. Notebook hosts such as Colab export
``MPLBACKEND=module://matplotlib_inline.backend_inline`` into every child process; the project's own
virtual environment does not contain that backend, so ``import matplotlib`` -- which MediaPipe does
internally -- would raise ``ValueError: Key backend ... is not a valid value``. This project only
writes plot files, so any ``module://`` (notebook) backend is replaced by ``Agg`` before matplotlib loads.
"""
import os as _os

if _os.environ.get("MPLBACKEND", "").startswith("module://"):
    _os.environ["MPLBACKEND"] = "Agg"
