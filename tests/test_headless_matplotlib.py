"""Regression test: Colab's MPLBACKEND must not break matplotlib / MediaPipe imports.

Each check runs in a FRESH interpreter, because the failure happens at ``import matplotlib`` time and
an already-imported matplotlib in the pytest process would hide it.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

COLAB_BACKEND = "module://matplotlib_inline.backend_inline"


def run(code: str, backend: str | None):
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    if backend is not None:
        env["MPLBACKEND"] = backend
    return subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("code", [
    "import isl, matplotlib.pyplot as p; print(p.get_backend())",
    "import ml.evaluation.word_eval, matplotlib.pyplot as p; print(p.get_backend())",
])
def test_colab_inline_backend_is_replaced_by_agg(code):
    result = run(code, COLAB_BACKEND)
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().lower() == "agg"


def test_mediapipe_import_chain_survives_colab_backend():
    result = run("import isl.landmarks; print('ok')", COLAB_BACKEND)
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().endswith("ok")


@pytest.mark.parametrize("backend", ["Agg", "svg", None])
def test_other_backend_settings_are_left_alone(backend):
    result = run("import isl, os; print(os.environ.get('MPLBACKEND', 'unset'))", backend)
    assert result.returncode == 0
    assert result.stdout.strip() == (backend if backend is not None else "unset")
