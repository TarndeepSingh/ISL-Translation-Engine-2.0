# Environment Audit

**Audit date:** 10 November 2026

The inventory below records the environment at the start of implementation. A Python 3.11.9 uv environment was subsequently prepared and verified; see **Post-audit verification** at the end of this document.

| Item | Observed value | Implication |
|---|---|---|
| OS/shell | Windows workstation; PowerShell commands | Windows paths and browser camera localhost rules apply |
| Workspace | `C:\Users\TARNDEEP SINGH\OneDrive\Documents\ChatGPT\ISL` | This is the only project write workspace |
| Python on PATH | 3.14.8 | Existing source requirements do not establish compatibility; do not run ML stack under this interpreter without checking wheels |
| Package manager | `uv` available at `C:\Users\TARNDEEP SINGH\.local\bin\uv.exe` | Use a project venv and committed lock; do not install globally |
| ML/API packages in current interpreter | NumPy, Torch, ONNX Runtime, MediaPipe, OpenCV, FastAPI, Uvicorn, pytest, datasets and sacrebleu all absent | No project runtime/test execution yet |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM | Appropriate for development/small experiments; not assumed sufficient for full continuous RGB training |
| Free C: space | About 32.1 GB of 297.4 GB | Not enough to stage the reported 56.8 GB INCLUDE archive plus extraction; Colab/runtime staging is appropriate |
| System RAM | Not measured; CIM access denied | Must be collected in a permitted environment before performance limits are chosen |
| Colab | No connected runtime/account state visible | Exact Python/GPU/VRAM/disk must be logged when a user starts a permitted training session |
| Git state | Empty repository before ZIP restoration; now 75 restored files untracked, no commit | Do not claim an existing checkout commit; the source archive's embedded commit is recorded in the audit |

## Environment decision

Create an isolated uv-managed project environment pinned to a tested Python version. Start with Python 3.11 as a conservative compatibility target for MediaPipe/PyTorch wheels, then verify the exact Windows and Colab wheel matrix before committing the version. Do not use PATH Python 3.14.8 as evidence that existing requirements install.

Local Windows work is for source editing, small permitted fixture validation, API/UI integration and CPU inference. Use Colab GPU for authorized preprocessing/training when data access and terms are settled. Stage active data in Colab runtime storage; checkpoint/report to Drive. The supplied Colab notebooks are not executed evidence and contain unpinned installs/source overwrite behavior that will be removed.

Performance limits, concurrency, worker count and maximum video size/duration are design settings until measured on representative hardware.

## Post-audit verification

- `uv.lock` is present and declares the Python 3.11–3.12 range. The verified local development environment uses Python 3.11.9; Python 3.12 was added for Colab. It was smoke-tested on 10 Nov 2026 in a CPU-only Linux sandbox (`uv sync --python 3.12 --locked`; torch 2.15.0, OpenCV 4.14.0, MediaPipe 0.10.14, ONNX Runtime 1.31.1, FastAPI 0.162.2; 94 tests passed at that time). That does not cover Colab's GPU/CUDA runtime.
- Runtime imports passed for NumPy 2.4.6, PyTorch 2.15.0+cpu, OpenCV 4.14.0, MediaPipe 0.10.14, FastAPI 0.162.2 and pytest. The `experiments` extra supplies scikit-learn for grouped-CV utilities and their tests.
- All 44 Python tests passed outside the sandbox on Python 3.11. The complete suite uses FastAPI `TestClient`, which blocks in Windows asyncio socket-pair setup when run inside the sandbox.
- `npm run build`, `python -m compileall -q backend isl ml tests` and `docker compose config --quiet` passed.
- Docker Desktop's Linux engine was unavailable, and no Colab runtime was attached. No dataset media or trained artifact was locally present.
- Current Colab runtime documentation lists Python 3.13.18 for 2026.10 and Python 3.12.12 for runtime 2025.10. The latter matches the project's declared range, subject to the Colab package/GPU smoke check.

## Python 3.12 lock and smoke check (10 Nov 2026, Linux sandbox)

- `pyproject.toml` allows `<3.13` but the committed `uv.lock` was still `==3.11.*`; CI's `uv sync --locked` would have failed. `uv lock` regenerated it for `>=3.11,<3.13`. The only package-set change is `httpx2-jsfetch`, gated to Python >=3.12 on Emscripten/WebAssembly (never installed on Linux/Colab/CI); no existing pin changed.
- `uv sync --python 3.11 --locked` and `--python 3.12 --locked` both succeed; the full suite passes on both. `torch.cuda.is_available()` was False (no GPU in the sandbox).
- Not verified: Colab's runtime, its CUDA build of torch and GPU training. Log those with `nvidia-smi` output when you run the notebook.
