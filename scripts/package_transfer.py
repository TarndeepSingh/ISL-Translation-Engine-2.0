"""Create a source-only ZIP for transferring the project to another coding agent."""
from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ISL_Translation_Engine_2.0_for_Claude.zip"
SPEC = Path.home() / "Downloads" / "ISL_Translation_Engine_2_0_Complete_50_Page_Codex_Specification.md"
EXCLUDED_DIRS = {
    ".git", ".venv", ".uv-python", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".next", "dist", "build", "runs", "cache", ".uv-cache", "reports", "test_videos",
}
EXCLUDED_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".npz", ".npy", ".pt", ".pth",
    ".onnx", ".task", ".pkl", ".joblib", ".h5", ".keras", ".db", ".sqlite", ".sqlite3",
    ".pem", ".key", ".p12", ".pfx", ".log",
}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.startswith(".venv-") for part in rel.parts):
        return False
    if rel.parts[0] == "data" and rel.as_posix() not in {"data/.gitkeep", "data/manifests/README.md"}:
        return False
    if path == OUTPUT or path.suffix.lower() in EXCLUDED_SUFFIXES or path.suffix.lower() == ".zip":
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    return True


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and include(path))
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, Path(ROOT.name) / path.relative_to(ROOT))
        if SPEC.is_file():
            bundle.write(SPEC, Path(ROOT.name) / "project_inputs" / SPEC.name)
    size_mib = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Created {OUTPUT}")
    print(f"Included {len(files)} repository files" + (" and the project specification" if SPEC.is_file() else ""))
    print(f"Archive size: {size_mib:.2f} MiB")


if __name__ == "__main__":
    main()
