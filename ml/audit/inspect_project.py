"""READ-ONLY audit. Run where the files live (your PC, or Colab with Drive mounted):
  python -m ml.audit.inspect_project --project . --datasets D:\\data\\INCLUDE50 D:\\data\\other --out docs
Writes docs/CURRENT_STATE_AUDIT.md and docs/DATASET_INVENTORY.md. Never modifies or deletes anything."""
import argparse, importlib.metadata as md, os, platform, shutil, statistics, subprocess, sys
from collections import Counter, defaultdict

SKIP = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "dist"}
VIDEO = (".mov", ".mp4", ".avi", ".mkv", ".webm")
KINDS = {"notebook": (".ipynb",), "checkpoint": (".pt", ".pth", ".ckpt", ".onnx", ".tflite", ".h5", ".keras", ".safetensors"),
         "annotation": (".csv", ".json", ".jsonl", ".tsv"), "python": (".py",), "frontend": (".ts", ".tsx", ".js", ".jsx", ".html")}
DEPS = ("requirements.txt", "pyproject.toml", "package.json", "Dockerfile", "docker-compose.yml")
PKGS = ("numpy", "torch", "onnxruntime", "mediapipe", "opencv-python", "fastapi", "uvicorn", "datasets", "sacrebleu")


def scan_tree(root):
    res = {"kinds": defaultdict(list), "tests": [], "deps": [], "videos": 0, "bytes": 0}
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in fs:
            p, low = os.path.join(d, f), f.lower()
            try: size = os.path.getsize(p)
            except OSError: continue
            res["bytes"] += size
            if low.endswith(VIDEO): res["videos"] += 1
            if f in DEPS: res["deps"].append(p)
            if low.startswith("test_") and low.endswith(".py"): res["tests"].append(p)
            for k, ex in KINDS.items():
                if low.endswith(ex): res["kinds"][k].append((p, size))
    return res


def dataset_inventory(root, check_videos=0):
    per, ext, zero, total = defaultdict(int), Counter(), [], 0
    ann = []
    for d, _, fs in os.walk(root):
        for f in fs:
            p, low = os.path.join(d, f), f.lower()
            if low.endswith(VIDEO):
                per[d] += 1; ext[os.path.splitext(low)[1]] += 1; total += 1
                if os.path.getsize(p) == 0: zero.append(p)
            elif low.endswith((".csv", ".json", ".jsonl", ".tsv")):
                ann.append(p)
    v = sorted(per.values()) or [0]
    inv = {"root": root, "videos": total, "folders_with_videos": len(per), "min": v[0], "median": statistics.median(v),
           "max": v[-1], "ext": dict(ext), "zero_byte": zero, "annotation_files": ann[:30], "corrupt": []}
    if check_videos:
        try:
            import cv2
            for d, _, fs in os.walk(root):
                for f in fs:
                    if f.lower().endswith(VIDEO) and len(inv["corrupt"]) + 1 and check_videos > 0:
                        c = cv2.VideoCapture(os.path.join(d, f)); ok, _ = c.read(); c.release(); check_videos -= 1
                        if not ok: inv["corrupt"].append(os.path.join(d, f))
        except ImportError:
            inv["corrupt"] = ["cv2 not installed: corruption check skipped"]
    return inv


def hardware():
    gpu = "none"
    if shutil.which("nvidia-smi"):
        try: gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip() or "nvidia-smi gave no output"
        except Exception as e: gpu = f"nvidia-smi failed: {e}"
    pk = {}
    for p in PKGS:
        try: pk[p] = md.version(p)
        except md.PackageNotFoundError: pk[p] = "not installed"
    free = shutil.disk_usage(".").free / 1e9
    return {"os": platform.platform(), "python": sys.version.split()[0], "cpus": os.cpu_count(), "disk_free_gb": round(free, 1), "gpu": gpu, "packages": pk}


def render_audit(project, tree, hw):
    L = [f"# Current state audit\n\nProject root: `{project}`  (read-only scan)\n", "## Hardware and environment",
         *[f"- {k}: {v}" for k, v in hw.items() if k != "packages"], "", "## Packages", *[f"- {k}: {v}" for k, v in hw["packages"].items()],
         "", "## What exists", f"- total size: {tree['bytes'] / 1e9:.2f} GB", f"- video files inside project: {tree['videos']}", f"- test files: {len(tree['tests'])}",
         f"- dependency files: {', '.join(os.path.basename(x) for x in tree['deps']) or 'none'}", ""]
    for k, items in tree["kinds"].items():
        L += [f"### {k} ({len(items)})", *[f"- `{p}` ({s / 1e6:.2f} MB)" for p, s in items[:15]], "" if len(items) <= 15 else f"- ... {len(items) - 15} more\n"]
    L += ["## Migration table (fill in after review)", "| Item | KEEP / REFACTOR / REPLACE / REMOVE | Reason |", "|---|---|---|"]
    return "\n".join(L) + "\n"


def render_inventory(invs):
    L = ["# Dataset inventory (local files)\n"]
    for i in invs:
        L += [f"## `{i['root']}`", f"- videos: {i['videos']} in {i['folders_with_videos']} folders", f"- videos per folder: min {i['min']}, median {i['median']}, max {i['max']}",
              f"- extensions: {i['ext']}", f"- zero-byte files: {len(i['zero_byte'])}", f"- corrupt/unreadable (sampled): {len(i['corrupt'])}",
              "- annotation files: " + (", ".join(f"`{a}`" for a in i["annotation_files"][:8]) or "none found"), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="."); ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--out", default="docs"); ap.add_argument("--check-videos", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "CURRENT_STATE_AUDIT.md"), "w", encoding="utf-8").write(render_audit(a.project, scan_tree(a.project), hardware()))
    open(os.path.join(a.out, "DATASET_INVENTORY.md"), "w", encoding="utf-8").write(render_inventory([dataset_inventory(d, a.check_videos) for d in a.datasets]))
    print("wrote", os.path.join(a.out, "CURRENT_STATE_AUDIT.md"), "and DATASET_INVENTORY.md")
