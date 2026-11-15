"""Build the unified manifest from a local INCLUDE folder tree.
  python -m ml.datasets.build_manifest --root D:\\data\\INCLUDE50 --keep data/keep.txt --official-from-hf --split-mode session --out data/unified_manifest.csv
--split-mode session : whole sessions go to one split (proxy for new-signer testing)  <- use for model selection
--split-mode official: INCLUDE's own split (video-level)                               <- comparison only
Both modes print a leakage report; the official split normally shows shared sessions, which is the evidence for
why random/official test scores overestimate live performance."""
import argparse, json
from . import manifest as M

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--dataset", default="include")
    ap.add_argument("--keep"); ap.add_argument("--official-from-hf", action="store_true")
    ap.add_argument("--split-mode", choices=["session", "official"], default="session")
    ap.add_argument("--gap", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/unified_manifest.csv")
    a = ap.parse_args()
    rows = M.from_isolated_tree(a.root, a.dataset)
    if a.keep: rows = M.filter_keep(rows, set(open(a.keep).read().split("\n")))
    if a.official_from_hf:
        from datasets import load_dataset
        ds, name = load_dataset("ai4bharat/INCLUDE"), {"validation": "val"}
        off = {M.key3(r["video_path"]): name.get(s, s) for s in ds for r in ds[s]}
        for r in rows: r["split_official"] = off.get(M.key3(r["path"]), "")
    M.infer_sessions(rows, a.gap); M.add_fingerprints(rows)
    if a.split_mode == "official":
        if not a.official_from_hf: raise SystemExit("--split-mode official needs --official-from-hf")
        for r in rows: r["split"] = r["split_official"]
        rows = [r for r in rows if r["split"]]
    else:
        M.make_group_split(rows, seed=a.seed)
    M.write_manifest(rows, a.out)
    print(json.dumps(M.summary(rows), indent=1))
    leak = M.check_leakage(rows)
    print("LEAKAGE REPORT:", json.dumps(leak, indent=1) if leak else "clean (no session/file shared across splits)")
