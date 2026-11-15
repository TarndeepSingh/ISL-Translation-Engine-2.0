"""REST API for upload, async jobs, the automated test library, reports and caption export.
Production rule: if no trained model is loaded these endpoints answer 503. They never fabricate predictions."""
import csv, hashlib, json, logging, os, sqlite3, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from ml.inference.subtitles import to_srt, to_txt, to_vtt
from .media import UploadProblem, discard_upload, run_bounded_inference, stage_upload, upload_root

router = APIRouter(prefix="/api")
ROOT = Path(__file__).resolve().parent.parent
TEST_ROOT = Path(os.getenv("ISL_TEST_ROOT", ROOT / "test_videos"))
REPORTS = Path(os.getenv("ISL_REPORTS", ROOT / "reports" / "word_eval"))
UPLOADS = upload_root()
DB = Path(os.getenv("ISL_DB", ROOT / "data" / "jobs.db"))
MAX_BYTES, JOB_TIMEOUT, RETRIES = 200 * 1024 * 1024, 300, 2
JOB_TTL_SECONDS = int(os.getenv("ISL_JOB_TTL_SECONDS", str(7 * 24 * 60 * 60)))
EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
REPORT_FILES = {"confusion_matrix.png", "error_analysis.html", "predictions.csv", "classification_report.json", "eval_summary.json"}
_pool = ThreadPoolExecutor(max_workers=1)


def _pipe(request: Request):
    p = getattr(request.app.state, "pipeline", None)
    if p is None:
        raise HTTPException(503, "No trained model is loaded (models/isl.onnx + isl.labels.json). Train and export one first.")
    return p


def _db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT,created REAL,updated REAL,filename TEXT,result TEXT,error TEXT)")
    cutoff = time.time() - JOB_TTL_SECONDS
    c.execute("DELETE FROM jobs WHERE updated < ? AND status IN ('done','failed')", (cutoff,))
    c.execute("DELETE FROM jobs WHERE updated < ? AND status IN ('queued','running')", (time.time() - JOB_TIMEOUT - 60,))
    c.commit()
    return c


def _library():
    f = TEST_ROOT / "manifests" / "test_manifest.csv"
    return list(csv.DictReader(open(f, encoding="utf-8"))) if f.exists() else []


def _library_path(row):
    root = TEST_ROOT.resolve()
    candidate = (root / row["file"]).resolve()
    try:
        inside = os.path.commonpath((str(root), str(candidate))) == str(root)
    except ValueError:
        inside = False
    if not inside or not candidate.is_file():
        raise HTTPException(404, "sample not found")
    return candidate


async def _save_upload(file: UploadFile):
    try:
        staged = await stage_upload(file, root=UPLOADS)
    except UploadProblem as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc
    return staged["sha256"], staged["path"], staged["filename"]


@router.get("/model")
def model_info(request: Request):
    p = getattr(request.app.state, "pipeline", None)
    return {"word_model_loaded": p is not None, "labels": len(p.predictor.labels) if p else 0,
            "classes": p.predictor.labels if p else [], "input": "32 x 170 landmark features (whole sign)",
            "sentence_translation": {"available": False, "reason": "No continuous translation model package is loaded."},
            "scope": "Isolated-word recognition of the trained vocabulary only. Not general ISL translation."}


@router.get("/test-library")
def test_library():
    return {"items": _library()}


@router.get("/test-library/{sid}/video")
def test_video(sid: str):
    for r in _library():
        if r["id"] == sid:
            return FileResponse(_library_path(r))
    raise HTTPException(404, "sample not found")


@router.post("/test-library/{sid}/predict")
def test_predict(sid: str, request: Request):
    pipe = _pipe(request)
    r = next((x for x in _library() if x["id"] == sid), None)
    if r is None: raise HTTPException(404, "sample not found")
    path = _library_path(r)
    try: res = pipe.predict_video(str(path))
    except Exception as e:
        logging.getLogger(__name__).exception("test-library inference failed")
        raise HTTPException(422, {"code": "INFERENCE_FAILED", "message": "The sample could not be processed."}) from e
    return {**res, "ground_truth": r["label"], "correct": res["label"] == r["label"], "sample": r}


@router.post("/predict")
async def predict_upload(request: Request, file: UploadFile = File(...)):
    pipe = _pipe(request)
    _, path, name = await _save_upload(file)
    try:
        return {**(await run_bounded_inference(pipe.predict_video, str(path))), "filename": name}
    except UploadProblem as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc
    except Exception as e:
        logging.getLogger(__name__).exception("legacy isolated inference failed")
        raise HTTPException(422, {"code": "INFERENCE_FAILED", "message": "The video could not be processed."}) from e
    finally:
        discard_upload(path, UPLOADS)


def _job_run(pipe, jid, path):
    c = _db()
    try:
        c.execute("UPDATE jobs SET status='running',updated=? WHERE id=?", (time.time(), jid)); c.commit()
        err = "inference_failed"
        for attempt in range(RETRIES):
            try:
                res = _pool.submit(pipe.predict_video, str(path)).result(timeout=JOB_TIMEOUT)
                c.execute("UPDATE jobs SET status='done',result=?,error=NULL,updated=? WHERE id=?",
                          (json.dumps(res), time.time(), jid)); c.commit()
                return
            except FutTimeout:
                err = "inference_timeout"
                break
            except Exception:
                logging.getLogger(__name__).exception("background inference failed")
                time.sleep(0.2)
        c.execute("UPDATE jobs SET status='failed',error=?,updated=? WHERE id=?", (err, time.time(), jid)); c.commit()
    finally:
        c.close()
        discard_upload(path, UPLOADS)


@router.post("/jobs")
async def create_job(request: Request, background: BackgroundTasks, file: UploadFile = File(...)):
    pipe = _pipe(request)
    c = _db()
    pending = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
    c.close()
    if pending >= int(os.getenv("ISL_MAX_PENDING_JOBS", "10")):
        raise HTTPException(503, {"code": "JOB_CAPACITY_REACHED", "message": "The job queue is full; retry shortly."})
    jid, path, name = await _save_upload(file)          # id = content hash -> idempotent
    c = _db(); old = c.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()
    if old and old[0] != "failed":
        c.close(); discard_upload(path, UPLOADS)
        return {"id": jid, "status": old[0], "reused": True}
    now = time.time()
    c.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?,NULL,NULL)", (jid, "queued", now, now, name)); c.commit(); c.close()
    background.add_task(_job_run, pipe, jid, path)
    return {"id": jid, "status": "queued", "reused": False}


def _job(jid):
    c = _db()
    r = c.execute("SELECT id,status,created,updated,filename,result,error FROM jobs WHERE id=?", (jid,)).fetchone()
    c.close()
    if not r: raise HTTPException(404, "job not found")
    return {"id": r[0], "status": r[1], "created": r[2], "updated": r[3], "filename": r[4],
            "result": json.loads(r[5]) if r[5] else None, "error": r[6]}


@router.get("/jobs/{jid}")
def job_status(jid: str):
    return _job(jid)


@router.get("/jobs/{jid}/export")
def job_export(jid: str, format: str = "txt"):
    j = _job(jid)
    if j["status"] != "done": raise HTTPException(409, f"job is {j['status']}")
    res = j["result"]
    segs = [{"start": 0.0, "end": max(res["duration_s"], 0.5), "text": res["label"]}] if res["status"] == "confident" else []
    if not segs: raise HTTPException(409, f"no confident prediction to export (status: {res['status']})")
    fmt = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt}.get(format)
    if fmt is None: raise HTTPException(400, "format must be srt, vtt or txt")
    return PlainTextResponse(fmt(segs), headers={"Content-Disposition": f'attachment; filename="{jid}.{format}"'})


@router.get("/reports/summary")
def report_summary():
    f = REPORTS / "eval_summary.json"
    if not f.exists(): raise HTTPException(404, "no evaluation has been run yet (python -m ml.evaluation.word_eval)")
    return json.loads(f.read_text())


@router.get("/reports/file/{name}")
def report_file(name: str):
    if name not in REPORT_FILES or not (REPORTS / name).exists(): raise HTTPException(404, "report file not found")
    return FileResponse(REPORTS / name)


@router.post("/translate")
def translate():
    raise HTTPException(503, {"code": "MODEL_NOT_READY", "message": "Continuous translation model is not loaded."})
