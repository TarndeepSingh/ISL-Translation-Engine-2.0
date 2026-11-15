import { useEffect, useRef, useState } from "react";
import { api, type Job, type Pred } from "../api";
import Result from "./Result";

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [res, setRes] = useState<Pred | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => { window.clearInterval(timer.current); if (url) URL.revokeObjectURL(url); }, [url]);

  const pick = (f: File | null) => { setFile(f); setRes(null); setJob(null); setErr(""); setUrl(f ? URL.createObjectURL(f) : ""); };
  const predict = async () => { if (!file) return; setBusy(true); setErr(""); try { setRes(await api.predict(file)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); } };
  const startJob = async () => {
    if (!file) return; setBusy(true); setErr(""); setRes(null);
    try {
      const { id } = await api.createJob(file);
      window.clearInterval(timer.current);
      timer.current = window.setInterval(async () => {
        try { const j = await api.job(id); setJob(j); if (j.status === "done" || j.status === "failed") { window.clearInterval(timer.current); setBusy(false); } }
        catch (e) { setErr((e as Error).message); window.clearInterval(timer.current); setBusy(false); }
      }, 1000);
    } catch (e) { setErr((e as Error).message); setBusy(false); }
  };
  return (
    <div className="grid2">
      <section className="glass card">
        <h2>Upload a video</h2>
        <input type="file" accept="video/*" aria-label="Video file" onChange={(e) => pick(e.target.files?.[0] ?? null)} />
        {url && <video src={url} controls className="vid" />}
        <div className="row">
          <button className="btn" disabled={!file || busy} onClick={predict}>Predict now</button>
          <button className="btn alt" disabled={!file || busy} onClick={startJob}>Run as background job</button>
        </div>
        <p className="muted">Works for one isolated sign of the trained vocabulary. Sentences are not supported.</p>
      </section>
      <section>
        {err && <p className="glass note bad">{err}</p>}
        {busy && !res && <p className="glass note">Processing{job ? ` (${job.status})` : ""}…</p>}
        {res && <Result r={res} />}
        {job?.status === "failed" && <p className="glass note bad">Job failed: {job.error}</p>}
        {job?.status === "done" && job.result && (<>
          <Result r={job.result} />
          <div className="glass card row">{(["srt", "vtt", "txt"] as const).map((f) => (
            <a key={f} className="btn alt" href={`/api/jobs/${job.id}/export?format=${f}`} download>{f.toUpperCase()}</a>))}
          </div></>)}
      </section>
    </div>
  );
}
