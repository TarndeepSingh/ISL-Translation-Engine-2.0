import { useEffect, useMemo, useState } from "react";
import { api, type Pred, type Sample } from "../api";
import Result from "./Result";

export default function TestLibrary() {
  const [items, setItems] = useState<Sample[] | null>(null);
  const [err, setErr] = useState("");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<Sample | null>(null);
  const [res, setRes] = useState<Pred | null>(null);
  const [busy, setBusy] = useState(false);
  const [tally, setTally] = useState("");
  useEffect(() => { api.library().then((d) => setItems(d.items)).catch((e) => setErr(String(e.message))); }, []);
  const shown = useMemo(() => (items ?? []).filter((s) => s.label.includes(q.toLowerCase()) || s.id.includes(q)), [items, q]);

  const run = async (s: Sample) => { setBusy(true); setErr(""); try { setRes(await api.testPredict(s.id)); } catch (e) { setErr((e as Error).message); setRes(null); } finally { setBusy(false); } };
  const runMany = async () => {
    setBusy(true); setErr(""); let ok = 0, n = 0;
    try { for (const s of shown.slice(0, 20)) { const r = await api.testPredict(s.id); n++; if (r.correct) ok++; setTally(`${ok}/${n} correct so far`); } }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  if (items === null && !err) return <p className="glass note">Loading test library…</p>;
  if (items && items.length === 0) return <p className="glass note">The test library is empty. Build it with <code>python -m ml.test_library.build --manifest data/unified_manifest.csv</code>.</p>;
  return (
    <div className="grid2">
      <section className="glass card">
        <h2>Held-out dataset videos</h2>
        <input aria-label="Filter" placeholder="Filter by word or id" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn" disabled={busy || shown.length === 0} onClick={runMany}>Run first {Math.min(20, shown.length)} shown</button>
        {tally && <p className="muted">{tally}</p>}
        <ul className="list">{shown.map((s) => (
          <li key={s.id}><button className={sel?.id === s.id ? "item on" : "item"} onClick={() => { setSel(s); setRes(null); setErr(""); }}>
            <b>{s.label}</b> <span className="muted">{s.id} · {s.session_id || "no session"}</span></button></li>))}</ul>
        {shown.length === 0 && <p className="muted">No match.</p>}
      </section>
      <section>
        {err && <p className="glass note bad">{err}</p>}
        {!sel ? <p className="glass note">Select a video to preview and test it.</p> : (
          <div className="glass card">
            <video key={sel.id} src={`/api/test-library/${sel.id}/video`} controls className="vid" />
            <p>Ground truth: <b>{sel.label}</b> <span className="muted">({sel.dataset}, split {sel.split})</span></p>
            <button className="btn" disabled={busy} onClick={() => run(sel)}>{busy ? "Running…" : "Run inference"}</button>
          </div>)}
        {res && <Result r={res} />}
      </section>
    </div>
  );
}
