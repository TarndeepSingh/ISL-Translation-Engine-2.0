import { useEffect, useState } from "react";
import { api, type Summary } from "../api";

const pct = (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");
export default function Evaluation() {
  const [s, setS] = useState<Summary | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.summary().then(setS).catch((e) => setErr(String(e.message))); }, []);
  if (err) return <p className="glass note">{err}</p>;
  if (!s) return <p className="glass note">Loading evaluation…</p>;
  const cards: [string, string][] = [["Top-1", pct(s.top1)], ["Top-3", pct(s.top3)], ["Top-5", pct(s.top5)], ["Macro-F1", pct(s.macro_f1)],
    ["ECE (calibration error)", pct(s.ece)], ["Videos scored", String(s.n_scored)], ["Mean time", s.mean_ms ? `${s.mean_ms.toFixed(0)} ms` : "—"]];
  return (
    <div>
      <p className="glass note warn">{s.note}</p>
      <div className="cards">{cards.map(([k, v]) => <div key={k} className="glass card stat"><span className="muted">{k}</span><b>{v}</b></div>)}</div>
      <div className="glass card">
        <h2>Confusion matrix</h2>
        <img className="cm" alt="Confusion matrix" src="/api/reports/file/confusion_matrix.png" />
        <div className="row">{["error_analysis.html", "predictions.csv", "classification_report.json"].map((f) => (
          <a key={f} className="btn alt" href={`/api/reports/file/${f}`} target="_blank" rel="noreferrer">{f}</a>))}</div>
      </div>
    </div>
  );
}
