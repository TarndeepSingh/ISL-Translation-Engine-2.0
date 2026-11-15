import type { Pred } from "../api";
export default function Result({ r }: { r: Pred }) {
  return (
    <div className="glass card">
      <div className="row">
        <h3>{r.label}</h3>
        {r.correct !== undefined && <span className={r.correct ? "pill ok" : "pill bad"}>{r.correct ? "correct" : "incorrect"}</span>}
        <span className={r.status === "confident" ? "pill ok" : "pill warn"}>{r.status.replace("_", " ")}</span>
      </div>
      {r.ground_truth !== undefined && <p className="muted">Ground truth: <b>{r.ground_truth}</b></p>}
      <div className="bar-bg" aria-label={`confidence ${(r.conf * 100).toFixed(0)}%`}><div className="bar-fg" style={{ width: `${r.conf * 100}%` }} /></div>
      <p>{r.score_semantics === "temperature_scaled_validation_calibrated" ? "Validation-calibrated score" : "Raw model score (uncalibrated)"} <b>{(r.conf * 100).toFixed(1)}%</b> · margin {(r.margin * 100).toFixed(1)}% · {r.ms} ms · {r.duration_s}s video</p>
      {r.model_version && <p className="muted">Model version: {r.model_version}</p>}
      <ol className="top">{r.top.map(([l, p]) => <li key={l}>{l} <span className="muted">{(p * 100).toFixed(1)}%</span></li>)}</ol>
      {r.status === "poor_input" && <p className="note bad">Tracking quality is low (shoulders or hands not visible), so this prediction should not be trusted.</p>}
    </div>
  );
}
