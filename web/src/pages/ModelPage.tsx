import { useEffect, useState } from "react";
import { api, type ModelInfo, type Readiness } from "../api";

export default function ModelPage() {
  const [m, setM] = useState<ModelInfo | null>(null);
  const [ready, setReady] = useState<Readiness | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { Promise.all([api.model(), api.readiness()]).then(([model, status]) => { setM(model); setReady(status); }).catch((e) => setErr(String(e.message))); }, []);
  if (err) return <p className="glass note bad">{err}</p>;
  if (!m) return <p className="glass note">Loading…</p>;
  return (
    <div className="glass card">
      <h2>Model readiness</h2>
      <p>System state: <span className={ready?.status === "ready" ? "pill ok" : "pill warn"}>{ready?.status ?? "unknown"}</span></p>
      <p>Isolated recogniser: <span className={m.word_model_loaded ? "pill ok" : "pill bad"}>{m.word_model_loaded ? `loaded, ${m.labels} classes` : "not loaded"}</span></p>
      {ready?.isolated_recognition.model_version && <p className="muted">Isolated model: {ready.isolated_recognition.model_version}</p>}
      <p className="muted">{m.scope}</p><p>Input: {m.input}</p>
      <p>Continuous translation: <span className={ready?.continuous_translation.available ? "pill ok" : "pill warn"}>{ready?.continuous_translation.available ? "ready" : "not ready"}</span> <span className="muted">{ready?.continuous_translation.reason ?? m.sentence_translation.reason}</span></p>
      <p className="muted">The isolated model score is raw softmax until a validation-set calibration artifact is attached.</p>
      <div className="chips">{m.classes.map((c) => <span key={c} className="chip">{c}</span>)}</div>
    </div>
  );
}
