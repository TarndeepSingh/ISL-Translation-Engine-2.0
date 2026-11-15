import { useEffect, useState } from "react";
import { api, type Readiness, type TranslationV1 } from "../api";

export default function Translation() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [result, setResult] = useState<TranslationV1 | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { api.readiness().then(setReadiness).catch((e) => setError(e.message)); }, []);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  const choose = (next: File | null) => {
    if (url) URL.revokeObjectURL(url);
    setFile(next); setUrl(next ? URL.createObjectURL(next) : ""); setResult(null); setError("");
  };
  const run = async () => {
    if (!file) return;
    setBusy(true); setError(""); setResult(null);
    try { setResult(await api.translate(file)); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  return <div className="grid2">
    <section className="glass card">
      <h2>Continuous ISL to English</h2>
      <p className="muted">Upload a complete signed phrase or sentence. This uses a dedicated translation model; isolated word predictions are not concatenated.</p>
      <input type="file" accept="video/*" aria-label="Continuous signing video" onChange={(e) => choose(e.target.files?.[0] ?? null)} />
      {url && <video src={url} controls className="vid" />}
      <button className="btn" disabled={!file || busy || !readiness?.continuous_translation.available} onClick={run}>
        {busy ? "Translating…" : "Translate video"}
      </button>
      {!readiness?.continuous_translation.available && <p className="glass note warn" role="status">
        Translation is not ready. {readiness?.continuous_translation.reason ?? "Checking model readiness…"}
        {" "}A real paired corpus with reviewed usage terms and a trained continuous model are required.
      </p>}
    </section>
    <section aria-live="polite">
      {error && <p className="glass note bad">{error}</p>}
      {result && <div className="glass card"><h3>Generated English</h3><p>{result.generated_text || "(no text generated)"}</p>
        <p className="muted">{result.model_version} · {result.processing_ms} ms</p></div>}
      {busy && <p className="glass note">Processing the full clip…</p>}
    </section>
  </div>;
}
