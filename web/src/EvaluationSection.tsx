import { useEffect, useMemo, useState } from "react";
import { fetchEvaluation, type EvalPerClass, type EvaluationReport } from "./api";
import "./evaluation.css";

type SortMode = "worst" | "best" | "alpha";
type MatrixMode = "percent" | "count";

const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

function band(acc: number | null): "ok" | "mid" | "low" | "none" {
  if (acc === null) return "none";
  if (acc >= 0.95) return "ok";
  if (acc >= 0.8) return "mid";
  return "low";
}

export default function EvaluationSection() {
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [state, setState] = useState<"loading" | "missing" | "error" | "ready">("loading");
  const [error, setError] = useState("");
  const [sort, setSort] = useState<SortMode>("worst");
  const [mode, setMode] = useState<MatrixMode>("percent");
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  useEffect(() => {
    let alive = true;
    fetchEvaluation()
      .then((r) => {
        if (!alive) return;
        if (r === null) setState("missing");
        else {
          setReport(r);
          setState("ready");
        }
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
        setState("error");
      });
    return () => {
      alive = false;
    };
  }, []);

  const rows: EvalPerClass[] = useMemo(() => {
    if (!report) return [];
    const list = [...report.per_class];
    const acc = (c: EvalPerClass) => (c.accuracy === null ? 2 : c.accuracy);
    if (sort === "alpha") list.sort((a, b) => a.word.toLowerCase().localeCompare(b.word.toLowerCase()));
    else if (sort === "best") list.sort((a, b) => acc(b) - acc(a) || a.word.localeCompare(b.word));
    else list.sort((a, b) => acc(a) - acc(b) || a.word.localeCompare(b.word));
    return list;
  }, [report, sort]);

  if (state === "loading") return <div className="ev-wrap ev-note">Loading evaluation…</div>;
  if (state === "missing")
    return (
      <div className="ev-wrap ev-note">
        Per-class evaluation has not been generated yet. Run <code>python -m ml.evaluation.class_report</code> and
        place the result at <code>models/isl.eval.json</code>.
      </div>
    );
  if (state === "error" || !report) return <div className="ev-wrap ev-note ev-bad">{error || "Could not load evaluation."}</div>;

  const d = report.deployed;
  const labels = report.labels;
  const n = labels.length;
  const cm = report.confusion_matrix;
  const support = cm.map((row) => row.reduce((a, b) => a + b, 0));

  // confusion-matrix geometry
  const CELL = 12;
  const LEFT = 112;
  const TOP = 112;
  const W = LEFT + n * CELL + 6;
  const H = TOP + n * CELL + 6;

  const hoverText = (() => {
    if (!hover) return "Hover a cell to see the exact numbers.";
    const { i, j } = hover;
    const c = cm[i][j];
    const share = support[i] ? (100 * c) / support[i] : 0;
    if (i === j) return `“${labels[i].word}” recognised correctly: ${c} of ${support[i]} test videos (${share.toFixed(0)}%).`;
    return `True “${labels[i].word}” predicted as “${labels[j].word}”: ${c} of ${support[i]} test videos (${share.toFixed(0)}%).`;
  })();

  return (
    <div className="ev-wrap">
      <div className="section-header" style={{ marginTop: "2.2rem" }}>
        <span className="eyebrow">Per-word results · held-out test videos</span>
        <h3 className="section-title" style={{ fontSize: "1.5rem" }}>
          Accuracy by word &amp; confusion matrix
        </h3>
        <p className="section-subtitle">
          Deployed model <code>{d.run}</code> · {d.n_test} test videos · {d.n_classes} words
        </p>
      </div>

      {report.labels_match_deployed_model === false && (
        <div className="ev-banner ev-bad">
          This report was made for a different class list than the model that is loaded. Do not trust these numbers until
          the report is regenerated for the deployed model.
        </div>
      )}
      {report.warnings.map((w) => (
        <div key={w} className="ev-banner ev-warn">
          {w}
        </div>
      ))}

      <div className="ev-cards">
        <div className="card ev-stat">
          <span>Overall accuracy</span>
          <b>{pct(d.overall_accuracy)}</b>
          <small>
            {d.correct} of {d.n_test} videos correct · every video counts equally
          </small>
        </div>
        <div className="card ev-stat">
          <span>Average class accuracy</span>
          <b>{pct(d.average_class_accuracy)}</b>
          <small>mean of the {d.n_classes} per-word accuracies · every word counts equally</small>
        </div>
        <div className="card ev-stat">
          <span>{report.seed_summary.n_seeds}-seed overall mean</span>
          <b>{pct(report.seed_summary.overall_mean)}</b>
          <small>± {(report.seed_summary.overall_std * 100).toFixed(1)} pts between seeds</small>
        </div>
        <div className="card ev-stat">
          <span>Test videos per word</span>
          <b>
            {d.min_videos_per_class}–{d.max_videos_per_class}
          </b>
          <small>very small, so single errors swing a word&apos;s score</small>
        </div>
      </div>

      <div className="card">
        <div className="ev-head">
          <h4>Accuracy for each word</h4>
          <div className="ev-seg" role="group" aria-label="Sort order">
            {(
              [
                ["worst", "Weakest first"],
                ["best", "Strongest first"],
                ["alpha", "A–Z"],
              ] as Array<[SortMode, string]>
            ).map(([k, label]) => (
              <button key={k} className={sort === k ? "on" : ""} onClick={() => setSort(k)}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <p className="ev-sub">
          Bar = share of this word&apos;s test videos recognised correctly (<b>correct/total</b> at the right). The thin line is
          the 95% interval — wide because each word has so few test videos.
        </p>
        <div className="ev-bars">
          {rows.map((c) => (
            <div
              key={c.id}
              className="ev-row"
              title={`${c.word}: ${c.correct}/${c.support} correct` +
                (c.most_confused_with ? ` · most often mistaken for “${c.most_confused_with.word}” (${c.most_confused_with.count}×)` : "") +
                ` · seeds: ${Object.entries(c.seed_accuracy).map(([s, v]) => `${s}=${pct(v, 0)}`).join(", ")}`}
            >
              <span className="ev-name">{c.word}</span>
              <span className="ev-track">
                <span className={`ev-fill ${band(c.accuracy)}`} style={{ width: `${(c.accuracy ?? 0) * 100}%` }} />
                {c.support > 0 && (
                  <span
                    className="ev-ci"
                    style={{ left: `${c.ci_low * 100}%`, width: `${Math.max(0.5, (c.ci_high - c.ci_low) * 100)}%` }}
                  />
                )}
              </span>
              <span className="ev-val">
                {c.support === 0 ? "no test videos" : `${c.correct}/${c.support}`}
                <em>{pct(c.accuracy, 0)}</em>
              </span>
            </div>
          ))}
        </div>
        <div className="ev-legend">
          <span><i className="ok" /> ≥ 95%</span>
          <span><i className="mid" /> 80–95%</span>
          <span><i className="low" /> &lt; 80%</span>
        </div>
      </div>

      <div className="card">
        <div className="ev-head">
          <h4>Confusion matrix</h4>
          <div className="ev-seg" role="group" aria-label="Matrix shading">
            {(
              [
                ["percent", "Share of word"],
                ["count", "Video count"],
              ] as Array<[MatrixMode, string]>
            ).map(([k, label]) => (
              <button key={k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <p className="ev-sub">
          Rows = the word that was actually signed, columns = what the model said. Blue diagonal = correct; red cells off the
          diagonal = mistakes. A perfect model has only the diagonal.
        </p>
        <div className="ev-scroll">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="ev-matrix"
            role="img"
            aria-label={`Confusion matrix of ${n} words`}
            onMouseLeave={() => setHover(null)}
          >
            <rect x={LEFT} y={TOP} width={n * CELL} height={n * CELL} className="ev-grid-bg" />
            {labels.map((l, k) => (
              <g key={l.id}>
                <text x={LEFT - 4} y={TOP + k * CELL + CELL / 2 + 3} textAnchor="end" className="ev-axis">
                  {l.word}
                </text>
                <text
                  transform={`translate(${LEFT + k * CELL + CELL / 2 + 3},${TOP - 4}) rotate(-90)`}
                  textAnchor="start"
                  className="ev-axis"
                >
                  {l.word}
                </text>
              </g>
            ))}
            {cm.map((row, i) =>
              row.map((v, j) => {
                if (v === 0) return null;
                const share = support[i] ? v / support[i] : 0;
                const opacity = mode === "percent" ? 0.18 + 0.82 * share : Math.min(1, 0.3 + 0.2 * v);
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={LEFT + j * CELL + 0.5}
                    y={TOP + i * CELL + 0.5}
                    width={CELL - 1}
                    height={CELL - 1}
                    className={i === j ? "ev-cell-ok" : "ev-cell-err"}
                    fillOpacity={opacity}
                    onMouseEnter={() => setHover({ i, j })}
                  >
                    <title>{`true ${labels[i].word} → predicted ${labels[j].word}: ${v}`}</title>
                  </rect>
                );
              })
            )}
            {hover && (
              <rect
                x={LEFT + hover.j * CELL}
                y={TOP + hover.i * CELL}
                width={CELL}
                height={CELL}
                className="ev-hover"
                pointerEvents="none"
              />
            )}
            <text x={LEFT + (n * CELL) / 2} y={12} textAnchor="middle" className="ev-axis-title">
              predicted word
            </text>
            <text
              transform={`translate(12,${TOP + (n * CELL) / 2}) rotate(-90)`}
              textAnchor="middle"
              className="ev-axis-title"
            >
              true word
            </text>
          </svg>
        </div>
        <p className="ev-hover-text" aria-live="polite">{hoverText}</p>
      </div>

      <div className="ev-two">
        <div className="card">
          <h4>Most frequent mistakes</h4>
          {report.top_confusions.length === 0 ? (
            <p className="ev-sub">No mistakes on the test videos.</p>
          ) : (
            <table className="bench-table">
              <thead>
                <tr>
                  <th>Signed</th>
                  <th>Predicted</th>
                  <th>Videos</th>
                  <th>Of that word</th>
                </tr>
              </thead>
              <tbody>
                {report.top_confusions.map((c) => (
                  <tr key={`${c.true_id}-${c.pred_id}`}>
                    <td>{c.true}</td>
                    <td>{c.pred}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{c.count}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{pct(c.rate, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="card">
          <h4>Each training seed</h4>
          <table className="bench-table">
            <thead>
              <tr>
                <th>Seed</th>
                <th>Overall</th>
                <th>Avg class</th>
              </tr>
            </thead>
            <tbody>
              {report.seeds.map((s) => (
                <tr key={s.seed}>
                  <td>
                    {s.seed}
                    {s.seed === d.seed ? " (deployed)" : ""}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{pct(s.overall_accuracy)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{pct(s.average_class_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="ev-sub" style={{ marginTop: "0.8rem" }}>
            Overall = correct ÷ all videos. Average class = the mean of the per-word accuracies; it differs from overall
            whenever words have different numbers of test videos.
          </p>
        </div>
      </div>

      <div className="alert-banner">
        {report.notes.map((t) => (
          <p key={t} style={{ margin: "0 0 0.4rem" }}>
            {t}
          </p>
        ))}
      </div>
    </div>
  );
}
