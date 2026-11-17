export interface TopPrediction {
  rank: number;
  class_id: number;
  label: string;
  raw_label: string;
  confidence: number;
}

export interface PredictionResponse {
  status: "ok" | "low_confidence" | "no_hands" | string;
  predicted_word: string;
  raw_top1_word: string;
  confidence: number;
  threshold: number;
  is_confident: boolean;
  motion_energy: number;
  top_predictions: TopPrediction[];
  frames_processed: number;
  latency_ms: number;
  model_variant: string;
  temperature: number;
}

export interface VocabularyItem {
  id: number;
  word: string;
  raw: string;
}

export interface ModelMetaResponse {
  project: string;
  dataset: string;
  dataset_license: string;
  model_Loaded: boolean;
  architecture: string;
  experiment_run: string;
  benchmarks: {
    split_policy: string;
    variants: Array<{ name: string; top1_mean: number; top3_mean: number }>;
    note: string;
  };
  input_shape: number[];
  calibration: {
    temperature?: number;
    confidence_threshold?: number;
    top_k?: number;
  };
  vocabulary_count: number;
  vocabulary: VocabularyItem[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  vocabulary_size: number;
  onnx_path_exists: boolean;
  labels_path_exists: boolean;
  calibration_path_exists: boolean;
  error: string | null;
}

export interface EvalPerClass {
  id: number;
  raw: string;
  word: string;
  support: number;
  correct: number;
  accuracy: number | null;
  ci_low: number;
  ci_high: number;
  most_confused_with: { word: string; id: number; count: number } | null;
  seed_accuracy: Record<string, number | null>;
  seed_mean: number | null;
  seed_std: number;
}

export interface EvalConfusion {
  true: string;
  pred: string;
  true_id: number;
  pred_id: number;
  count: number;
  rate: number;
}

export interface EvalSeed {
  seed: number;
  n_test: number;
  overall_accuracy: number;
  average_class_accuracy: number;
  reported_top1: number | null;
  reported_top3: number | null;
  same_test_videos: boolean;
}

export interface EvaluationReport {
  schema_version: string;
  generated_at: string;
  split: string;
  policy: string;
  variant: string;
  deployed: {
    run: string;
    seed: number;
    n_test: number;
    n_classes: number;
    correct: number;
    overall_accuracy: number;
    average_class_accuracy: number;
    reported_top1: number | null;
    reported_top3: number | null;
    min_videos_per_class: number;
    max_videos_per_class: number;
  };
  labels: Array<{ id: number; raw: string; word: string }>;
  per_class: EvalPerClass[];
  confusion_matrix: number[][];
  top_confusions: EvalConfusion[];
  seeds: EvalSeed[];
  seed_summary: { n_seeds: number; overall_mean: number; overall_std: number; average_mean: number; average_std: number };
  warnings: string[];
  notes: string[];
  /** true = report matches the loaded ONNX model, false = it does not, null = model not loaded so unknown */
  labels_match_deployed_model: boolean | null;
}

export interface FrameLandmarkSnapshot {
  pose: number[][] | null; // 33 x [x, y, z] or null when not detected
  hands: number[][][];     // 0..2 detected hands, each 21 x [x, y, z]
}

const metaEnv = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
const API_BASE = (metaEnv?.VITE_API_BASE_URL || "").replace(/\/$/, "");

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/v1/health`);
  if (!res.ok) {
    throw new Error(`Backend health check failed (${res.status})`);
  }
  return res.json();
}

export async function fetchModelMeta(): Promise<ModelMetaResponse> {
  const res = await fetch(`${API_BASE}/api/v1/meta`);
  if (!res.ok) {
    throw new Error(`Failed to load model metadata (${res.status})`);
  }
  return res.json();
}

export async function predictFramesBatch(
  framesBase64: string[],
  landmarksSequence?: FrameLandmarkSnapshot[]
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/api/v1/predict-frames`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frames: framesBase64,
      landmarks: landmarksSequence || [],
    }),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.detail || `Prediction request failed (${res.status})`);
  }
  return res.json();
}

export async function predictVideoFile(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/predict-video`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.detail || `Video upload prediction failed (${res.status})`);
  }
  return res.json();
}

/** Returns null when the backend has no evaluation report yet (HTTP 404). */
export async function fetchEvaluation(): Promise<EvaluationReport | null> {
  const res = await fetch(`${API_BASE}/api/v1/evaluation`);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`Failed to load evaluation report (${res.status})`);
  }
  return res.json();
}
