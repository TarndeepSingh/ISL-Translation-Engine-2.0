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