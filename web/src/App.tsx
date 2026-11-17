import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchHealth,
  fetchModelMeta,
  type FrameLandmarkSnapshot,
  type HealthResponse,
  type ModelMetaResponse,
  type PredictionResponse,
  predictFramesBatch,
  predictVideoFile,
  type VocabularyItem,
} from "./api";
import EvaluationSection from "./EvaluationSection";
import "./styles.css";

interface SignInstruction {
  hands: "One-Handed" | "Two-Handed";
  handshape: string;
  startPos: string;
  movement: string;
}

interface PracticeEvaluation {
  status: "pass" | "close" | "fail";
  title: string;
  feedback: string;
  matchedRank: number | null;
  matchedConfidence: number;
}

const SIGN_INSTRUCTIONS: Record<string, SignInstruction> = {
  dog: {
    hands: "One-Handed",
    handshape: "Open palm / slightly curved fingers",
    startPos: "Dominant hand near hip/outer thigh level",
    movement: "Pat your outer thigh/hip twice, then snap or wag index finger as if calling a dog.",
  },
  loud: {
    hands: "Two-Handed",
    handshape: "Index fingers pointing to ears, then opening into spread '5' hands",
    startPos: "Both hands near the ears",
    movement: "Touch near the ears and move both hands outward with a strong, vibrating/expanding motion.",
  },
  car: {
    hands: "Two-Handed",
    handshape: "Both hands in closed fists ('S' handshape)",
    startPos: "Chest level in front of torso, holding an imaginary steering wheel",
    movement: "Move both fists up and down alternately as if turning a car steering wheel.",
  },
  election: {
    hands: "Two-Handed",
    handshape: "Non-dominant hand forms 'O' ballot slot; dominant hand pinches index & thumb",
    startPos: "Chest level",
    movement: "Mime dropping a pinched ballot slip into the circular opening of the other hand.",
  },
  "train ticket": {
    hands: "Two-Handed",
    handshape: "Index & middle fingers extended on both hands, then ticket-punch gesture",
    startPos: "In front of chest",
    movement: "Slide two fingers of dominant hand over non-dominant fingers (train tracks), then mime punching a rectangular ticket.",
  },
  house: {
    hands: "Two-Handed",
    handshape: "Flat open palms ('B' handshape)",
    startPos: "Fingertips touching at eye/forehead level forming a peaked roof",
    movement: "Touch fingertips together to form a triangle roof, then trace straight down the two side walls.",
  },
  death: {
    hands: "Two-Handed",
    handshape: "Flat open palms ('B' handshape), one palm up, one palm down",
    startPos: "In front of torso",
    movement: "Flip both flat hands over simultaneously so the palm-up hand turns palm-down and vice versa.",
  },
  quiet: {
    hands: "Two-Handed",
    handshape: "Index finger on lips, flat palms moving down",
    startPos: "Index finger touching lips, other hand at chest",
    movement: "Place index finger vertically against lips ('shh'), then spread both flat palms smoothly downward and outward.",
  },
  court: {
    hands: "Two-Handed",
    handshape: "Index fingers and thumbs forming 'F' or scale-pan hands",
    startPos: "Shoulder width apart at chest height",
    movement: "Move both hands up and down alternately like the two balancing pans of a justice scale.",
  },
  "store or shop": {
    hands: "Two-Handed",
    handshape: "Flattened 'O' hands (all fingertips touching thumb)",
    startPos: "Chest level, wrists bent upward",
    movement: "Flick or pivot both wrists outward twice as if displaying goods on a counter.",
  },
  window: {
    hands: "Two-Handed",
    handshape: "Flat open palms ('B' handshape), horizontal",
    startPos: "Upper edge of non-dominant palm resting in front of chest",
    movement: "Bring the bottom edge of your dominant flat hand down to tap the top edge of your base hand twice (shutting a window).",
  },
  happy: {
    hands: "Two-Handed",
    handshape: "Flat open palms ('B' handshape)",
    startPos: "Upper chest level",
    movement: "Brush both palms upward over the chest repeatedly (or clap/rub palms lightly with an upward motion).",
  },
  pen: {
    hands: "Two-Handed",
    handshape: "Dominant thumb and index pinched together; non-dominant flat palm up",
    startPos: "In front of chest",
    movement: "Mime writing across the open palm of your non-dominant hand from left to right.",
  },
  bank: {
    hands: "Two-Handed",
    handshape: "Right hand rubs thumb over index/middle fingers (money), then drops into left palm",
    startPos: "Chest level",
    movement: "Perform the money gesture (rubbing thumb and fingertips) and place/stack it into the other palm.",
  },
  hat: {
    hands: "One-Handed",
    handshape: "Curved open hand or pinched brim grip",
    startPos: "Top of the head",
    movement: "Pat the top of your head twice (or mime pulling down the brim of a cap).",
  },
  bird: {
    hands: "Two-Handed",
    handshape: "Index and thumb pinched at mouth (beak), then open palms at sides",
    startPos: "Mouth level, then shoulder level",
    movement: "Pinch index and thumb at mouth like a beak, then flap both open hands at shoulder height like wings.",
  },
  i: {
    hands: "One-Handed",
    handshape: "Extended index finger ('1' handshape)",
    startPos: "In front of chest",
    movement: "Point your dominant index finger inward and tap the center of your chest.",
  },
  paint: {
    hands: "Two-Handed",
    handshape: "Dominant flat flexible hand (paintbrush); non-dominant flat palm",
    startPos: "In front of chest",
    movement: "Brush the fingers of your dominant hand up and down across the back/palm of your non-dominant hand.",
  },
  "t-shirt": {
    hands: "Two-Handed",
    handshape: "Curved fingers or pinched fingertips",
    startPos: "At the outer shoulders / short sleeve line",
    movement: "Touch both shoulders and trace down to the mid-upper arm where a T-shirt sleeve ends.",
  },
  shoes: {
    hands: "Two-Handed",
    handshape: "Closed fists ('S' handshape) or flat hands",
    startPos: "Waist/chest level in front of body",
    movement: "Tap the inner sides of both closed fists together twice (or mime sliding one hand into the other).",
  },
  it: {
    hands: "One-Handed",
    handshape: "Extended index finger",
    startPos: "Chest level",
    movement: "Point your dominant index finger forward and slightly to the side toward a neutral reference point.",
  },
  "you (plural)": {
    hands: "One-Handed",
    handshape: "Extended index finger",
    startPos: "Chest height, pointing forward-left",
    movement: "Point forward with index finger and sweep it in a horizontal arc from left to right across the front.",
  },
  red: {
    hands: "One-Handed",
    handshape: "Extended index finger",
    startPos: "Touching the lower lip",
    movement: "Touch the tip of your index finger to your lower lip and brush it downward twice.",
  },
  hello: {
    hands: "One-Handed",
    handshape: "Flat open hand ('B' handshape), fingers together",
    startPos: "Index fingertip touching the side of the forehead/temple",
    movement: "Move the flat hand outward and slightly upward from the temple in a friendly salute/wave.",
  },
  cow: {
    hands: "Two-Handed",
    handshape: "Thumbs touching temples with pinky/index curved upward ('Y' or horn shape)",
    startPos: "Both sides of the forehead/temples",
    movement: "Place thumbs at temples and curve both hands upward and outward like cow horns.",
  },
  "good morning": {
    hands: "Two-Handed",
    handshape: "Thumbs-up ('good'), then flat hand rising from crook of other arm ('morning')",
    startPos: "Chest level",
    movement: "Show thumbs-up for 'Good', then raise your dominant arm up like the rising sun from your horizontal non-dominant forearm.",
  },
  fan: {
    hands: "One-Handed",
    handshape: "Extended index finger pointing upward",
    startPos: "Above head level",
    movement: "Hold hand above head and spin the index finger/hand in horizontal circles like a ceiling fan.",
  },
  black: {
    hands: "One-Handed",
    handshape: "Extended index finger",
    startPos: "Touching the eyebrow or hair/cheek",
    movement: "Draw the tip of your index finger horizontally across your eyebrow (or point/brush hair).",
  },
  "cell phone": {
    hands: "One-Handed",
    handshape: "Thumb and pinky extended ('Y' handshape) or curved 'C' grip",
    startPos: "Beside the ear and mouth",
    movement: "Hold the 'Y' or phone-grip hand up to your ear as if speaking on a mobile phone.",
  },
  "thank you": {
    hands: "One-Handed",
    handshape: "Flat open hand ('B' handshape), fingers together",
    startPos: "Fingertips touching the chin / lips",
    movement: "Bring fingertips to your chin and move the flat hand forward and slightly downward toward the viewer.",
  },
  white: {
    hands: "One-Handed",
    handshape: "Open 5-finger hand closing into a flattened 'O'",
    startPos: "Center of the chest",
    movement: "Place open fingers on your chest and pull the hand forward while pinching all fingertips together.",
  },
  father: {
    hands: "One-Handed",
    handshape: "Open '5' hand or index/thumb mustache trace",
    startPos: "Forehead / upper lip",
    movement: "Tap the thumb of an open '5' hand to your forehead (or trace a mustache outward above the upper lip).",
  },
  summer: {
    hands: "One-Handed",
    handshape: "Curved index finger ('X' hook) or flat hand fanning",
    startPos: "Across the forehead",
    movement: "Wipe your curved index finger across your forehead from left to right as if wiping away sweat.",
  },
  fall: {
    hands: "Two-Handed",
    handshape: "Dominant 'V' (index & middle fingers) standing on non-dominant flat palm",
    startPos: "Chest level",
    movement: "Stand two fingers on your flat palm and tip them over so the back of the fingers lands on the palm.",
  },
  brother: {
    hands: "Two-Handed",
    handshape: "Mustache/male sign at lip, then both index fingers brought together side-by-side",
    startPos: "Face level, then chest level",
    movement: "Indicate male at upper lip/forehead, then bring both extended index fingers together parallel in front of the chest.",
  },
  monday: {
    hands: "One-Handed",
    handshape: "Three fingers ('M') or closed 'M' handshape",
    startPos: "Chest / shoulder level",
    movement: "Move the 'M' handshape in a small clockwise circle in front of your shoulder.",
  },
  boy: {
    hands: "One-Handed",
    handshape: "Index finger and thumb brushing upper lip (mustache) or cap-brim grab",
    startPos: "Upper lip / forehead",
    movement: "Brush index finger and thumb outward across the upper lip (or pinch imaginary cap brim at forehead).",
  },
  girl: {
    hands: "One-Handed",
    handshape: "Thumb of closed 'A' hand or index finger",
    startPos: "Side of the cheek / jawline",
    movement: "Stroke the thumb downward along the jawline toward the chin (or touch side of nose).",
  },
  year: {
    hands: "Two-Handed",
    handshape: "Both hands in closed 'S' fists",
    startPos: "Dominant fist stacked on top of non-dominant fist at chest",
    movement: "Circle the dominant fist completely around the non-dominant fist once and rest it back on top.",
  },
  long: {
    hands: "Two-Handed",
    handshape: "Non-dominant arm extended; dominant index finger",
    startPos: "Dominant index finger touching non-dominant wrist",
    movement: "Slide your dominant index finger slowly all the way up your non-dominant forearm to the shoulder.",
  },
  short: {
    hands: "Two-Handed",
    handshape: "Flat hands or extended index/middle ('H') fingers",
    startPos: "In front of chest, close together",
    movement: "Hold hands a tiny distance apart (or rub two fingers a short distance along the other fingers).",
  },
  "big large": {
    hands: "Two-Handed",
    handshape: "Curved 'L' or open '5' hands facing each other",
    startPos: "Together in front of the chest",
    movement: "Move both hands wide apart horizontally to show a large expanse.",
  },
  teacher: {
    hands: "Two-Handed",
    handshape: "Flattened 'O' hands at temples, then flat palms moving down sides",
    startPos: "Both temples / forehead",
    movement: "Move flattened 'O' hands forward from both temples (imparting knowledge), then drop flat palms down at sides (person).",
  },
  "small little": {
    hands: "Two-Handed",
    handshape: "Flat open palms facing each other",
    startPos: "Shoulder width apart in front of chest",
    movement: "Push both flat palms inward toward each other until they are only a few centimeters apart.",
  },
  time: {
    hands: "Two-Handed",
    handshape: "Dominant crooked index finger; non-dominant wrist horizontal",
    startPos: "Chest level",
    movement: "Tap the back of your non-dominant wrist twice with your dominant index finger (pointing to a wristwatch).",
  },
  hot: {
    hands: "One-Handed",
    handshape: "Claw/curved 'C' hand",
    startPos: "In front of the mouth / forehead",
    movement: "Place claw hand near mouth and turn the wrist sharply outward and away as if expelling heat.",
  },
  priest: {
    hands: "Two-Handed",
    handshape: "Fingertips applying tilak on forehead, or prayer gesture",
    startPos: "Forehead / chest",
    movement: "Touch forehead horizontally (tilak mark) and bring palms together in namaste/blessing.",
  },
  new: {
    hands: "Two-Handed",
    handshape: "Curved open hands",
    startPos: "Non-dominant palm up; dominant hand brushing across it",
    movement: "Scoop the back of your dominant curved hand through the upward palm of your non-dominant hand.",
  },
  good: {
    hands: "One-Handed",
    handshape: "Thumbs-up ('A' hand with thumb extended upward)",
    startPos: "In front of chest",
    movement: "Hold a clear thumbs-up in front of your chest and push it slightly forward.",
  },
  dry: {
    hands: "One-Handed",
    handshape: "Extended index finger bending into a hook ('1' to 'X')",
    startPos: "Side of the chin",
    movement: "Draw the side of your index finger horizontally across your chin while crooking the finger into a hook.",
  },
};

function normalizeWordKey(word: string): string {
  return (word.split(":").pop() || word).trim().toLowerCase();
}

function getInstructionForWord(word: string): SignInstruction {
  const clean = normalizeWordKey(word);
  if (SIGN_INSTRUCTIONS[clean]) {
    return SIGN_INSTRUCTIONS[clean];
  }
  return {
    hands: "Two-Handed",
    handshape: "Clear open palm / natural articulation",
    startPos: "Center of chest within 16:9 camera frame",
    movement: `Perform the isolated Indian Sign Language gesture for "${word}" smoothly over 2.5 seconds and return hands to rest.`,
  };
}

function evaluatePracticeAttempt(
  targetWord: string,
  res: PredictionResponse,
  instr: SignInstruction
): PracticeEvaluation {
  const targetKey = normalizeWordKey(targetWord);

  if (res.status === "no_hands" || res.confidence <= 0) {
    return {
      status: "fail",
      title: "⚠️ Hands Not Detected During Recording",
      feedback: `Keep ${instr.hands === "Two-Handed" ? "BOTH hands" : "your signing hand"} clearly inside the camera frame for the full 3 seconds.`,
      matchedRank: null,
      matchedConfidence: 0,
    };
  }

  const matchedItem = res.top_predictions.find(
    (p) => normalizeWordKey(p.label) === targetKey || normalizeWordKey(p.raw_label) === targetKey
  );

  if (matchedItem && matchedItem.rank === 1) {
    return {
      status: "pass",
      title: `✅ Correct! You Nailed "${targetWord}"!`,
      feedback: `Great job! The AI recognized "${targetWord}" as its #1 match with ${(matchedItem.confidence * 100).toFixed(1)}% confidence.`,
      matchedRank: 1,
      matchedConfidence: matchedItem.confidence,
    };
  }

  if (matchedItem && (matchedItem.rank === 2 || matchedItem.rank === 3)) {
    return {
      status: "close",
      title: `🟡 Almost There! "${targetWord}" Ranked #${matchedItem.rank}`,
      feedback: `Your gesture was close — "${targetWord}" was #${matchedItem.rank} (${(matchedItem.confidence * 100).toFixed(1)}%), just behind "${res.raw_top1_word}" (${(res.confidence * 100).toFixed(1)}%). Focus on Step 1 (${instr.handshape}) and try once more!`,
      matchedRank: matchedItem.rank,
      matchedConfidence: matchedItem.confidence,
    };
  }

  return {
    status: "fail",
    title: `❌ Not Quite — Model Detected "${res.raw_top1_word}"`,
    feedback: `You practiced "${targetWord}" (${instr.hands}), but the motion looked like "${res.raw_top1_word}" (${(res.confidence * 100).toFixed(1)}%). Check Step 2 (${instr.startPos}) & Step 3 (${instr.movement}) and try again.`,
    matchedRank: null,
    matchedConfidence: 0,
  };
}

const FALLBACK_INCLUDE50_WORDS: string[] = [
  "Dog", "loud", "Car", "Election", "train ticket", "House", "Death", "quiet", "Court", "Store or Shop",
  "Window", "happy", "Pen", "Bank", "Hat", "Bird", "I", "Paint", "T-Shirt", "Shoes",
  "it", "you (plural)", "Red", "Hello", "Cow", "Good Morning", "Fan", "Black", "Cell phone", "Thank you",
  "White", "Father", "Summer", "Fall", "Brother", "Monday", "Boy", "Girl", "Year", "long",
  "short", "big large", "Teacher", "small little", "Time", "hot", "Priest", "new", "good", "dry"
];

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

const POSE_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]
];

export default function App() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [meta, setMeta] = useState<ModelMetaResponse | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [trackerStatus, setTrackerStatus] = useState<string>("Initializing MediaPipe...");
  const [detectedHandsCount, setDetectedHandsCount] = useState<number>(0);
  const [poseLocked, setPoseLocked] = useState<boolean>(false);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [isRecordingSign, setIsRecordingSign] = useState(false);
  const [autoStream, setAutoStream] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<Array<{ word: string; conf: number; time: string }>>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [vocabQuery, setVocabQuery] = useState("");

  // Practice Coach State
  const [selectedWordItem, setSelectedWordItem] = useState<VocabularyItem | null>(null);
  const [practiceVerdict, setPracticeVerdict] = useState<PracticeEvaluation | null>(null);
  const [masteredIds, setMasteredIds] = useState<number[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const autoIntervalRef = useRef<number | null>(null);
  const animFrameRef = useRef<number | null>(null);

  const handLandmarkerRef = useRef<any>(null);
  const poseLandmarkerRef = useRef<any>(null);
  const latestSnapshotRef = useRef<FrameLandmarkSnapshot>({
    pose: null,
    hands: [],
  });
  const showOverlayRef = useRef<boolean>(true);

  useEffect(() => {
    showOverlayRef.current = showOverlay;
  }, [showOverlay]);

  useEffect(() => {
    const checkBackend = () => {
      fetchHealth().then(setHealth).catch(() => {});
      fetchModelMeta().then(setMeta).catch(() => {});
    };
    checkBackend();
    const timer = window.setInterval(checkBackend, 5000);
    return () => {
      window.clearInterval(timer);
      stopCamera();
    };
  }, []);

  async function ensureMediaPipeLoaded() {
    if (handLandmarkerRef.current && poseLandmarkerRef.current) return;
    try {
      setTrackerStatus("Loading MediaPipe Pose Full & Hand Models...");
      const cdnUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";
      const visionModule: any = await new Function("u", "return import(u)")(cdnUrl);
      const { FilesetResolver, HandLandmarker, PoseLandmarker } = visionModule;
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });

      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
      });

      setTrackerStatus("MediaPipe Skeleton Tracker Active");
    } catch (err) {
      console.warn("MediaPipe browser overlay initialization note:", err);
      setTrackerStatus("Camera Active (Server-Side Landmark Mode)");
    }
  }

  function startTrackingLoop() {
    let lastVideoTime = -1;

    const renderLoop = () => {
      const video = videoRef.current;
      const canvas = overlayCanvasRef.current;

      if (video && canvas && video.readyState >= 2) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth || 1280;
          canvas.height = video.videoHeight || 720;
        }
        const ctx = canvas.getContext("2d");
        if (ctx && video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          const nowMs = performance.now();

          ctx.clearRect(0, 0, canvas.width, canvas.height);

          let posePts: number[][] | null = null;
          const handsList: number[][][] = [];

          if (poseLandmarkerRef.current) {
            try {
              const poseRes = poseLandmarkerRef.current.detectForVideo(video, nowMs);
              if (poseRes?.landmarks && poseRes.landmarks.length > 0) {
                const lm = poseRes.landmarks[0];
                setPoseLocked(true);
                posePts = lm.map((p: any) => [p.x, p.y, p.z]);

                if (showOverlayRef.current) {
                  ctx.strokeStyle = "rgba(59, 130, 246, 0.85)";
                  ctx.lineWidth = 3;
                  for (const [i, j] of POSE_CONNECTIONS) {
                    if (lm[i] && lm[j] && (lm[i].visibility ?? 1) > 0.3 && (lm[j].visibility ?? 1) > 0.3) {
                      ctx.beginPath();
                      ctx.moveTo(lm[i].x * canvas.width, lm[i].y * canvas.height);
                      ctx.lineTo(lm[j].x * canvas.width, lm[j].y * canvas.height);
                      ctx.stroke();
                    }
                  }
                  for (let idx = 0; idx < Math.min(lm.length, 25); idx++) {
                    const p = lm[idx];
                    if ((p.visibility ?? 1) > 0.3) {
                      ctx.fillStyle = "#3b82f6";
                      ctx.beginPath();
                      ctx.arc(p.x * canvas.width, p.y * canvas.height, 4.5, 0, 2 * Math.PI);
                      ctx.fill();
                    }
                  }
                }
              } else {
                setPoseLocked(false);
              }
            } catch {
              // Ignore single-frame timestamp collisions
            }
          }

          if (handLandmarkerRef.current) {
            try {
              const handRes = handLandmarkerRef.current.detectForVideo(video, nowMs + 0.1);
              const hands = handRes?.landmarks || [];
              setDetectedHandsCount(hands.length);

              hands.forEach((handLm: any[], hIdx: number) => {
                handsList.push(handLm.map((p: any) => [p.x, p.y, p.z]));

                if (showOverlayRef.current) {
                  ctx.strokeStyle = hIdx === 0 ? "#10b981" : "#0ea5e9";
                  ctx.lineWidth = 3;
                  for (const [a, b] of HAND_CONNECTIONS) {
                    if (handLm[a] && handLm[b]) {
                      ctx.beginPath();
                      ctx.moveTo(handLm[a].x * canvas.width, handLm[a].y * canvas.height);
                      ctx.lineTo(handLm[b].x * canvas.width, handLm[b].y * canvas.height);
                      ctx.stroke();
                    }
                  }

                  handLm.forEach((pt: any, ptIdx: number) => {
                    ctx.fillStyle = ptIdx % 4 === 0 ? "#ffffff" : "#38bdf8";
                    ctx.beginPath();
                    ctx.arc(pt.x * canvas.width, pt.y * canvas.height, ptIdx % 4 === 0 ? 5.5 : 4, 0, 2 * Math.PI);
                    ctx.fill();
                  });
                }
              });
            } catch {
              // Ignore single-frame timestamp collisions
            }
          }

          latestSnapshotRef.current = {
            pose: posePts,
            hands: handsList,
          };
        }
      }

      animFrameRef.current = requestAnimationFrame(renderLoop);
    };

    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    animFrameRef.current = requestAnimationFrame(renderLoop);
  }

  const vocabularyList: VocabularyItem[] = useMemo(() => {
    if (meta?.vocabulary && meta.vocabulary.length > 0) {
      return meta.vocabulary;
    }
    return FALLBACK_INCLUDE50_WORDS.map((w, idx) => ({ id: idx, word: w, raw: w }));
  }, [meta]);

  const filteredVocab = useMemo(() => {
    const q = vocabQuery.trim().toLowerCase();
    if (!q) return vocabularyList;
    return vocabularyList.filter((item) => item.word.toLowerCase().includes(q));
  }, [vocabularyList, vocabQuery]);

  function selectWordForPractice(item: VocabularyItem) {
    setSelectedWordItem(item);
    setPracticeVerdict(null);
  }

  function applyPracticeEvaluation(res: PredictionResponse) {
    if (!selectedWordItem) return;
    const instr = getInstructionForWord(selectedWordItem.word);
    const verdict = evaluatePracticeAttempt(selectedWordItem.word, res, instr);
    setPracticeVerdict(verdict);
    if (verdict.status === "pass") {
      setMasteredIds((prev) => (prev.includes(selectedWordItem.id) ? prev : [...prev, selectedWordItem.id]));
    }
  }

  async function startCamera() {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      await ensureMediaPipeLoaded();
      startTrackingLoop();
    } catch {
      setErrorMsg(
        "Webcam access denied or unavailable. Please allow camera permission in your browser address bar, or test using the Video Clip Upload button."
      );
    }
  }

  function stopCamera() {
    if (autoIntervalRef.current) {
      window.clearInterval(autoIntervalRef.current);
      autoIntervalRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setAutoStream(false);
    setIsRecordingSign(false);
    setDetectedHandsCount(0);
    setPoseLocked(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }

  function captureSingleFrameBase64(): string | null {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;
    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
    }
    const canvas = captureCanvasRef.current;
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.75);
  }

  async function captureAndPredictWindow() {
    if (!cameraActive || isPredicting || isRecordingSign) return;
    setErrorMsg(null);
    setPracticeVerdict(null);
    setIsRecordingSign(true);

    const capturedFrames: string[] = [];
    const capturedLandmarks: FrameLandmarkSnapshot[] = [];
    const totalFrames = 32;
    const intervalMs = 90;

    for (let i = 0; i < totalFrames; i++) {
      const frame = captureSingleFrameBase64();
      if (frame) capturedFrames.push(frame);
      capturedLandmarks.push({
        pose: latestSnapshotRef.current.pose,
        hands: latestSnapshotRef.current.hands,
      });
      await new Promise((r) => setTimeout(r, intervalMs));
    }

    setIsRecordingSign(false);
    if (capturedFrames.length < 4) {
      setErrorMsg("Not enough webcam frames captured. Check that your camera is active.");
      return;
    }

    setIsPredicting(true);
    try {
      const res = await predictFramesBatch(capturedFrames, capturedLandmarks);
      setPrediction(res);
      applyPracticeEvaluation(res);
      if (res.is_confident) {
        setHistory((prev) => [
          {
            word: res.predicted_word,
            conf: res.confidence,
            time: new Date().toLocaleTimeString(),
          },
          ...prev.slice(0, 7),
        ]);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Prediction failed. Verify backend and models/isl.onnx.";
      setErrorMsg(msg);
    } finally {
      setIsPredicting(false);
    }
  }

  useEffect(() => {
    if (autoStream && cameraActive) {
      autoIntervalRef.current = window.setInterval(() => {
        void captureAndPredictWindow();
      }, 3400);
    } else if (autoIntervalRef.current) {
      window.clearInterval(autoIntervalRef.current);
      autoIntervalRef.current = null;
    }
    return () => {
      if (autoIntervalRef.current) window.clearInterval(autoIntervalRef.current);
    };
  }, [autoStream, cameraActive, isPredicting, isRecordingSign]);

  async function handleFileUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    setPracticeVerdict(null);
    setIsPredicting(true);
    try {
      const res = await predictVideoFile(file);
      setPrediction(res);
      applyPracticeEvaluation(res);
      setHistory((prev) => [
        {
          word: res.raw_top1_word,
          conf: res.confidence,
          time: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 7),
      ]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to process uploaded video.";
      setErrorMsg(msg);
    } finally {
      setIsPredicting(false);
      e.target.value = "";
    }
  }

  const activeInstruction = selectedWordItem ? getInstructionForWord(selectedWordItem.word) : null;
  const requiredHands = activeInstruction?.hands === "Two-Handed" ? 2 : 1;
  const handsCheckPassed = detectedHandsCount >= requiredHands;

  return (
    <div>
      <header className="navbar">
        <div className="navbar-inner">
          <a href="#home" className="brand">
            <span className="brand-badge">ISL</span>
            <span>ISL Translation Engine 2.0</span>
          </a>
          <nav className="nav-links">
            <a href="#studio" className="nav-link">Live Studio</a>
            <a href="#pipeline" className="nav-link">How It Works</a>
            <a href="#vocabulary" className="nav-link">50-Word Vocabulary</a>
            <a href="#performance" className="nav-link">Model Performance</a>
            <a href="#stack" className="nav-link">Tech Stack</a>
            <a href="#about" className="nav-link">About & Credits</a>
          </nav>
          <div className="nav-right">
            <button
              className="theme-toggle-btn"
              onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
            >
              {theme === "light" ? "🌙 Night Mode" : "☀️ Day Mode"}
            </button>
            <div className="status-pill">
              <span className={`status-dot ${health?.model_loaded ? "online" : "offline"}`} />
              <span>
                {health?.model_loaded
                  ? `ONNX Engine Ready (${health.vocabulary_size} Classes)`
                  : health?.error
                  ? "Model File Missing in models/"
                  : "Backend Offline (Start Port 8000)"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="container">
        <section id="home" className="hero">
          <div className="hero-grid">
            <div>
              <span className="eyebrow">AI4Bharat INCLUDE-50 · MediaPipe · Bi-LSTM · ONNX</span>
              <h1 className="hero-title">
                Real-Time <span className="gradient-text">Indian Sign Language</span> Gesture-to-Text Recognition
              </h1>
              <p className="hero-lead">
                An accessibility-focused spatial-temporal AI engine that extracts 3D hand and upper-body landmarks from live video, classifies 50 isolated ISL gestures, and coaches your signing accuracy in real time.
              </p>
              <div className="hero-actions">
                <a href="#studio" className="btn btn-primary" onClick={() => !cameraActive && void startCamera()}>
                  Start Translating Live
                </a>
                <a href="#vocabulary" className="btn btn-secondary">
                  Practice 50 Supported Signs ({masteredIds.length}/50 Mastered)
                </a>
              </div>
              <div className="hero-metrics">
                <div className="stat-mini">
                  <div className="stat-mini-val">97.6%</div>
                  <div className="stat-mini-lbl">Top-1 Mean (INCLUDE-50)</div>
                </div>
                <div className="stat-mini">
                  <div className="stat-mini-val">99.8%</div>
                  <div className="stat-mini-lbl">Top-3 Mean (3 Seeds)</div>
                </div>
                <div className="stat-mini">
                  <div className="stat-mini-val">{masteredIds.length}/50</div>
                  <div className="stat-mini-lbl">Signs Mastered in Session</div>
                </div>
              </div>
            </div>

            <div className="hero-visual-card">
              <div style={{ fontSize: "0.8rem", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)", marginBottom: "0.6rem" }}>
                INTERACTIVE AI SIGN COACH
              </div>
              <h3 style={{ fontSize: "1.2rem", marginBottom: "0.8rem" }}>
                Real-Time Posture &amp; Accuracy Verification
              </h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "1.1rem" }}>
                Click any word in the 50-word lexicon to launch Practice Mode. The AI checks your live hand count and shoulder framing before recording, then grades whether your 3-second gesture matches the target sign.
              </p>
              <div className="top3-list">
                <div className="top3-row">
                  <div className="top3-row-header">
                    <span>1. Greetings: Hello</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-emerald)" }}>98.2%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: "98.2%" }} />
                  </div>
                </div>
                <div className="top3-row">
                  <div className="top3-row-header">
                    <span>2. Greetings: Thank you</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>1.4%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: "8%" }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="studio" className="section">
          <div className="section-header">
            <span className="eyebrow">Interactive Studio &amp; Practice Coach</span>
            <h2 className="section-title">Live Sign Recognition</h2>
            <p className="section-subtitle">
              Position your upper body and both hands clearly inside the frame. Select any word from the Vocabulary below to activate <strong>AI Practice Coach Mode</strong>!
            </p>
          </div>

          {/* Pinned Interactive Practice Coach Card */}
          {selectedWordItem && activeInstruction && (
            <div className="sign-guide-card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem" }}>
                <div>
                  <span className="eyebrow" style={{ marginBottom: "0.3rem" }}>
                    Active Practice Target · Word #{selectedWordItem.id + 1} ({activeInstruction.hands})
                  </span>
                  <h3 style={{ fontSize: "1.4rem", fontWeight: 800 }}>
                    Practicing: &ldquo;{selectedWordItem.word}&rdquo;{" "}
                    {masteredIds.includes(selectedWordItem.id) && (
                      <span style={{ color: "var(--accent-emerald)", fontSize: "1rem" }}>✓ Mastered</span>
                    )}
                  </h3>
                </div>
                <div style={{ display: "flex", gap: "0.6rem" }}>
                  <button
                    className="btn btn-primary"
                    style={{ padding: "0.48rem 1rem", fontSize: "0.84rem" }}
                    disabled={isRecordingSign || isPredicting}
                    onClick={() => {
                      if (!cameraActive) {
                        void startCamera();
                      } else {
                        void captureAndPredictWindow();
                      }
                    }}
                  >
                    {!cameraActive
                      ? "1. Enable Camera to Practice"
                      : isRecordingSign
                      ? "Recording Your Sign..."
                      : isPredicting
                      ? "Grading Your Sign..."
                      : `Verify My "${selectedWordItem.word}" Sign Now`}
                  </button>
                  <button
                    className="btn btn-secondary"
                    style={{ padding: "0.48rem 0.9rem", fontSize: "0.82rem" }}
                    onClick={() => {
                      setSelectedWordItem(null);
                      setPracticeVerdict(null);
                    }}
                  >
                    ✕ Exit Practice Mode
                  </button>
                </div>
              </div>

              <div className="sign-guide-grid">
                <div className="sign-guide-step">
                  <div className="sign-guide-step-label">1. Handshape ({activeInstruction.hands})</div>
                  <div style={{ fontSize: "0.9rem" }}>{activeInstruction.handshape}</div>
                </div>
                <div className="sign-guide-step">
                  <div className="sign-guide-step-label">2. Starting Position</div>
                  <div style={{ fontSize: "0.9rem" }}>{activeInstruction.startPos}</div>
                </div>
                <div className="sign-guide-step">
                  <div className="sign-guide-step-label">3. Action / Movement</div>
                  <div style={{ fontSize: "0.9rem" }}>{activeInstruction.movement}</div>
                </div>
              </div>

              {/* Stage 1: Real-time Pre-Flight Posture & Hand Check */}
              {cameraActive && (
                <div className="live-check-bar">
                  <span className={`check-pill ${poseLocked ? "ok" : "warn"}`}>
                    {poseLocked
                      ? "✅ Upper-Body Framing: Locked"
                      : "⚠️ Upper-Body Framing: Step back so shoulders are visible"}
                  </span>
                  <span className={`check-pill ${handsCheckPassed ? "ok" : "warn"}`}>
                    {handsCheckPassed
                      ? `✅ Hand Count Ready: ${detectedHandsCount} visible (${activeInstruction.hands})`
                      : `⚠️ Raise ${requiredHands === 2 ? "BOTH hands" : "your hand"} in frame (${detectedHandsCount}/${requiredHands} visible)`}
                  </span>
                </div>
              )}

              {/* Stage 2: AI Verdict Banner After Recording */}
              {practiceVerdict && (
                <div className={`verdict-banner ${practiceVerdict.status}`}>
                  <div style={{ fontWeight: 800, fontSize: "1.02rem", marginBottom: "0.25rem" }}>
                    {practiceVerdict.title}
                  </div>
                  <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                    {practiceVerdict.feedback}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="studio-grid">
            <div className="card">
              <div className="camera-stage">
                <video ref={videoRef} className="camera-video" playsInline muted />
                <canvas
                  ref={overlayCanvasRef}
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    transform: "scaleX(-1)",
                    pointerEvents: "none",
                  }}
                />
                {!cameraActive && (
                  <div style={{ position: "absolute", textAlign: "center", padding: "1.5rem" }}>
                    <p style={{ color: "#cbd5e1", marginBottom: "1rem" }}>
                      Camera is currently off. Click below to enable your webcam and live skeleton pointers.
                    </p>
                    <button className="btn btn-primary" onClick={() => void startCamera()}>
                      Enable Webcam &amp; Pointers
                    </button>
                  </div>
                )}
                {cameraActive && (
                  <>
                    <div className="camera-badge-overlay">
                      {isRecordingSign ? <span className="pulse-rec" /> : <span className="status-dot online" />}
                      <span>
                        {isRecordingSign
                          ? selectedWordItem
                            ? `Perform "${selectedWordItem.word}" Now (2.9s)...`
                            : "Recording 2.9s Sign Gesture..."
                          : isPredicting
                          ? "Running Bi-LSTM ONNX Model..."
                          : trackerStatus}
                      </span>
                    </div>
                    <div
                      style={{
                        position: "absolute",
                        bottom: "12px",
                        left: "12px",
                        background: "rgba(15, 23, 42, 0.84)",
                        border: "1px solid rgba(255, 255, 255, 0.14)",
                        padding: "0.3rem 0.7rem",
                        borderRadius: "999px",
                        fontSize: "0.75rem",
                        fontFamily: "var(--font-mono)",
                        color: detectedHandsCount > 0 ? "#34d399" : "#cbd5e1",
                      }}
                    >
                      Hands Tracked: {detectedHandsCount} | Body Pose: {poseLocked ? "Locked" : "Searching"}
                    </div>
                  </>
                )}
              </div>

              <div className="controls-bar">
                <div style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap" }}>
                  {!cameraActive ? (
                    <button className="btn btn-primary" onClick={() => void startCamera()}>
                      Start Camera
                    </button>
                  ) : (
                    <>
                      <button
                        className="btn btn-primary"
                        onClick={() => void captureAndPredictWindow()}
                        disabled={isRecordingSign || isPredicting}
                      >
                        {isRecordingSign
                          ? "Capturing Sign..."
                          : isPredicting
                          ? "Predicting..."
                          : selectedWordItem
                          ? `Test "${selectedWordItem.word}" (3s)`
                          : "Record Sign (3s)"}
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => setAutoStream((v) => !v)}
                      >
                        {autoStream ? "Pause Auto-Loop" : "Auto-Loop"}
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => setShowOverlay((v) => !v)}
                      >
                        {showOverlay ? "Hide Pointers" : "Show Pointers"}
                      </button>
                      <button className="btn btn-danger" onClick={stopCamera}>
                        Stop Camera
                      </button>
                    </>
                  )}
                </div>

                <label className="btn btn-secondary" style={{ cursor: "pointer" }}>
                  <span>Upload Video Clip</span>
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime,video/webm,video/x-msvideo"
                    style={{ display: "none" }}
                    onChange={(e) => void handleFileUpload(e)}
                  />
                </label>
              </div>

              {errorMsg && <div className="error-banner">{errorMsg}</div>}
            </div>

            <div className="card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <span style={{ fontSize: "0.8rem", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                  PREDICTION OUTPUT
                </span>
                {prediction && (
                  <span style={{ fontSize: "0.78rem", fontFamily: "var(--font-mono)", color: "var(--accent-cyan)" }}>
                    {prediction.latency_ms} ms · {prediction.frames_processed} frames
                  </span>
                )}
              </div>

              <div className="prediction-hero-box">
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  Recognized ISL Word
                </div>
                <div className="prediction-word">
                  {isPredicting
                    ? "Analyzing..."
                    : prediction
                    ? prediction.predicted_word
                    : "Awaiting Gesture"}
                </div>
                <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                  {prediction ? (
                    <>
                      Top-1 Confidence:{" "}
                      <strong style={{ color: prediction.is_confident ? "var(--accent-emerald)" : "var(--accent-amber)" }}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </strong>{" "}
                      (Threshold: {(prediction.threshold * 100).toFixed(0)}% · Temp: {prediction.temperature})
                    </>
                  ) : (
                    "Perform one of the 50 supported signs or upload a video clip."
                  )}
                </div>
              </div>

              <h4 style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                Top-3 Candidate Predictions
              </h4>
              <div className="top3-list">
                {(prediction?.top_predictions || [
                  { rank: 1, label: "—", confidence: 0, class_id: 0, raw_label: "" },
                  { rank: 2, label: "—", confidence: 0, class_id: 1, raw_label: "" },
                  { rank: 3, label: "—", confidence: 0, class_id: 2, raw_label: "" },
                ]).map((item) => (
                  <div key={item.rank} className="top3-row">
                    <div className="top3-row-header">
                      <span>
                        #{item.rank} {item.label}
                      </span>
                      <span style={{ fontFamily: "var(--font-mono)" }}>
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: `${Math.min(100, Math.max(2, item.confidence * 100))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {history.length > 0 && (
                <div style={{ marginTop: "1.4rem" }}>
                  <h4 style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                    Recent Session Transcript
                  </h4>
                  <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                    {history.map((h, i) => (
                      <span key={i} className="status-pill">
                        <strong>{h.word}</strong>
                        <span style={{ color: "var(--text-muted)" }}>{(h.conf * 100).toFixed(0)}%</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        <section id="pipeline" className="section">
          <div className="section-header">
            <span className="eyebrow">End-to-End Architecture</span>
            <h2 className="section-title">How It Works</h2>
            <p className="section-subtitle">
              Every video clip passes through a deterministic 6-stage spatial-temporal pipeline that matches training and deployment preprocessing.
            </p>
          </div>

          <div className="pipeline-grid">
            {[
              {
                step: "01 / VIDEO INPUT",
                title: "Webcam / Clip Capture",
                desc: "Captures a 2.9-second isolated sign window via WebRTC webcam stream or uploaded video file.",
              },
              {
                step: "02 / FRAME PROCESSING",
                title: "Temporal Resampling",
                desc: "Decodes RGB frames and uniformly resamples variable-length clips onto a fixed 32-timestep sequence.",
              },
              {
                step: "03 / MEDIAPIPE LANDMARKS",
                title: "Skeletal Extraction",
                desc: "Extracts 3D upper-body pose and 21-point left/right hand landmarks (75x3 raw coordinates).",
              },
              {
                step: "04 / INVARIANT FEATURES",
                title: "170-D Normalization",
                desc: "Centers pose on mid-shoulder span and hands on palm width (36 pose + 126 hand + 6 wrist + 2 presence).",
              },
              {
                step: "05 / BI-LSTM + ONNX",
                title: "Sequence Classification",
                desc: "Bidirectional LSTM processes forward and backward motion trajectories via ONNX Runtime CPU execution.",
              },
              {
                step: "06 / CALIBRATED TEXT",
                title: "Top-3 Word & Coach Verdict",
                desc: "Applies temperature scaling, outputs Top-3 probabilities, and verifies if your practice attempt matched.",
              },
            ].map((p) => (
              <div key={p.step} className="pipeline-step">
                <div className="step-num">{p.step}</div>
                <h3 style={{ fontSize: "1.02rem", marginBottom: "0.45rem" }}>{p.title}</h3>
                <p style={{ fontSize: "0.86rem", color: "var(--text-secondary)" }}>{p.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="vocabulary" className="section">
          <div className="section-header">
            <span className="eyebrow">INCLUDE-50 Interactive Lexicon</span>
            <h2 className="section-title">Supported 50-Word Vocabulary (Click Any Word to Practice)</h2>
            <p className="section-subtitle">
              Click any word below to view its step-by-step Indian Sign Language instructions and test yourself on the live camera.
            </p>
          </div>

          {selectedWordItem && activeInstruction && (
            <div className="sign-guide-card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem" }}>
                <div>
                  <span className="eyebrow" style={{ marginBottom: "0.35rem" }}>
                    Word #{selectedWordItem.id + 1} · {activeInstruction.hands} Sign
                  </span>
                  <h3 style={{ fontSize: "1.5rem", fontWeight: 800 }}>
                    {selectedWordItem.word}{" "}
                    {masteredIds.includes(selectedWordItem.id) && (
                      <span style={{ color: "var(--accent-emerald)", fontSize: "1rem" }}>✓ Mastered</span>
                    )}
                  </h3>
                </div>
                <div style={{ display: "flex", gap: "0.6rem" }}>
                  <a
                    href="#studio"
                    className="btn btn-primary"
                    style={{ padding: "0.5rem 1rem", fontSize: "0.84rem" }}
                    onClick={() => !cameraActive && void startCamera()}
                  >
                    Practice &amp; Verify &ldquo;{selectedWordItem.word}&rdquo; on Camera ↑
                  </a>
                  <button
                    className="btn btn-secondary"
                    style={{ padding: "0.5rem 0.9rem", fontSize: "0.84rem" }}
                    onClick={() => {
                      setSelectedWordItem(null);
                      setPracticeVerdict(null);
                    }}
                  >
                    Close
                  </button>
                </div>
              </div>

              <div className="sign-guide-grid">
                <div className="sign-guide-step">
                  <div className="sign-guide-step-label">1. Handshape &amp; Fingers</div>
                  <div style={{ fontSize: "0.92rem" }}>{activeInstruction.handshape}</div>
                </div>
                <div className="sign-guide-step">
                  <div className="sign-guide-step-label">2. Starting Position</div>
                  <div style={{ fontSize: "0.92rem" }}>{activeInstruction.startPos}</div>
                </div>
                <div className="sign-guide-step">
                  <div className="sign-guide-step-label">3. How to Perform</div>
                  <div style={{ fontSize: "0.92rem" }}>{activeInstruction.movement}</div>
                </div>
              </div>
            </div>
          )}

          <div className="vocab-toolbar">
            <input
              type="text"
              className="search-input"
              placeholder="Search supported words (e.g., Hello, Cow, Car, Dog, Red)..."
              value={vocabQuery}
              onChange={(e) => setVocabQuery(e.target.value)}
            />
            <span style={{ color: "var(--text-secondary)", fontSize: "0.88rem" }}>
              Mastered: <strong>{masteredIds.length}</strong> / <strong>{vocabularyList.length}</strong> · <strong>Click a word to practice</strong>
            </span>
          </div>

          <div className="vocab-grid">
            {filteredVocab.map((item) => {
              const isMastered = masteredIds.includes(item.id);
              return (
                <div
                  key={item.id}
                  className={`vocab-chip ${selectedWordItem?.id === item.id ? "active" : ""} ${isMastered ? "mastered" : ""}`}
                  onClick={() => selectWordForPractice(item)}
                  title={`Click to practice "${item.word}"`}
                >
                  <span>
                    {isMastered ? "✓ " : ""}
                    {item.word}
                  </span>
                  <span className="vocab-id">#{item.id + 1}</span>
                </div>
              );
            })}
          </div>
        </section>

        <section id="performance" className="section">
          <div className="section-header">
            <span className="eyebrow">Empirical Evaluation</span>
            <h2 className="section-title">Model Performance &amp; Methodology</h2>
            <p className="section-subtitle">
              Evaluated across three independent random seeds (13, 14, 15) on the 943 verified videos of the AI4Bharat INCLUDE-50 dataset under the balanced split policy.
            </p>
          </div>

          <div className="perf-grid">
            <div className="card">
              <h3 style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>
                Three-Seed Benchmark Comparison (Seeds 13, 14, 15)
              </h3>
              <table className="bench-table">
                <thead>
                  <tr>
                    <th>Experiment Variant</th>
                    <th>Top-1 Mean</th>
                    <th>Top-3 Mean</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Baseline Bi-LSTM</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>96.3%</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>99.7%</td>
                  </tr>
                  <tr>
                    <td>
                      <strong>aug7-warp0.3 (Deployed)</strong>
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-emerald)", fontWeight: 700 }}>
                      97.6%
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-cyan)", fontWeight: 700 }}>
                      99.8%
                    </td>
                  </tr>
                </tbody>
              </table>

              <div className="alert-banner">
                <strong>Important Limitation Disclosure:</strong> The 97.6% Top-1 and 99.8% Top-3 figures measure accuracy on held-out INCLUDE-50 video clips recorded in controlled studio settings. They do not represent guaranteed accuracy on unconstrained real-world webcams for unseen users.
              </div>
            </div>

            <div className="card">
              <h3 style={{ fontSize: "1.1rem", marginBottom: "0.65rem" }}>
                Understanding the Metrics
              </h3>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.85rem" }}>
                <strong>Top-1 Accuracy (97.6%):</strong> The percentage of held-out test videos where the model's single highest-probability prediction exactly matches the ground-truth sign label.
              </p>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.85rem" }}>
                <strong>Top-3 Accuracy (99.8%):</strong> The percentage of test videos where the correct sign appears within the model's top three candidate predictions.
              </p>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                <strong>Augmentation (`aug7-warp0.3`):</strong> Combines spatial coordinate jitter, scaling, rotation, and 30% temporal time-warping.
              </p>
            </div>
          </div>

          <EvaluationSection />
        </section>

        <section id="stack" className="section">
          <div className="section-header">
            <span className="eyebrow">Engineering Stack</span>
            <h2 className="section-title">Technology Stack</h2>
          </div>

          <div className="stack-grid">
            {[
              { name: "Python & FastAPI", role: "Asynchronous REST backend, video decoding, and model serving." },
              { name: "MediaPipe Tasks Vision", role: "Real-time 60 FPS hand and upper-body skeletal landmark extraction." },
              { name: "Bi-LSTM (Deep Learning)", role: "Bidirectional recurrent sequence modeling for temporal gesture dynamics." },
              { name: "ONNX Runtime", role: "Hardware-agnostic, low-latency CPU model execution." },
              { name: "React 18 & TypeScript", role: "Responsive interactive webcam studio and evaluation interface." },
              { name: "Docker", role: "Multi-stage reproducible containerization for cloud deployment." },
            ].map((tech) => (
              <div key={tech.name} className="card">
                <h3 style={{ fontSize: "1rem", marginBottom: "0.4rem" }}>{tech.name}</h3>
                <p style={{ fontSize: "0.86rem", color: "var(--text-secondary)" }}>{tech.role}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="about" className="section" style={{ borderBottom: "none" }}>
          <div className="section-header">
            <span className="eyebrow">Attribution &amp; Governance</span>
            <h2 className="section-title">About, Dataset Credits &amp; Privacy</h2>
          </div>

          <div className="perf-grid">
            <div className="card">
              <h3 style={{ fontSize: "1.08rem", marginBottom: "0.6rem" }}>
                AI4Bharat &amp; IIT Madras INCLUDE Dataset Attribution
              </h3>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.8rem" }}>
                This project is trained on the <strong>INCLUDE (Indian Sign Language Dataset)</strong> created by <strong>AI4Bharat</strong> and the <strong>Indian Institute of Technology (IIT) Madras</strong> in collaboration with deaf students from <strong>St. Louis School for the Deaf, Adyar, Chennai</strong>.
              </p>
              <p style={{ fontSize: "0.84rem", color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
                Citation: Sridhar, A., Ganesan, R. G., Kumar, P., &amp; Khapra, M. (2020). INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. ACM Multimedia 2020. DOI: 10.1145/3394171.3413528 · Zenodo Record 4010759 (Licensed under CC BY 4.0).
              </p>
            </div>

            <div className="card">
              <h3 style={{ fontSize: "1.08rem", marginBottom: "0.6rem" }}>
                User Privacy &amp; Non-Commercial Academic Scope
              </h3>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.8rem" }}>
                <strong>Zero Video Retention:</strong> Webcam frames and uploaded clips are processed exclusively in volatile memory (RAM) to extract numerical MediaPipe landmark coordinates and are immediately deleted after inference.
              </p>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                <strong>Developer Credits:</strong> Engineered by Tarndeep Singh as an end-to-end Machine Learning &amp; MLOps Accessibility Capstone Project.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}