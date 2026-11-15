import { PoseLandmarker, HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const words = [];
let ws, pose, hands, lastT = -1;

const flat = (l) => l.flatMap((p) => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Assign detected hands to the PERSON's left/right by proximity to pose wrists (15=L, 16=R).
// Matches how training data was built and avoids MediaPipe's mirrored-handedness label.
function assignHands(poseLm, hs) {
  const out = { left: null, right: null };
  if (!hs.length) return out;
  if (!poseLm) { hs.slice(0, 2).forEach((h) => (out[h[0].x > 0.5 ? "left" : "right"] ||= h)); return out; }
  const L = poseLm[15], R = poseLm[16];
  if (hs.length === 1) { const h = hs[0]; out[dist(h[0], L) <= dist(h[0], R) ? "left" : "right"] = h; return out; }
  const [a, b] = hs;
  const straight = dist(a[0], L) + dist(b[0], R), swapped = dist(b[0], L) + dist(a[0], R);
  return straight <= swapped ? { left: a, right: b } : { left: b, right: a };
}

function draw(poseLm, hs) {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const dot = (p, c, r) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, r, 0, 7); ctx.fill(); };
  if (poseLm) [11, 12, 13, 14, 15, 16].forEach((i) => dot(poseLm[i], "#3ddc97", 5));
  hs.forEach((h) => h.forEach((p) => dot(p, "#ffb347", 3)));
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ conf: +$("conf").value }));
  ws.onclose = () => { $("pill").textContent = "disconnected — retrying"; setTimeout(connect, 1500); };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "hello" && !m.model_loaded) $("caption").textContent = "Backend running without a model (see README step 7)";
    if (m.type !== "frame") return;
    const h = m.hands || [false, false];
    $("pill").textContent = (m.active ? "SIGNING" : "idle") + ` · hands ${h[0] ? "L" : "-"}${h[1] ? "R" : "-"}`;
    $("pill").classList.toggle("on", m.active);
    if (m.sign) {
      const t3 = (m.sign.top3 || [[m.sign.label, m.sign.conf]]).map((t) => `${t[0]} ${(+t[1]).toFixed(2)}`).join("  |  ");
      $("last").textContent = `last (${m.sign.frames ?? "?"} frames): ${t3}${m.sign.accepted ? "  ✓ accepted" : "  ✗ rejected"}`;
      if (m.sign.accepted) { words.push(m.sign.label); if (words.length > 8) words.shift(); $("caption").textContent = words.join(" "); }
    }
  };
}

async function loop() {
  if (video.readyState >= 2 && video.currentTime !== lastT && ws?.readyState === 1) {
    lastT = video.currentTime;
    const t = performance.now();
    const p = pose.detectForVideo(video, t).landmarks[0] || null;
    const hs = hands.detectForVideo(video, t).landmarks;
    const { left, right } = assignHands(p, hs);
    ws.send(JSON.stringify({ pose: p && flat(p), left: left && flat(left), right: right && flat(right) }));
    draw(p, hs);
  }
  requestAnimationFrame(loop);
}

(async () => {
  $("conf").oninput = (e) => { $("cv").textContent = (+e.target.value).toFixed(2); ws?.readyState === 1 && ws.send(JSON.stringify({ conf: +e.target.value })); };
  $("clear").onclick = () => { words.length = 0; $("caption").textContent = ""; };
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  await video.play();
  const fs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const base = "https://storage.googleapis.com/mediapipe-models";
  pose = await PoseLandmarker.createFromOptions(fs, { runningMode: "VIDEO", numPoses: 1,
    baseOptions: { modelAssetPath: `${base}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`, delegate: "GPU" } });
  hands = await HandLandmarker.createFromOptions(fs, { runningMode: "VIDEO", numHands: 2,
    baseOptions: { modelAssetPath: `${base}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" } });
  connect(); loop();
})().catch((e) => { $("pill").textContent = "error"; $("caption").textContent = String(e); });
