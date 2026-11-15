"""Record webcam clips. SPACE = start/stop clip, Q = quit.
  python -m isl.record --label idle          (record LONG, varied idle: sit, talk, scratch, drink, reach)
  python -m isl.record --label hello         (custom sign clips ~2-3s each)"""
import argparse, os, time, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--label", required=True)
ap.add_argument("--root", default="data/custom"); ap.add_argument("--cam", type=int, default=0)
a = ap.parse_args()
d = os.path.join(a.root, a.label); os.makedirs(d, exist_ok=True)
cap = cv2.VideoCapture(a.cam); w = None; n = len(os.listdir(d))
while True:
    ok, f = cap.read()
    if not ok: break
    if w is not None: w.write(f)
    v = f.copy(); cv2.putText(v, f"{a.label}  clips:{n}  {'REC' if w else 'SPACE=rec'}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255) if w else (0, 255, 0), 2)
    cv2.imshow("record", v); k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == 32:
        if w is None:
            h, ww = f.shape[:2]
            w = cv2.VideoWriter(os.path.join(d, f"{int(time.time())}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (ww, h))
        else:
            w.release(); w = None; n += 1
if w: w.release()
cap.release(); cv2.destroyAllWindows()
