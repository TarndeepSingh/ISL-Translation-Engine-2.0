# Known issues
1. **Signer leakage.** INCLUDE has about 7 signers; official splits are by video, so test scores overestimate performance on new people. (Mitigation: session-grouped split; fresh-signer test.)
2. **Session proxy is an assumption.** Sessions are inferred from camera file numbers (gap = 30). Check clusters by eye; there is no signer id in the data we use.
3. **Idle class domain shortcut.** Idle clips recorded by the user differ from INCLUDE recordings; a model can learn "environment", not motion.
4. **Live camera quality.** Dark, close-up frames lose hands; the API reports `poor_input` when shoulders or hands are rarely visible.
5. **Unpinned Colab packages** (mediapipe) vs browser 0.10.14: same model files, but a parity test (`isl.diagnose`) is still required.
6. **943 vs 958 INCLUDE-50 videos.** The manifest builder drops metadata rows whose label disagrees with the folder or whose file is missing; not yet traced.
7. **Zero extracted files** from two zips (Jobs_2of2, Animals_2of2) were seen; probably legitimate, unverified.
8. **No trained model ships in this repo.** `/api/predict` answers 503 until `models/isl.onnx` + `isl.labels.json` exist.
9. **Server-side video decoding** needs `mediapipe` + OpenCV in the backend environment; the first run downloads two `.task` model files.
10. **Sentence translation does not exist.** `/api/translate` returns 501 by design.
11. **Overlap between iSign and ISLTranslate is unchecked** (no data inspected yet); never count overlapping data as independent.
12. **Licences unchecked** for iSign, ISLTranslate, CISLR, ISH-NEWS; INCLUDE terms must be read on its dataset page before any non-research use.
