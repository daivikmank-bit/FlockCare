# FlockCare — Parts 8–12: Deployment, Validation, Roadmap, Risks & Future Work

*Standalone build guide. Assumes Parts 3–6 are done: a trained model, a FastAPI backend (with a Dockerfile from Part 5), and a React frontend (Part 6). This file covers everything after that — getting it live, proving it works, and what comes next.*

---

## Part 8 — Deployment

### 8.0 A capacity reality check worth doing before picking a host

TensorFlow's RAM footprint is the real constraint here, not CPU. A bare `import tensorflow` commonly costs several hundred MB before you've loaded anything — that alone can exceed some hosting free tiers. Checked current specs for the usual suspects (hosting pricing/limits shift often, so re-verify before committing to one):

| Platform | Free tier | Real free tier? | Notes |
|---|---|---|---|
| Render | 512 MB RAM, 0.1 CPU | Yes, but tight | One review put it plainly: fine for light workloads, not AI inference workloads. Free services also spin down after 15 minutes idle, with a 30–60 second cold start on the next request. |
| Railway | 0.5 GB RAM | Not really anymore | Requires a card for the initial trial credit; after 30 days it's $1/month for just 0.5GB RAM — budget for the $5/month Hobby plan if you go this route at all. |
| Hugging Face Spaces (Docker SDK) | 16 GB RAM, 2 vCPU, 50GB disk | Yes, no card required | By far the best fit for a TensorFlow backend. Must listen on a fixed port (7860) — not configurable. Also sleeps on inactivity, same cold-start caveat as the others. |

Two viable paths, pick one:

**Path A — simplest: deploy as-is to Hugging Face Spaces.** Keep full `tensorflow`, use the Docker SDK, and the RAM ceiling stops being a problem entirely.

**Path B — leaner: swap to TFLite and the RAM ceiling stops mattering anywhere.** Use the `.tflite` export already stubbed in Part 4, swap `tensorflow` for the much smaller `tflite-runtime` package, and the memory footprint drops enough that even Render's free tier becomes plausible. Bonus: cold starts get faster too, which matters more than it sounds like for a live demo.

### 8.1 Path A: Hugging Face Spaces (Docker SDK)
1. Create a new Space, choose **Docker** as the SDK.
2. Spaces read config from YAML frontmatter at the top of the Space's `README.md`:
   ```yaml
   ---
   title: FlockCare API
   emoji: 🐔
   colorFrom: red
   colorTo: yellow
   sdk: docker
   app_port: 7860
   ---
   ```
3. Adjust Part 5's Dockerfile — Spaces expects port **7860**, not 8000, and won't let you remap it:
   ```dockerfile
   EXPOSE 7860
   CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
4. Push to the Space (Spaces are Git repos — `git push` directly, or connect a GitHub repo for auto-deploy on push).
5. Set any secrets/env overrides (e.g. a non-default `FLOCKCARE_MODEL_PATH`) under the Space's **Settings → Variables and secrets**.
6. Update the frontend's `VITE_API_BASE_URL` to the Space's public URL: `https://<username>-<space-name>.hf.space`.

### 8.2 Path B: TFLite + a lighter host
Convert once (from Part 4's export step):
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("ml/saved_models/flockcare_cnn.tflite", "wb") as f:
    f.write(tflite_model)
```
Swap the serving code in Part 5's `inference.py`:
```python
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="ml/saved_models/flockcare_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_batch(specs: np.ndarray) -> np.ndarray:
    results = []
    for spec in specs:
        interpreter.set_tensor(input_details[0]['index'], spec[np.newaxis].astype(np.float32))
        interpreter.invoke()
        results.append(interpreter.get_tensor(output_details[0]['index'])[0])
    return np.array(results)
```
And in `backend/requirements.txt`, replace `tensorflow` with `tflite-runtime`. If your Python version doesn't have a prebuilt `tflite-runtime` wheel, `tensorflow-cpu` is a middle ground — still lighter than the default GPU-enabled `tensorflow` package.

### 8.3 Frontend
Vercel or Netlify: connect the `frontend/` folder, build command `npm run build`, output directory `dist`, and set `VITE_API_BASE_URL` in the platform's environment variable settings to whichever backend URL you land on from 8.1 or 8.2. Both give HTTPS by default — required per Part 6.0's `getUserMedia` constraint.

### 8.4 Before you call it deployed
- Tighten Part 5's CORS from `allow_origins=["*"]` to your actual frontend URL.
- **Warm the backend up** (hit `/health` a few minutes) before any live demo — every option above sleeps on inactivity, and a judge's first request landing on a 30–60s cold start is a bad first impression that has nothing to do with whether the product actually works.
- Re-run Part 5's Docker verification checklist against the *actual deployed container*, not just your local build — this is where a missed dependency (like `ffmpeg`, see Part 5.0) would first surface for real.

---

## Part 9 — Testing & Validation

### 9.1 What you already have
Parts 4–6's verification checklists already cover model sanity checks, API contract tests, and end-to-end smoke tests. Re-run those explicitly once more against the **deployed** stack, not just your local dev setup, before calling this done — a working local demo and a working deployed demo are not the same claim.

### 9.2 Worth adding now
- **Light load test:** fire 10–20 concurrent `/analyze` requests at the deployed backend and confirm it doesn't fall over. TensorFlow inference isn't free CPU-wise, and a free-tier single-core instance could queue badly under demo-day traffic from multiple judges hitting it at once.
- **Cross-device audio test:** run the *same physical recording* through both an Android and an iOS device (per Part 6's checklist) and confirm they land on the same `risk_level`. This is the real test of Part 5's format-conversion path — a passing `curl` request with a clean `.wav` doesn't exercise it at all.

### 9.3 Field validation (post-MVP)
Same plan as the original deck names as the next phase:
1. Partner with a local poultry cooperative or veterinary college.
2. Collect labeled recordings from real coops, ideally vet-confirmed per flock.
3. Re-tune the 70/40 risk thresholds (Part 4.10 / 5.7) against real coop noise — fan hum, wind, other animals will very likely shift the optimal cutoff from what clean public datasets suggested.
4. Track false-negative rate specifically in the field pilot — that's the number that determines whether this is actually safe to hand a farmer as a decision aid.

---

## Part 10 — Roadmap

### 10.1 Where you actually are right now
Parts 3–6 are built: data pipeline, trained-and-benchmarked model, API, frontend. Part 8 above gets it deployed. That's a complete, demoable v1 — not a plan for one.

### 10.2 Post-MVP phases
- **Phase 2:** augmentation (Part 3.7, if skipped for time during the initial build), multilingual polish (Part 6.11), threshold re-tuning once you have real usage data.
- **Phase 3:** field validation (9.3) with a cooperative or vet partner.
- **Phase 4:** on-device inference, expanded disease coverage beyond the binary healthy/elevated-respiratory signal, SMS/IVR fallback for feature phones (Part 12).

---

## Part 11 — Risks & Mitigations

| Risk | Mitigation |
|---|---|
| False negatives (missed sick birds) — the costliest failure mode for a health tool | Recall-first threshold tuning (4.2, 4.10); "screening, not diagnosis" disclaimer always shown |
| Real coop noise not represented in public training data | Augmentation (3.7); field validation (9.3) |
| Model bias from limited dataset diversity (species, region, mic hardware) | Held-out test set treated as a floor, not a ceiling; plan regional data collection before wider rollout |
| Free-tier hosting sleeping mid-demo | Warm the backend before presenting (8.4), or pay for an always-on tier specifically on demo day |
| TensorFlow's RAM footprint exceeding free-tier limits | Path B's TFLite swap (8.2), or Path A's higher-RAM host (8.1) |
| Browser/device audio format inconsistency (webm vs mp4) | Format detection + conversion (5.5, 6.3); cross-device testing (9.2) |
| Over-claiming diagnostic accuracy | Consistent "screening, not diagnosis" language throughout (4.10, 5.7, 6.9) |

---

## Part 12 — Future Enhancements (beyond v1)
- On-device inference (TFLite, already built toward in 8.2) for fully offline screening — the real fix for the low-connectivity-farms limitation noted back in the original plan.
- SMS/IVR-based results for farmers without smartphones.
- "Find nearest vet" directory integration, replacing the static Maps link from Part 6.9.
- Multi-disease differentiation beyond a binary healthy/elevated-respiratory signal.
- Community-level dashboard aggregating anonymized regional results into an early-warning heatmap for local disease spread.

---

## That's the full plan
Parts 3–6 build it, Part 8 deploys it, Part 9 proves it works, Parts 10–12 say what's next. Every part in this series has been standalone on purpose — if you hand any one of these files to a coding assistant on its own, it has everything it needs for that piece without guessing at the rest.
