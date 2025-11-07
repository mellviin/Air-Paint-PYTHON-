# AirDraw — Render Deploy (Streamlit + WebRTC)

This repository contains a Render-ready Streamlit + WebRTC version of **AirDraw** (gesture-driven drawing using MediaPipe + OpenCV).

## Files
- `server.py` — Streamlit app using `streamlit-webrtc` (main file)
- `requirements.txt` — pinned package versions suitable for Render
- `runtime.txt` — Python runtime (3.10)

## Run locally (recommended for testing)
1. Create a virtual environment (Python 3.10):
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run server.py --server.port=8501 --server.address=0.0.0.0
   ```
3. Open your browser to `http://localhost:8501` and allow camera access.

## Deploy to Render (step-by-step)
1. Push this repo to GitHub.
2. In Render dashboard, create a new **Web Service**.
3. Connect your GitHub repo and select the branch.
4. Set **Start Command** to:
   ```bash
   streamlit run server.py --server.port=$PORT --server.address=0.0.0.0
   ```
5. Set the environment to use Python 3.10 (Render will use `runtime.txt`).
6. Deploy and open the provided URL. Allow camera access in the browser.

## Notes & troubleshooting
- If the deployment fails building `av` or similar, check Render build logs. Render generally handles these dependencies better than Streamlit Cloud.
- If camera preview doesn't show, ensure the page is served over HTTPS and your browser allows camera access.
- This app uses medium video quality by default to balance performance and latency.
