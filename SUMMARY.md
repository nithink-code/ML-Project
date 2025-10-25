## Project summary: how to run backend, frontend, and ML model flow

This single-file summary describes how to start the backend server and the frontend app, and gives a concise explanation of the ML model flow used in this project.

---

## Quick commands (PowerShell)

Backend (Windows PowerShell):
```powershell
cd "c:\Users\nithi\OneDrive\Documents\ML Project\backend"

& "C:\Program Files\Python313\python.exe" -m uvicorn server:app --reload

Frontend (Windows PowerShell):

```powershell
cd "c:\Users\nithi\OneDrive\Documents\ML Project\frontend"
npm install
# run the dev server (check `package.json` scripts — typically `start`):
npm start
```

Notes:
- If `backend/server.py` uses a specific port or environment variables, inspect it for `PORT`/`HOST` and any `os.environ` usage. The server may also be started by `start_server.bat` which wraps the same command.
- The frontend uses a React-based app with `craco` present; if `npm start` fails, run `npm run start` or inspect `package.json` scripts.

---

## Files to inspect (keys used in summary)

- `backend/server.py` — backend API entrypoint. Look here for host/port, dependencies (Flask/FastAPI/other) and endpoints that serve the ML model.
- `backend/requirements.txt` — Python dependencies for the backend and the ML code.
- `backend/start_server.bat` — Windows helper to run the backend.
- `backend/generate_secret.py` — helper script for generating secrets/keys used by the backend.
- `backend/test_connection.py`, `backend/test_openai.py` — tests/examples showing how the backend connects to external services or verifies environment.
- `backend/nlp/trainer.py` — where the ML model training logic lives (data loading, preprocessing, training loop, saving models).
- `frontend/package.json` — scripts and dependencies for the frontend.
- `frontend/env.txt` — environment variables or example variables for the frontend.
- `frontend/src/lib/axios.js` — client API helper used by frontend pages to call the backend.
- `frontend/src/pages/*` — pages that call the backend API (e.g., `InterviewPage.jsx`, `EvaluationPage.jsx`, `Dashboard.jsx`).

---

## ML model flow (conceptual, matching this repo layout)

This project separates ML work into the `backend/nlp` area and exposes inference through the backend server.

1. Data ingestion and preprocessing
   - Training data is loaded by `backend/nlp/trainer.py` (or helper modules it imports).
   - Preprocessing steps (tokenization, normalization, feature extraction) happen before model training. Check the trainer for functions named `load_data`, `preprocess`, or similar.

2. Model training
   - `trainer.py` contains the model architecture and training loop. It will:
     - Configure model hyperparameters
     - Create datasets / dataloaders
     - Train epochs and compute metrics
     - Save a trained model to disk (look for `model.save`, `torch.save`, `joblib.dump`, or similar calls)

3. Model persistence
   - Trained model files are stored in a directory (search for `models/`, `artifacts/`, or `saved_models/`). The `trainer.py` or training script will log or return the saved path.

4. Loading and serving the model
   - `backend/server.py`'s startup path typically loads the saved model into memory (look for `load_model`, `torch.load`, `joblib.load`, or custom loader functions).
   - The server exposes endpoints that accept requests from the frontend (JSON payloads). On request:
     - Input is validated and preprocessed (same steps as during training)
     - The model runs inference and returns predictions in JSON

5. Frontend interaction
   - Frontend pages use `frontend/src/lib/axios.js` to call backend endpoints (e.g., `/predict`, `/evaluate`, `/start-interview`).
   - Pages such as `InterviewPage.jsx` and `EvaluationPage.jsx` send user inputs and display model results.

6. Optional external services
   - There are tests like `test_openai.py` indicating optional integration with external APIs (OpenAI or other). If used, check `backend/test_openai.py` and environment variables required (API keys).

Edge cases and operational notes:
- Ensure training/inference use the same preprocessing steps and tokenizer/feature pipeline. A mismatch is the most common source of bad results.
- Watch memory usage when loading models into the backend; for large models consider lazy loading or using a model server.
- If you use GPU acceleration, the `requirements.txt` or `trainer.py` will reference CUDA-capable libraries (PyTorch/CUDA); confirm host GPU drivers.

---

## How the pieces connect (flow diagram in words)

User (browser) -> Frontend pages (`InterviewPage`, `EvaluationPage`) -> HTTP request via `axios` -> Backend API (`backend/server.py`) -> Preprocessing -> ML model (loaded from `backend/nlp/*.py`) -> Prediction -> Backend returns JSON -> Frontend renders results

---

## Running tests / sanity checks

- Backend quick connectivity test:

```powershell
cd "c:\Users\nithi\OneDrive\Documents\ML Project\backend"
python test_connection.py
```

- If there is an OpenAI integration or similar, run:

```powershell
python test_openai.py
```

---

## Troubleshooting tips

- If the backend fails to start: open `server.py` to see required env vars. Use `generate_secret.py` if the project requires a secret key.
- If the frontend can't talk to the backend: check CORS configuration in `server.py` and ensure ports match. Inspect the network tab in browser devtools.
- If inference results differ from training: verify the serializer/tokenizer files and preprocessing code are used consistently in `trainer.py` and `server.py`.

---

## Next steps / useful checks

- Inspect `backend/server.py` for the exact server framework (Flask/FastAPI) and the route names used by the frontend.
- Inspect `trainer.py` to find saved model paths and any configuration files (YAML/JSON) for reproducible training.
- Add small README sections inside `backend/` and `frontend/` if you want more granular run instructions.

If you want, I can also:
- Extract actual port numbers and commands by reading `server.py` and `package.json` and update this file with exact run commands.
- Add a tiny script to automate creating a virtualenv and starting both backend and frontend for local development.

---

File created by automation on behalf of the project owner to provide a single reference for running services and understanding the ML flow.
