# Heart_failure_predication
# CardioCheck: Heart Failure Risk Predictor

A small Flask web app that estimates the risk of a heart failure death event from 12 clinical measurements. It serves a pre-trained K-Nearest Neighbors model (`heart_failure_predication.pkl`) behind a responsive, single-page interface.

> **Disclaimer:** This project is for learning and demonstration only. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Features

- Clean, responsive UI with a navy, red and teal palette and an animated ECG header
- Three-section form: patient details, lab results and medical history
- Animated risk ring showing the predicted probability, coloured red (higher risk) or teal (lower risk)
- JSON API endpoint (`/predict`) that can be used without the UI
- Health check endpoint (`/health`) for hosting platforms
- Ready to deploy on Render with Gunicorn

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Gunicorn |
| Model | scikit-learn `KNeighborsClassifier` |
| Data handling | pandas, NumPy, joblib |
| Frontend | HTML, CSS and vanilla JavaScript, embedded in `app.py` |

## Project structure

```
.
├── app.py                          # Flask app, API and UI
├── requirements.txt                # Pinned dependencies
├── heart_failure_predication.pkl   # Trained KNN model
└── README.md
```

## Model inputs

The model expects these 12 features, in this order:

| Feature | Description | Values |
|---|---|---|
| `age` | Age of the patient | years |
| `anaemia` | Decrease of red blood cells | 0 = No, 1 = Yes |
| `creatinine_phosphokinase` | CPK enzyme level | mcg/L |
| `diabetes` | Patient has diabetes | 0 = No, 1 = Yes |
| `ejection_fraction` | Percentage of blood pumped out per heartbeat | % |
| `high_blood_pressure` | Patient has hypertension | 0 = No, 1 = Yes |
| `platelets` | Platelet count | kiloplatelets/mL |
| `serum_creatinine` | Creatinine level in blood | mg/dL |
| `serum_sodium` | Sodium level in blood | mEq/L |
| `sex` | Sex of the patient | 0 = Female, 1 = Male |
| `smoking` | Patient smokes | 0 = No, 1 = Yes |
| `time` | Follow-up period | days |

**Output:** class `1` means a death event is predicted, class `0` means survival. The app also shows the model's probability for class `1`.

## Run locally

1. **Clone the repo and enter the folder**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the app**

   ```bash
   python app.py
   ```

5. Open <http://localhost:5000> in your browser.

## API

### `POST /predict`

Send all 12 features as numbers in a JSON body.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 75,
    "anaemia": 0,
    "creatinine_phosphokinase": 582,
    "diabetes": 0,
    "ejection_fraction": 20,
    "high_blood_pressure": 1,
    "platelets": 265000,
    "serum_creatinine": 1.9,
    "serum_sodium": 130,
    "sex": 1,
    "smoking": 0,
    "time": 4
  }'
```

Response:

```json
{ "prediction": 1, "probability": 0.6 }
```

Errors return a JSON message with status `400`, for example `{"error": "Missing field: age."}`.

### `GET /health`

Returns `{"status": "ok"}`.

## Deploy on Render

1. Push `app.py`, `requirements.txt` and `heart_failure_predication.pkl` to the root of a GitHub repository.
2. On [Render](https://render.com), create a **New Web Service** and connect the repository.
3. Use these settings:

   | Setting | Value |
   |---|---|
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn app:app` |
   | Environment variable | `PYTHON_VERSION` = `3.12.7` |

4. Deploy. Optionally set the health check path to `/health`.

Render's free tier sleeps after inactivity, so the first request after a break can take 30 to 60 seconds.

## Important notes

- **Keep the scikit-learn version pinned.** The model was saved with scikit-learn 1.9.1, and loading a pickle under a different version can cause warnings or wrong results. If you retrain the model, update the version in `requirements.txt` to match.
- **No feature scaling.** The model was saved without a scaler, so the app passes raw values, as the model was trained. Features with large values, such as platelets and CPK, dominate the distance calculation in KNN. For better accuracy, retrain with a scaler (for example, a scikit-learn `Pipeline`) and save the whole pipeline.
- **Coarse probabilities.** With the default 5 neighbors, the probability can only be 0%, 20%, 40%, 60%, 80% or 100%.
- **Only load pickle files you trust.** Loading an untrusted pickle can run arbitrary code.

## Possible improvements

- Retrain with scaling and compare other models, such as Random Forest or Logistic Regression
- Show which inputs contributed most to the result
- Add input validation ranges on the server
- Add tests for the `/predict` endpoint


