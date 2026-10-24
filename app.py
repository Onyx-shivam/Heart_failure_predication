import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_failure_predication.pkl")
model = joblib.load(MODEL_PATH)

# Order must match the order the model was trained on
FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CardioCheck · Heart failure risk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #0E1B3D;
    --ink-soft: #3B4A6B;
    --paper: #F2F5FA;
    --card: #FFFFFF;
    --line: #DCE3EF;
    --pulse: #E5384F;
    --pulse-soft: #FDE6EA;
    --calm: #12A38A;
    --calm-soft: #DDF5F0;
    --gold: #F5A524;
    --radius: 14px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    font-family: "Figtree", system-ui, sans-serif;
    background: var(--paper);
    color: var(--ink);
    line-height: 1.55;
    min-height: 100vh;
  }

  .layout { display: grid; grid-template-columns: minmax(300px, 420px) 1fr; min-height: 100vh; }

  /* ---------- Left hero ---------- */
  .hero {
    background: radial-gradient(circle at 20% 0%, #1B2F66 0%, var(--ink) 55%);
    color: #fff;
    padding: 48px 40px;
    position: sticky;
    top: 0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 1.15rem; }
  .brand-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--pulse); box-shadow: 0 0 0 0 rgba(229,56,79,.7); animation: beat 1.6s infinite; }
  @keyframes beat {
    0% { box-shadow: 0 0 0 0 rgba(229,56,79,.7); }
    70% { box-shadow: 0 0 0 14px rgba(229,56,79,0); }
    100% { box-shadow: 0 0 0 0 rgba(229,56,79,0); }
  }
  .hero h1 {
    font-family: "Bricolage Grotesque", sans-serif;
    font-weight: 800;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    line-height: 1.04;
    letter-spacing: -0.02em;
    margin-bottom: 18px;
  }
  .hero p { color: #B9C4E0; max-width: 32ch; font-size: 1.02rem; }
  .ecg { width: 140%; margin-left: -20px; height: 90px; }
  .ecg path {
    fill: none; stroke: var(--pulse); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round;
    stroke-dasharray: 1200; stroke-dashoffset: 1200;
    animation: draw 3.2s ease-out forwards, glow 2.4s 3.2s ease-in-out infinite;
  }
  @keyframes draw { to { stroke-dashoffset: 0; } }
  @keyframes glow { 0%,100% { filter: drop-shadow(0 0 2px rgba(229,56,79,.4)); } 50% { filter: drop-shadow(0 0 9px rgba(229,56,79,.95)); } }
  .fineprint { font-size: .8rem; color: #8C9ABD; max-width: 38ch; }

  /* ---------- Right form ---------- */
  main { padding: 48px clamp(20px, 5vw, 64px) 64px; max-width: 880px; width: 100%; }
  .group {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 26px 26px 8px;
    margin-bottom: 20px;
  }
  .group h2 { font-family: "Bricolage Grotesque", sans-serif; font-size: 1.15rem; margin-bottom: 4px; }
  .group .sub { color: var(--ink-soft); font-size: .92rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px 22px; margin-bottom: 18px; }
  .field label { display: block; font-weight: 600; font-size: .92rem; margin-bottom: 6px; }
  .field .hint { display: block; font-weight: 400; color: var(--ink-soft); font-size: .8rem; }
  input[type=number] {
    width: 100%; padding: 12px 14px; font: inherit; color: var(--ink);
    background: var(--paper); border: 1.5px solid transparent; border-radius: 10px;
    transition: border-color .15s, background .15s;
  }
  input[type=number]:focus { outline: none; border-color: var(--ink); background: #fff; }
  input:focus-visible, .seg input:focus-visible + span, .switch input:focus-visible + .track { outline: 3px solid var(--gold); outline-offset: 2px; }

  /* Yes / No segmented */
  .seg { display: inline-flex; background: var(--paper); border-radius: 10px; padding: 4px; width: 100%; }
  .seg label { flex: 1; margin: 0; cursor: pointer; }
  .seg input { position: absolute; opacity: 0; pointer-events: none; }
  .seg span {
    display: block; text-align: center; padding: 9px 0; border-radius: 7px;
    font-weight: 600; font-size: .9rem; color: var(--ink-soft); transition: background .15s, color .15s;
  }
  .seg input:checked + span { background: var(--ink); color: #fff; }

  .actions { display: flex; align-items: center; gap: 16px; margin-top: 8px; flex-wrap: wrap; }
  button {
    font: inherit; font-weight: 700; cursor: pointer; border: none;
    background: var(--pulse); color: #fff; padding: 15px 34px; border-radius: 12px;
    transition: transform .12s, box-shadow .12s, background .12s;
    box-shadow: 0 8px 20px -8px rgba(229,56,79,.7);
  }
  button:hover { transform: translateY(-2px); background: #D12D44; }
  button:disabled { opacity: .6; cursor: wait; transform: none; }
  button.ghost { background: transparent; color: var(--ink-soft); box-shadow: none; padding: 15px 8px; font-weight: 600; }
  button.ghost:hover { color: var(--ink); background: transparent; transform: none; }

  /* ---------- Result ---------- */
  #result { margin-top: 28px; display: none; }
  #result.show { display: block; animation: rise .5s ease-out; }
  @keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
  .result-card {
    display: grid; grid-template-columns: 150px 1fr; gap: 26px; align-items: center;
    border-radius: var(--radius); padding: 26px; border: 1.5px solid;
  }
  .result-card.high { background: var(--pulse-soft); border-color: var(--pulse); }
  .result-card.low { background: var(--calm-soft); border-color: var(--calm); }
  .ring { position: relative; width: 150px; height: 150px; }
  .ring svg { transform: rotate(-90deg); }
  .ring circle { fill: none; stroke-width: 12; stroke-linecap: round; }
  .ring .bg { stroke: rgba(14,27,61,.12); }
  .ring .fg { transition: stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1); }
  .high .fg { stroke: var(--pulse); }
  .low .fg { stroke: var(--calm); }
  .ring .pct {
    position: absolute; inset: 0; display: grid; place-items: center;
    font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 2rem;
  }
  .result-card h3 { font-family: "Bricolage Grotesque", sans-serif; font-size: 1.5rem; margin-bottom: 6px; }
  .result-card p { color: var(--ink-soft); }
  .disclaimer { margin-top: 14px; font-size: .82rem; color: var(--ink-soft); }
  .error { background: #FFF3D6; border: 1.5px solid var(--gold); padding: 16px 20px; border-radius: 12px; }

  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .hero { position: relative; height: auto; gap: 28px; padding: 32px 24px; }
    .ecg { width: 120%; }
    main { padding-top: 28px; }
  }
  @media (max-width: 560px) {
    .grid { grid-template-columns: 1fr; }
    .result-card { grid-template-columns: 1fr; justify-items: center; text-align: center; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
    .ecg path { stroke-dashoffset: 0; }
  }
</style>
</head>
<body>
<div class="layout">

  <aside class="hero">
    <div class="brand"><span class="brand-dot"></span>CardioCheck</div>
    <div>
      <h1>Estimate heart failure risk in seconds.</h1>
      <p>Enter a patient’s clinical measurements and get a risk estimate from a trained machine learning model.</p>
    </div>
    <div>
      <svg class="ecg" viewBox="0 0 600 90" preserveAspectRatio="none" aria-hidden="true">
        <path d="M0 50 H110 L130 50 L145 20 L165 78 L185 6 L205 60 L220 50 H330 L350 50 L365 24 L385 76 L405 10 L425 58 L440 50 H600"/>
      </svg>
      <p class="fineprint">For education and demonstration. This is not a medical diagnosis.</p>
    </div>
  </aside>

  <main>
    <form id="form" novalidate>

      <section class="group">
        <h2>About the patient</h2>
        <p class="sub">Basic details and how long they have been followed.</p>
        <div class="grid">
          <div class="field">
            <label for="age">Age <span class="hint">years</span></label>
            <input type="number" id="age" name="age" min="1" max="120" step="1" placeholder="e.g. 60" required>
          </div>
          <div class="field">
            <label>Sex</label>
            <div class="seg">
              <label><input type="radio" name="sex" value="0"><span>Female</span></label>
              <label><input type="radio" name="sex" value="1" checked><span>Male</span></label>
            </div>
          </div>
          <div class="field">
            <label for="time">Follow-up period <span class="hint">days</span></label>
            <input type="number" id="time" name="time" min="0" max="500" step="1" placeholder="e.g. 130" required>
          </div>
          <div class="field">
            <label>Smoker</label>
            <div class="seg">
              <label><input type="radio" name="smoking" value="0" checked><span>No</span></label>
              <label><input type="radio" name="smoking" value="1"><span>Yes</span></label>
            </div>
          </div>
        </div>
      </section>

      <section class="group">
        <h2>Lab results</h2>
        <p class="sub">Values from the patient’s blood work and heart scan.</p>
        <div class="grid">
          <div class="field">
            <label for="ejection_fraction">Ejection fraction <span class="hint">% of blood pumped per beat</span></label>
            <input type="number" id="ejection_fraction" name="ejection_fraction" min="1" max="100" step="1" placeholder="e.g. 38" required>
          </div>
          <div class="field">
            <label for="serum_creatinine">Serum creatinine <span class="hint">mg/dL</span></label>
            <input type="number" id="serum_creatinine" name="serum_creatinine" min="0" max="15" step="0.01" placeholder="e.g. 1.1" required>
          </div>
          <div class="field">
            <label for="serum_sodium">Serum sodium <span class="hint">mEq/L</span></label>
            <input type="number" id="serum_sodium" name="serum_sodium" min="100" max="160" step="1" placeholder="e.g. 137" required>
          </div>
          <div class="field">
            <label for="creatinine_phosphokinase">Creatinine phosphokinase <span class="hint">mcg/L</span></label>
            <input type="number" id="creatinine_phosphokinase" name="creatinine_phosphokinase" min="1" max="10000" step="1" placeholder="e.g. 250" required>
          </div>
          <div class="field">
            <label for="platelets">Platelets <span class="hint">kiloplatelets/mL</span></label>
            <input type="number" id="platelets" name="platelets" min="1000" max="900000" step="1" placeholder="e.g. 263358" required>
          </div>
        </div>
      </section>

      <section class="group">
        <h2>Medical history</h2>
        <p class="sub">Select Yes for any condition the patient has.</p>
        <div class="grid">
          <div class="field">
            <label>Anaemia</label>
            <div class="seg">
              <label><input type="radio" name="anaemia" value="0" checked><span>No</span></label>
              <label><input type="radio" name="anaemia" value="1"><span>Yes</span></label>
            </div>
          </div>
          <div class="field">
            <label>Diabetes</label>
            <div class="seg">
              <label><input type="radio" name="diabetes" value="0" checked><span>No</span></label>
              <label><input type="radio" name="diabetes" value="1"><span>Yes</span></label>
            </div>
          </div>
          <div class="field">
            <label>High blood pressure</label>
            <div class="seg">
              <label><input type="radio" name="high_blood_pressure" value="0" checked><span>No</span></label>
              <label><input type="radio" name="high_blood_pressure" value="1"><span>Yes</span></label>
            </div>
          </div>
        </div>
      </section>

      <div class="actions">
        <button type="submit" id="submit">Check risk</button>
        <button type="button" class="ghost" id="reset">Clear form</button>
      </div>
    </form>

    <div id="result" aria-live="polite"></div>
  </main>
</div>

<script>
  const form = document.getElementById("form");
  const result = document.getElementById("result");
  const submit = document.getElementById("submit");
  const CIRC = 2 * Math.PI * 64;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(form).entries());

    const missing = Array.from(form.querySelectorAll("input[type=number]")).filter(i => i.value === "");
    if (missing.length) {
      missing[0].focus();
      show(`<div class="error">Please fill in <strong>${missing.map(m => m.closest(".field").querySelector("label").firstChild.textContent.trim()).join(", ")}</strong> before checking the risk.</div>`);
      return;
    }

    submit.disabled = true; submit.textContent = "Checking…";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || "Something went wrong.");
      render(out);
    } catch (err) {
      show(`<div class="error">${err.message} Check the values and try again.</div>`);
    } finally {
      submit.disabled = false; submit.textContent = "Check risk";
    }
  });

  document.getElementById("reset").addEventListener("click", () => {
    form.reset(); result.className = ""; result.innerHTML = "";
  });

  function render(out) {
    const high = out.prediction === 1;
    const pct = Math.round(out.probability * 100);
    show(`
      <div class="result-card ${high ? "high" : "low"}">
        <div class="ring">
          <svg width="150" height="150" viewBox="0 0 150 150">
            <circle class="bg" cx="75" cy="75" r="64"></circle>
            <circle class="fg" id="fg" cx="75" cy="75" r="64" stroke-dasharray="${CIRC}" stroke-dashoffset="${CIRC}"></circle>
          </svg>
          <div class="pct">${pct}%</div>
        </div>
        <div>
          <h3>${high ? "Higher risk of a death event" : "Lower risk of a death event"}</h3>
          <p>${high
            ? "The model estimates this profile is similar to patients who did not survive the follow-up period. A clinician should review these results."
            : "The model estimates this profile is similar to patients who survived the follow-up period. Regular check-ups are still important."}</p>
        </div>
      </div>
      <p class="disclaimer">This estimate comes from a machine learning model for learning purposes. It is not a medical diagnosis.</p>
    `);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.getElementById("fg").style.strokeDashoffset = CIRC * (1 - out.probability);
    }));
  }

  function show(html) {
    result.innerHTML = html;
    result.className = "show";
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        row = {name: float(payload[name]) for name in FEATURES}
    except KeyError as missing:
        return jsonify(error=f"Missing field: {missing.args[0]}."), 400
    except (TypeError, ValueError):
        return jsonify(error="All fields must be numbers."), 400

    X = pd.DataFrame([row], columns=FEATURES)
    prediction = int(model.predict(X)[0])

    probability = float(prediction)
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probability = float(model.predict_proba(X)[0][classes.index(1)])

    return jsonify(prediction=prediction, probability=probability)


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
