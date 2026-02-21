# main.py
import os
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template_string, flash
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")  # if empty, UI will allow manual inputs
LAT = os.getenv("LATITUDE", "13.0827")      # change if needed
LON = os.getenv("LONGITUDE", "80.2707")

DATA_CSV = "dataset.csv"
MODEL_FILE = "model.joblib"
LE_CROP_FILE = "le_crop.joblib"
LE_SEASON_FILE = "le_season.joblib"
PRED_LOG = "predictions_log.csv"

app = Flask(__name__)
app.secret_key = "devkey"

def season_from_month(month):
    if month in [6,7,8,9]:
        return "Monsoon"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [12,1,2]:
        return "Winter"
    else:
        return "Autumn"

def fetch_weather(lat, lon):
    if not API_KEY:
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        temp = float(j["main"]["temp"])
        hum = float(j["main"]["humidity"])
        wind = float(j.get("wind", {}).get("speed", 0.0))
        return temp, hum, wind
    except Exception as e:
        print("Weather fetch failed:", e)
        return None

def ensure_dataset_exists():
    if not os.path.exists(DATA_CSV):
        df = pd.DataFrame({
            "crop":["paddy","paddy","wheat","maize","cotton"],
            "season":["Monsoon","Summer","Winter","Monsoon","Summer"],
            "temperature":[30,34,18,28,33],
            "humidity":[78,60,75,80,55],
            "wind_speed":[4.5,3.8,2.1,5.0,4.0],
            "yield":[3200,2800,2600,3000,2200]
        })
        df.to_csv(DATA_CSV, index=False)
        print("Created demo dataset.csv — replace with real data for better results.")

def train_and_save_model():
    ensure_dataset_exists()
    df = pd.read_csv(DATA_CSV)
    if df.empty:
        raise ValueError("dataset.csv is empty")
    # encode categories
    le_crop = LabelEncoder()
    le_season = LabelEncoder()
    df['crop_enc'] = le_crop.fit_transform(df['crop'].astype(str))
    df['season_enc'] = le_season.fit_transform(df['season'].astype(str))
    X = df[['temperature','humidity','wind_speed','crop_enc','season_enc']].values
    y = df['yield'].values
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le_crop, LE_CROP_FILE)
    joblib.dump(le_season, LE_SEASON_FILE)
    print("Model trained and saved.")
    return model, le_crop, le_season

def load_model_or_train():
    if os.path.exists(MODEL_FILE) and os.path.exists(LE_CROP_FILE) and os.path.exists(LE_SEASON_FILE):
        model = joblib.load(MODEL_FILE)
        le_crop = joblib.load(LE_CROP_FILE)
        le_season = joblib.load(LE_SEASON_FILE)
        return model, le_crop, le_season
    else:
        return train_and_save_model()

model, le_crop, le_season = load_model_or_train()

# HTML template (simple)
TEMPLATE = """
<!doctype html>
<title>Crop Yield Predictor</title>
<h2>Crop Yield Predictor (temp, humidity, wind, season)</h2>
<form method="post">
  Crop:
  <select name="crop">
    {% for c in crops %}
      <option value="{{c}}">{{c}}</option>
    {% endfor %}
  </select>
  <br><br>
  Use API for weather? {{ api_ok }}
  <br>
  If API not available or you prefer manual, enter values below:
  <br>
  Temp (°C): <input name="temp" value="{{temp or ''}}">
  Humidity (%): <input name="hum" value="{{hum or ''}}">
  Wind speed (m/s): <input name="wind" value="{{wind or ''}}">
  <br><br>
  <button type="submit">Predict Yield</button>
</form>

{% if result %}
<hr>
<h3>Prediction Result</h3>
<ul>
  <li>Crop: {{result.crop}}</li>
  <li>Season: {{result.season}}</li>
  <li>Temperature: {{result.temperature}} °C</li>
  <li>Humidity: {{result.humidity}} %</li>
  <li>Wind speed: {{result.wind}} m/s</li>
  <li><b>Predicted yield: {{result.pred:.2f}}</b> (same unit as dataset.csv)</li>
</ul>

<form method="post" action="/save">
  <input type="hidden" name="crop" value="{{result.crop}}">
  <input type="hidden" name="season" value="{{result.season}}">
  <input type="hidden" name="temperature" value="{{result.temperature}}">
  <input type="hidden" name="humidity" value="{{result.humidity}}">
  <input type="hidden" name="wind" value="{{result.wind}}">
  <input type="hidden" name="pred" value="{{result.pred}}">
  <button type="submit">Save this sample to dataset.csv (for future retraining)</button>
</form>
{% endif %}

<hr>
<form method="post" action="/retrain">
  <button type="submit">Retrain model from dataset.csv</button>
</form>
<p>Prediction log saved to <b>predictions_log.csv</b></p>
"""

@app.route('/', methods=['GET','POST'])
def index():
    global model, le_crop, le_season
    api_ok = bool(API_KEY)
    crops = list(le_crop.classes_)
    temp = hum = wind = None
    result = None

    if request.method == 'POST':
        # try API first
        w = fetch_weather(LAT, LON) if api_ok else None
        # If user entered manual values, prefer those
        t_in = request.form.get('temp', '').strip()
        h_in = request.form.get('hum', '').strip()
        w_in = request.form.get('wind', '').strip()
        if t_in and h_in and w_in:
            try:
                temp = float(t_in); hum = float(h_in); wind = float(w_in)
            except:
                flash("Invalid manual numeric inputs")
                return redirect(url_for('index'))
        elif w is not None:
            temp, hum, wind = w
        else:
            flash("No API key and no manual weather values provided")
            return redirect(url_for('index'))

        # season detection
        month = datetime.now().month
        season = season_from_month(month)

        crop = request.form.get('crop') or crops[0]
        # find exact crop class (case sensitive)
        if crop not in le_crop.classes_:
            # try case-insensitive match
            found = None
            for c in le_crop.classes_:
                if c.lower() == crop.lower():
                    found = c; break
            if found:
                crop = found
            else:
                flash(f"Crop '{crop}' not in training data. Add to dataset.csv and retrain.")
                return redirect(url_for('index'))

        # encode and predict
        crop_enc = int(le_crop.transform([crop])[0])
        if season not in list(le_season.classes_):
            flash(f"Season '{season}' not in training encoder. Add rows with this season in dataset.csv then retrain.")
            return redirect(url_for('index'))
        season_enc = int(le_season.transform([season])[0])
        X = np.array([[temp, hum, wind, crop_enc, season_enc]])
        pred = float(model.predict(X)[0])

        # log prediction
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "crop": crop, "season": season,
               "temperature": temp, "humidity": hum, "wind_speed": wind,
               "predicted_yield": pred}
        dfp = pd.DataFrame([row])
        header = not os.path.exists(PRED_LOG)
        dfp.to_csv(PRED_LOG, mode='a', header=header, index=False)

        result = {
            "crop": crop,
            "season": season,
            "temperature": temp,
            "humidity": hum,
            "wind": wind,
            "pred": pred
        }

    return render_template_string(TEMPLATE, crops=crops, api_ok=api_ok, temp=temp, hum=hum, wind=wind, result=result)

@app.route('/save', methods=['POST'])
def save_sample():
    # append predicted sample to dataset.csv (so later you can update yield manually and retrain)
    crop = request.form['crop']
    season = request.form['season']
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind = float(request.form['wind'])
    pred = float(request.form['pred'])
    # append
    df = pd.DataFrame([{
        "crop": crop, "season": season,
        "temperature": temperature, "humidity": humidity, "wind_speed": wind,
        "yield": pred
    }])
    header = not os.path.exists(DATA_CSV)
    df.to_csv(DATA_CSV, mode='a', header=header, index=False)
    flash("Sample appended to dataset.csv. Retrain to update model.")
    return redirect(url_for('index'))

@app.route('/retrain', methods=['POST'])
def retrain():
    global model, le_crop, le_season
    try:
        model, le_crop, le_season = train_and_save_model()
        flash("Model retrained from dataset.csv")
    except Exception as e:
        flash("Retrain failed: " + str(e))
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
