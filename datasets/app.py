from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, pandas as pd, numpy as np

app = Flask(__name__)
CORS(app)

with open("co2_model.pkl","rb") as f:
    d = pickle.load(f)
rf_model        = d["model"]
scaler          = d["scaler"]
feature_columns = d["feature_columns"]
print(f"✅ Model loaded. Features: {len(feature_columns)}")

def build_row(data):
    row = {col: 0 for col in feature_columns}
    row["engine_size"]            = float(data.get("engine_size", 2.0))
    row["cylinders"]              = float(data.get("cylinders", 4))
    row["fuel_consumption_city"]  = float(data.get("fuel_consumption_city", 10.0))
    row["fuel_consumption_hwy"]   = float(data.get("fuel_consumption_hwy", 7.5))
    row["fuel_consumption_comb"]  = float(data.get("fuel_consumption_comb", 9.0))
    row["fuel_consumption_mpg"]   = float(data.get("fuel_consumption_mpg", 28))

    for prefix, key in [("vehicle_class_","vehicle_class"),
                         ("fuel_type_","fuel_type"),
                         ("transmission_","transmission")]:
        k = f"{prefix}{data.get(key,'')}"
        if k in row:
            row[k] = True

    return pd.DataFrame([row])[feature_columns]

@app.route("/")
def home():
    return jsonify({"status": "CO2 API running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        inp  = build_row(data)
        co2  = round(float(rf_model.predict(scaler.transform(inp))[0]), 1)
        annual = round(co2 * 15000 / 1000, 0)
        if   co2 < 120: rating,cls = "Low emissions","A+"
        elif co2 < 150: rating,cls = "Below average","A"
        elif co2 < 180: rating,cls = "Below average","B"
        elif co2 < 220: rating,cls = "Average","C"
        elif co2 < 260: rating,cls = "Above average","D"
        elif co2 < 300: rating,cls = "Above average","E"
        else:           rating,cls = "High emissions","F"
        return jsonify({"co2_gkm": co2, "annual_co2_kg": annual,
                        "rating": rating, "emission_class": cls,
                        "vs_average_gkm": round(co2-249, 1)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
