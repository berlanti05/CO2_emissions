
import pickle, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT_DIR = "."
file_path = os.path.join(PROJECT_ROOT_DIR, "datasets", "CO2_Emissions.csv")

df = pd.read_csv(file_path)
df.columns = ["make","model","vehicle_class","engine_size","cylinders","transmission",
              "fuel_type","fuel_consumption_city","fuel_consumption_hwy",
              "fuel_consumption_comb","fuel_consumption_mpg","co2_emissions"]

df = df.drop_duplicates()

for col in ["engine_size","cylinders","fuel_consumption_city","fuel_consumption_hwy",
            "fuel_consumption_comb","fuel_consumption_mpg"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["vehicle_class","fuel_type","transmission"]:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop(columns=["make","model"])

df_encoded = pd.get_dummies(df, columns=["vehicle_class","fuel_type","transmission"])

X = df_encoded.drop("co2_emissions", axis=1)
y = df_encoded["co2_emissions"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(np.mean((y_test-y_pred)**2)):.4f}")

with open("co2_model.pkl","wb") as f:
    pickle.dump({"model": rf_model, "scaler": scaler,
                 "feature_columns": feature_columns}, f)

print(f"\n✅ Saved → co2_model.pkl  ({len(feature_columns)} features)")
print("Columns:", feature_columns)
