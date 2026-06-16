# 🚗 CarbonIQ — CO₂ Emissions Prediction System

A machine learning project that predicts vehicle CO₂ emissions based on engine specifications, fuel consumption, transmission type, and vehicle category. Built as part of an **Introduction to Machine Learning** course.

**Authors:** Tasneem Ashour, Berlanti Masalha, Batool Basalat

---

## 📌 Overview

Climate change and air pollution are among the most pressing global issues today. Vehicle CO₂ emissions are a major contributing factor. This project uses machine learning regression models to accurately predict how much CO₂ a vehicle emits (in g/km) based on its characteristics — helping governments, manufacturers, and consumers make more informed, eco-friendly decisions.

---

## 📊 Dataset

- **Size:** 7,385 vehicles × 12 features
- **Target variable:** `co2_emissions` — CO₂ emissions in g/km (continuous)
- **Features include:**

| Feature | Type |
|---|---|
| Make / Model | Categorical |
| Vehicle Class | Categorical |
| Engine Size | Numerical |
| Cylinders | Numerical |
| Transmission | Categorical |
| Fuel Type | Categorical |
| Fuel Consumption (City / Hwy / Combined / MPG) | Numerical |

---

## 🧠 Models Trained

Five regression models were trained and compared:

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | Regularized linear model |
| SVR | Support Vector Regression |
| Decision Tree | Non-linear tree-based model |
| **Random Forest** ⭐ | Ensemble model — **best performer** |

---

## 📈 Evaluation Metrics

Models were evaluated using:
- **R² Score** — Explained variance
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **Residual analysis**
- **Actual vs. Predicted plots**

### 🏆 Best Model: Random Forest Regressor

Random Forest achieved the highest R² score and the lowest MAE/RMSE among all tested models. It also showed the most balanced residual distribution, demonstrating strong generalization and stable performance.

---

## 🔑 Key Findings

- **Fuel consumption features** (city, highway, combined, MPG) were the most influential predictors of CO₂ emissions across all models.
- **Engine size** and **number of cylinders** also contributed significantly.
- Transmission type had relatively minor influence on predictions.
- The CO₂ distribution is slightly right-skewed, with most vehicles emitting between **180–300 g/km** and a mean of ~**249 g/km**.

---

## 🌐 Web Interface — CarbonIQ

The project includes an interactive prediction web app (`index.html`) powered by the trained Random Forest model.

**Features:**
- Input vehicle specs (class, engine size, cylinders, transmission, fuel type, fuel consumption)
- Get an instant CO₂ emission prediction
- Visual gauge with emission rating
- Environmental impact summary (trees equivalent, etc.)

To use it, open `index.html` in any modern browser.

---

## ⚙️ How to Run the Notebook

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd co2-emissions-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Place the dataset**
   ```
   datasets/CO2_Emissions.csv
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook CO2_Emissions.ipynb
   ```

5. **Load the saved model** (optional)
   ```python
   import pickle
   with open("random_forest_model.pkl", "rb") as f:
       model = pickle.load(f)
   ```

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Machine Learning** — scikit-learn regression models, GridSearchCV, Pipelines
- **Frontend** — HTML, CSS, JavaScript (CarbonIQ web app)

---

## 📄 License

This project was developed for academic purposes as part of an Introduction to Machine Learning course.
