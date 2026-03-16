import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("combined_satellite_buoy_with_chl.csv")

# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------
df = df.dropna(subset=[
    "pco2_sw_sat",
    "chlor_a",
    "satellite_sst_mean",
    "xco2_sw_dry"
])

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# seasonal encoding
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

# log chlorophyll (important)
df["log_chl"] = np.log10(df["chlor_a"] + 1e-6)

# --------------------------------------------------
# FEATURES + TARGET
# --------------------------------------------------
features = [
    "satellite_sst_mean",
    "log_chl",
    "xco2_sw_dry",
    "sin_month",
    "cos_month"
]

X = df[features]
y = df["pco2_sw_sat"]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R²:", r2)
print("RMSE:", rmse)

# --------------------------------------------------
# PLOT: Predicted vs Observed
# --------------------------------------------------
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Observed pCO2")
plt.ylabel("Predicted pCO2")
plt.title("Predicted vs Observed pCO2")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.figtext(0.5, 0.05, f"R^2: {r2}", ha = "center", fontsize=12, color="red") 
plt.show()

