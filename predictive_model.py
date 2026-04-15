import warnings, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# CONFIG
EXCEL_FILE = "Final_XLL.xlsx"
SHEET_NAME = "Simulated Data"
PREDICT_YEARS = [2025, 2026, 2027]

# ─────────────────────────────
# LOAD
# ─────────────────────────────
def load_and_prepare():
    if not os.path.exists(EXCEL_FILE):
        sys.exit("❌ Excel file not found")

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    year_cols = [c for c in df.columns if c.isdigit()]
    meta_cols = [c for c in df.columns if not c.isdigit()]

    melted = df.melt(id_vars=meta_cols, value_vars=year_cols,
                     var_name="Year", value_name="Value")

    melted["Year"] = melted["Year"].astype(int)

    pivot = melted.pivot_table(
        index=["Company", "Location_HQ", "Industry", "Year"],
        columns="Type",
        values="Value",
        aggfunc="sum"
    ).reset_index()

    pivot.columns.name = None
    return pivot.fillna(0)


# ─────────────────────────────
# FEATURES
# ─────────────────────────────
def create_features(df):
    df = df.sort_values(["Company", "Year"]).copy()

    for col in ["Hiring", "Layoff", "Fund"]:
        df[f"{col}_lag1"] = df.groupby("Company")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("Company")[col].shift(2)

        df[f"{col}_rolling"] = (
            df.groupby("Company")[col]
            .rolling(2).mean().reset_index(0, drop=True)
        )

    df["Hiring_growth"] = df.groupby("Company")["Hiring"].pct_change()
    df["Layoff_growth"] = df.groupby("Company")["Layoff"].pct_change()
    df["Fund_growth"] = df.groupby("Company")["Fund"].pct_change()

    df["Year_index"] = df["Year"] - df["Year"].min()

    # FIX INF
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    le = LabelEncoder()
    df["Industry_enc"] = le.fit_transform(df["Industry"])

    return df, le


# ─────────────────────────────
# TRAIN
# ─────────────────────────────
def train(df):

    features = [
        "Year","Year_index",
        "Hiring_lag1","Hiring_lag2",
        "Layoff_lag1","Layoff_lag2",
        "Fund_lag1","Fund_lag2",
        "Hiring_rolling","Layoff_rolling","Fund_rolling",
        "Hiring_growth","Layoff_growth","Fund_growth",
        "Industry_enc"
    ]

    models = {}
    accuracy = {}

    for target in ["Hiring","Layoff","Fund"]:

        X = df[features]
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=200, random_state=42)

        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)

        pred = (rf.predict(X_scaled) + gb.predict(X_scaled)) / 2
        r2 = r2_score(y, pred)

        models[target] = {
            "rf": rf,
            "gb": gb,
            "scaler": scaler
        }

        accuracy[target] = round(r2, 3)

    return models, features, accuracy


# ─────────────────────────────
# PREDICT
# ─────────────────────────────
def predict_company(name, df, models, features, le):

    company_df = df[df["Company"].str.lower() == name.lower()]

    if company_df.empty:
        return None

    company_df = company_df.sort_values("Year")
    last = company_df.iloc[-1]

    current = last.copy()
    predictions = []

    for year in PREDICT_YEARS:

        row = {
            "Year": year,
            "Year_index": year - df["Year"].min(),

            "Hiring_lag1": current["Hiring"],
            "Hiring_lag2": current["Hiring"],

            "Layoff_lag1": current["Layoff"],
            "Layoff_lag2": current["Layoff"],

            "Fund_lag1": current["Fund"],
            "Fund_lag2": current["Fund"],

            "Hiring_rolling": current["Hiring"],
            "Layoff_rolling": current["Layoff"],
            "Fund_rolling": current["Fund"],

            "Hiring_growth": 0,
            "Layoff_growth": 0,
            "Fund_growth": 0,

            "Industry_enc": le.transform([current["Industry"]])[0]
        }

        X = pd.DataFrame([row])[features]

        preds = {}

        for target in models:
            scaler = models[target]["scaler"]
            rf = models[target]["rf"]
            gb = models[target]["gb"]

            val = (rf.predict(scaler.transform(X))[0] +
                   gb.predict(scaler.transform(X))[0]) / 2

            preds[target] = int(max(0, val))

        preds["Year"] = year
        predictions.append(preds)

        current["Hiring"] = preds["Hiring"]
        current["Layoff"] = preds["Layoff"]
        current["Fund"] = preds["Fund"]

    return {
        "history": company_df,
        "predictions": predictions
    }