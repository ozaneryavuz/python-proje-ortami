import os
import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import gp_minimize
from skopt.space import Real
import optuna

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def load_data(path: str) -> pd.DataFrame:
    """Load and clean data from Excel file."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.translate(str.maketrans("ıİşŞöÖüÜÇçğĞ", "iIsSoOuUCcgG"))
    df = df[["Girdi 1", "Girdi 2", "Girdi 3", "Cikti 2"]].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply domain specific feature engineering."""
    df_fe = df.copy()
    o1, o2, o3 = "Girdi 1", "Girdi 2", "Girdi 3"
    df_fe["diff_ev_dep"] = df[o1] - df[o3]
    df_fe["mean_odds"] = df[[o1, o2, o3]].mean(axis=1, numeric_only=True)
    df_fe["logit_ev"] = np.log(df[o1] / df[[o2, o3]].mean(axis=1, numeric_only=True))
    df_fe["imp1"] = 1 / df[o1]
    df_fe["impX"] = 1 / df[o2]
    df_fe["imp2"] = 1 / df[o3]
    df_fe["overround"] = df_fe["imp1"] + df_fe["impX"] + df_fe["imp2"]
    return df_fe

def tune_xgb(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """Hyper parameter tuning for XGBoost using Optuna."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(X, y)
        preds = model.predict_proba(X)
        log_loss = -np.mean(np.log(preds[np.arange(len(y)), y]))
        return log_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    return XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)

def train_pipeline(df: pd.DataFrame):
    """Train a stacking model and preprocessing pipeline."""
    X = df.drop("Cikti 2", axis=1)
    y = df["Cikti 2"].astype(str).map({"1": 0, "X": 1, "2": 2}).astype(int)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
    ])
    X_transformed = pipeline.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    xgb = tune_xgb(X_transformed, y)
    lgbm = LGBMClassifier(n_estimators=150, random_state=42)

    stack = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        n_jobs=-1
    )
    stack.fit(X_transformed, y)
    return pipeline, stack

def optimize_threshold(model, X: np.ndarray, y: np.ndarray, odds: np.ndarray) -> float:
    """Optimize probability threshold for betting ROI."""
    def roi_loss(threshold):
        probs = model.predict_proba(X)
        total_profit = 0
        total_bet = 0
        for i, prob in enumerate(probs):
            expected_values = prob * odds[i]
            bet_class = np.argmax(expected_values)
            if expected_values[bet_class] >= threshold[0]:
                total_bet += 1
                if bet_class == y[i]:
                    total_profit += odds[i, bet_class] - 1
                else:
                    total_profit -= 1
        if total_bet == 0:
            return 1
        roi = total_profit / total_bet
        return -roi

    res = gp_minimize(roi_loss, [Real(1.0, 2.5)], n_calls=25, random_state=42)
    return res.x[0]

def main() -> None:
    """Run ROI based betting model training and evaluation."""
    df = load_data("Girdi ve Ciktilar.xlsx")
    df = feature_engineering(df)
    X = df.drop("Cikti 2", axis=1)
    y = df["Cikti 2"].astype(str).map({"1": 0, "X": 1, "2": 2}).astype(int)
    odds = df[["Girdi 1", "Girdi 2", "Girdi 3"]].values

    X_train, X_test, y_train, y_test, odds_train, odds_test = train_test_split(
        X, y, odds, test_size=0.3, random_state=42)

    pipeline, model = train_pipeline(pd.concat([X_train, y_train], axis=1))
    X_scaled_test = pipeline.transform(X_test)

    threshold = optimize_threshold(model, pipeline.transform(X_train), y_train.values, odds_train)

    probs = model.predict_proba(X_scaled_test)
    evs = probs * odds_test
    bet_class = np.argmax(evs, axis=1)
    bets = evs[np.arange(len(bet_class)), bet_class] >= threshold

    profit = 0
    total_bet = 0
    wins = 0
    for i in range(len(y_test)):
        if bets[i]:
            total_bet += 1
            if bet_class[i] == y_test.values[i]:
                profit += odds_test[i, bet_class[i]] - 1
                wins += 1
            else:
                profit -= 1

    roi = profit / total_bet if total_bet > 0 else 0
    acc = wins / total_bet if total_bet > 0 else 0

    print(f"Toplam Kar: {profit:.2f} | Oynanan Bahis: {total_bet} | ROI: {roi:.2%} | Kazanma Oranı: {acc:.2%}")

    results = pd.DataFrame({
        "Actual": y_test,
        "BetClass": bet_class,
        "Bet": bets
    })
    results.to_excel("Optimum_Tahminler_Karlilik.xlsx", index=False)
    logging.info("✅ Karlılık odaklı tahmin dosyası oluşturuldu: Optimum_Tahminler_Karlilik.xlsx")

if __name__ == "__main__":
    main()
