from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audit_workspace" / "rich_health" / "RC+Health+Care+Data-20180820.xlsx"
OUT = ROOT / "unified_rich_model" / "output"
FEATURES = ["age", "male", "bmi", "fpg_mmol_l"]
HORIZON = 3.0


def source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(path: Path):
    raw = pd.read_excel(path, sheet_name="RC")
    data = pd.DataFrame({
        "participant_id": raw["id"].astype(int),
        "time_years": raw["year of followup"].astype(float),
        "event": raw["censor of diabetes at followup(1, Yes; 0, No)"].astype(int),
        "age": raw["Age (y)"].astype(float),
        "male": (raw["Gender(1, male; 2, female)"].astype(int) == 1).astype(int),
        "bmi": raw["BMI(kg/m2)"].astype(float),
        "fpg_mmol_l": raw["FPG (mmol/L)"].astype(float),
    })
    # Exclude physiologically implausible source values and diabetes-range FPG.
    data = data.loc[data.fpg_mmol_l.between(2.8, 6.999999)].reset_index(drop=True)
    if data.isna().any().any():
        raise ValueError("Missing values in core predictors or outcome")
    return data


def censor_survival_function(time, event):
    censor_times = np.sort(np.unique(time[event == 0]))
    survival = []
    current = 1.0
    for value in censor_times:
        current *= 1 - (((time == value) & (event == 0)).sum() / (time >= value).sum())
        survival.append(current)
    survival = np.asarray(survival)

    def evaluate(query):
        positions = np.searchsorted(censor_times, np.asarray(query), side="left") - 1
        return np.where(positions < 0, 1.0, survival[np.maximum(positions, 0)])
    return evaluate


def model():
    return HistGradientBoostingClassifier(
        max_iter=180,
        max_leaf_nodes=12,
        min_samples_leaf=50,
        l2_regularization=5,
        learning_rate=0.05,
        monotonic_cst=[1, 0, 1, 1],
        random_state=20260920,
    )


def weighted_calibration(y, prediction, weight):
    logit = np.log(np.clip(prediction, 1e-8, 1 - 1e-8) /
                   np.clip(1 - prediction, 1e-8, 1))
    fit = LogisticRegression(C=1e8, max_iter=1000).fit(
        logit.reshape(-1, 1), y, sample_weight=weight
    )
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def metric_row(name, mask, y, prediction, weight):
    yy, pp, ww = y[mask], prediction[mask], weight[mask]
    return {
        "group": name,
        "n_evaluable": int(mask.sum()),
        "cases": int(yy.sum()),
        "ipcw_observed_risk": float(np.average(yy, weights=ww)),
        "ipcw_mean_predicted_risk": float(np.average(pp, weights=ww)),
        "ipcw_auc": float(roc_auc_score(yy, pp, sample_weight=ww)),
        "ipcw_average_precision": float(average_precision_score(yy, pp, sample_weight=ww)),
        "ipcw_brier": float(brier_score_loss(yy, pp, sample_weight=ww)),
    }


def bootstrap_ci(y, prediction, weight, repeats=1000):
    rng = np.random.default_rng(20260920)
    aucs, briers = [], []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y[index], prediction[index], sample_weight=weight[index]))
        briers.append(brier_score_loss(y[index], prediction[index], sample_weight=weight[index]))
    return {
        "auc_95_ci": [float(x) for x in np.quantile(aucs, [.025, .975])],
        "brier_95_ci": [float(x) for x in np.quantile(briers, [.025, .975])],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = prepare(SOURCE)
    time = data.time_years.to_numpy(float)
    event = data.event.to_numpy(int)
    known = ((event == 1) & (time <= HORIZON)) | (time > HORIZON)
    evaluable = data.loc[known].reset_index(drop=True)
    y = ((evaluable.event == 1) & (evaluable.time_years <= HORIZON)).astype(int).to_numpy()
    x = evaluable[FEATURES].to_numpy(float)

    # IPCW uses the complete eligible cohort's censoring distribution.
    censor_survival = censor_survival_function(time, event)
    query_time = np.where(y == 1, evaluable.time_years.to_numpy(), HORIZON)
    weight = 1 / np.clip(censor_survival(query_time), .05, None)

    folds = StratifiedKFold(5, shuffle=True, random_state=20260920)
    oof = np.zeros(len(evaluable))
    for train, test in folds.split(x, y):
        fitted = model().fit(x[train], y[train], sample_weight=weight[train])
        oof[test] = fitted.predict_proba(x[test])[:, 1]

    fpg = evaluable.fpg_mmol_l.to_numpy()
    age = evaluable.age.to_numpy()
    male = evaluable.male.to_numpy()
    bmi = evaluable.bmi.to_numpy()
    groups = {
        "all": np.ones(len(y), dtype=bool),
        "fpg_below_100": fpg < 5.6,
        "fpg_100_109": (fpg >= 5.6) & (fpg < 6.1),
        "fpg_110_125": fpg >= 6.1,
        "women": male == 0,
        "men": male == 1,
        "age_20_29": age < 30,
        "age_30_49": (age >= 30) & (age < 50),
        "age_50_69": (age >= 50) & (age < 70),
        "age_70_plus": age >= 70,
        "bmi_below_25": bmi < 25,
        "bmi_25_29": (bmi >= 25) & (bmi < 30),
        "bmi_30_plus": bmi >= 30,
    }
    rows = [metric_row(name, mask, y, oof, weight) for name, mask in groups.items()]
    intercept, slope = weighted_calibration(y, oof, weight)
    report = {
        "model": "unified_rich_monotonic_hist_gradient_boosting_3y",
        "target": "incident diabetes by approximately three years",
        "features": FEATURES,
        "cohort": {
            "eligible_n": int(len(data)),
            "three_year_evaluable_n": int(len(evaluable)),
            "cases_by_three_years": int(y.sum()),
            "kaplan_meier_risk_3y": float(rows[0]["ipcw_observed_risk"]),
        },
        "validation": {
            "method": "five-fold out-of-fold with IPCW for right censoring",
            **rows[0],
            **bootstrap_ci(y, oof, weight),
            "calibration_intercept_ideal_0": intercept,
            "calibration_slope_ideal_1": slope,
            "subgroups": rows[1:],
        },
        "evidence_ranges": {
            key: {
                "observed": [float(data[key].min()), float(data[key].max())],
                "central_1_99_percent": [float(data[key].quantile(.01)), float(data[key].quantile(.99))],
            } for key in ["age", "bmi", "fpg_mmol_l"]
        },
        "source_sha256": source_hash(SOURCE),
    }
    final_model = model().fit(x, y, sample_weight=weight)
    joblib.dump(final_model, OUT / "unified_rich_3y_model.joblib")
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "participant_id": evaluable.participant_id,
        "time_years": evaluable.time_years,
        "event": evaluable.event,
        "ipcw_weight": weight,
        "oof_risk_3y": oof,
    }).to_csv(OUT / "oof_predictions.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
