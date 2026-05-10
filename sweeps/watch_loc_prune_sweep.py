"""Small watch-location pruning/threshold sweep.

This is intentionally separate from src/train_watch_loc.py so leaderboard
experiments can be compared before promoting a change to the production model.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_watch_loc import FEATURE_COLS, TwoStageWatchLoc, engineer_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, "results", "features_train.csv")

RANDOM_STATE = 42
N_FOLDS = 5


def make_pipe(clf):
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", clf),
        ]
    )


def make_two_stage():
    stage1 = make_pipe(
        SVC(
            kernel="rbf",
            C=1.03,
            gamma=0.019,
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
    )
    stage2 = make_pipe(
        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=3,
            min_samples_leaf=4,
            subsample=0.9,
            random_state=RANDOM_STATE,
        )
    )
    return TwoStageWatchLoc(stage1=stage1, stage2=stage2)


def eval_model(X_df, y, cols=None, thresholds=None):
    X = X_df[cols].values if cols is not None else X_df.values
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    per_class = {0: [], 1: [], 2: []}

    for tr, va in cv.split(X, y):
        model = make_two_stage()
        model.fit(X[tr], y[tr])
        if thresholds is None:
            pred = model.predict(X[va])
        else:
            proba = model.predict_proba(X[va])
            pred = proba.argmax(axis=1)
            for i, p in enumerate(proba):
                candidates = [k for k, t in thresholds.items() if p[k] >= t]
                if candidates:
                    pred[i] = max(candidates, key=lambda k: p[k] / thresholds[k])

        scores.append(balanced_accuracy_score(y[va], pred))
        cm = confusion_matrix(y[va], pred, labels=[0, 1, 2])
        for k in [0, 1, 2]:
            per_class[k].append(cm[k, k] / cm[k].sum())

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        {k: float(np.mean(v)) for k, v in per_class.items()},
    )


def separation_ranking(X_df, y):
    scores = {}
    for feat in X_df.columns:
        vals = [X_df.loc[y == k, feat].values for k in [0, 1, 2]]
        means = [v.mean() for v in vals]
        stds = [v.std() for v in vals]
        scores[feat] = np.std(means) / (np.mean(stds) + 1e-9)
    return sorted(X_df.columns, key=scores.get, reverse=True)


def gbm_ranking(X_df, y):
    model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=385,
                    learning_rate=0.086,
                    max_depth=3,
                    min_samples_leaf=4,
                    subsample=0.95,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_df.values, y)
    importances = model.named_steps["clf"].feature_importances_
    return [X_df.columns[i] for i in np.argsort(importances)[::-1]]


def print_result(name, result):
    mean, std, pc = result
    print(
        f"{name:<22} {mean:>7.4f} +/- {std:.4f}  "
        f"wrist={pc[0]:.3f} belt={pc[1]:.3f} ankle={pc[2]:.3f}"
    )


def main():
    df = engineer_features(pd.read_csv(FEATURES_TRAIN))
    X_df = df[FEATURE_COLS].astype(float)
    y = df["watch_loc"].values.astype(int)

    print(f"Samples: {len(df)}  Features: {len(FEATURE_COLS)}")
    print_result("baseline", eval_model(X_df, y))

    rankings = {
        "sep": separation_ranking(X_df, y),
        "gbm": gbm_ranking(X_df, y),
    }
    for rank_name, ranking in rankings.items():
        for k in [25, 30, 35, 40, 45, 50, 55, 60, 65, 69]:
            print_result(f"{rank_name} top{k:02d}", eval_model(X_df, y, ranking[:k]))

    print("\nThresholds >= baseline:")
    for tw in [0.45, 0.50, 0.55]:
        for tb in [0.45, 0.50, 0.55]:
            for ta in [0.45, 0.50, 0.55]:
                thresholds = {0: tw, 1: tb, 2: ta}
                result = eval_model(X_df, y, thresholds=thresholds)
                if result[0] >= 0.9644:
                    print_result(str(thresholds), result)


if __name__ == "__main__":
    main()
