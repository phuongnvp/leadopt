#!/usr/bin/env python3
"""
Build QSAR models from a CSV/TSV file.

Requirements:
    pip install pandas numpy scikit-learn joblib
    pip install rdkit
    pip install xgboost   # optional but recommended

Usage:
    python model_building.py --input Dataset.csv --output_model qsar_model.pkl

Expected columns:
    - Smiles
    - pIC50

Notes:
    - Fingerprint: Morgan (ECFP), radius=2, n_bits=2048, use_chirality=True, use_features=False
    - Split: train/val/test = 80/10/10
    - Best model is selected by validation R2 (higher is better), then validation RMSE (lower is better)
"""

import argparse
import os
import warnings
from typing import Dict, Tuple, List, Any

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Build QSAR models from SMILES and pIC50.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV/TSV file containing at least 'Smiles' and 'pIC50' columns."
    )
    parser.add_argument(
        "--output_model",
        default="final_qsar_model.pkl",
        help="Path to save final tuned model as pkl."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed."
    )
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    """
    Tries TSV first, then CSV.
    """
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] < 2:
            raise ValueError("TSV parse looks wrong.")
        return df
    except Exception:
        return pd.read_csv(path)


def smiles_to_ecfp(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
    use_features: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """
    Convert SMILES to Morgan fingerprints.
    Returns:
        X_valid: numpy array of shape (n_valid, n_bits)
        valid_idx: indices of valid SMILES in the original list
    """
    fps = []
    valid_idx = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=use_chirality,
            useFeatures=use_features,
        )

        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)

        fps.append(arr)
        valid_idx.append(i)

    if len(fps) == 0:
        raise ValueError("No valid SMILES could be parsed.")

    return np.array(fps, dtype=np.float32), valid_idx


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def evaluate_model(model, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    train_metrics = regression_metrics(y_train, pred_train)
    val_metrics = regression_metrics(y_val, pred_val)

    return {
        "model": model,
        "train_RMSE": train_metrics["RMSE"],
        "train_MAE": train_metrics["MAE"],
        "train_R2": train_metrics["R2"],
        "val_RMSE": val_metrics["RMSE"],
        "val_MAE": val_metrics["MAE"],
        "val_R2": val_metrics["R2"],
    }


def build_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=random_state))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=random_state, max_iter=10000))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(random_state=random_state, max_iter=10000))
        ]),
        "SVR_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf"))
        ]),
        "SVR_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="linear"))
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor())
        ]),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        ),
        "AdaBoost": AdaBoostRegressor(
            random_state=random_state,
            n_estimators=200
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

    # Optional XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1
        )
    except Exception:
        print("[INFO] xgboost is not installed. Skipping XGBoost.")

    return models


def get_param_grid(model_name: str, random_state: int = 42) -> Dict[str, List[Any]]:
    """
    Grid for the best model only.
    Keys must match the estimator/pipeline parameter names.
    """
    grids = {
        "LinearRegression": {},
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "Lasso": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        },
        "ElasticNet": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "SVR_RBF": {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
        },
        "SVR_Linear": {
            "model__C": [0.1, 1, 10, 100],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
        },
        "kNN": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "DecisionTree": {
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "RandomForest": {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            "loss": ["linear", "square", "exponential"],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "subsample": [0.8, 1.0],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
    }
    return grids.get(model_name, {})


def print_split_info(X_train, X_val, X_test):
    print("\n===== DATA SPLIT =====")
    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")


def print_comparison_table(results_df: pd.DataFrame):
    print("\n===== MODEL COMPARISON (TRAIN / VAL) =====")
    print(results_df.to_string(index=False))


def main():
    args = parse_args()
    random_state = args.random_state

    # -----------------------------
    # 1) Read and validate data
    # -----------------------------
    df = read_table(args.input)

    required_cols = ["Smiles", "pIC50"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["Smiles", "pIC50"]].copy()
    df = df.dropna(subset=["Smiles", "pIC50"])

    # Make sure pIC50 is numeric
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df.dropna(subset=["pIC50"]).reset_index(drop=True)

    # -----------------------------
    # 2) Build ECFP features
    # -----------------------------
    X_all, valid_idx = smiles_to_ecfp(
        df["Smiles"].tolist(),
        radius=2,
        n_bits=2048,
        use_chirality=True,
        use_features=False,
    )
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y_all = df_valid["pIC50"].values.astype(np.float32)

    print("===== INPUT SUMMARY =====")
    print(f"Original rows: {len(df)}")
    print(f"Valid SMILES:  {len(df_valid)}")
    print(f"Fingerprint:   Morgan/ECFP, radius=2, n_bits=2048, use_chirality=True, use_features=False")

    # -----------------------------
    # 3) Split into train/val/test
    #    80 / 10 / 10
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    print_split_info(X_train, X_val, X_test)

    # -----------------------------
    # 4) Train candidate models
    # -----------------------------
    models = build_candidate_models(random_state=random_state)

    results = []
    fitted_models = {}

    for name, model in models.items():
        try:
            res = evaluate_model(
                clone(model),
                X_train, y_train,
                X_val, y_val
            )
            res["Model"] = name
            fitted_models[name] = res["model"]
            results.append(res)
        except Exception as e:
            print(f"[WARN] Skipping {name} due to error: {e}")

    if len(results) == 0:
        raise RuntimeError("No model was successfully trained.")

    results_df = pd.DataFrame(results)[[
        "Model",
        "train_R2", "train_RMSE", "train_MAE",
        "val_R2", "val_RMSE", "val_MAE"
    ]]

    # Sort by validation R2 desc, then validation RMSE asc
    results_df = results_df.sort_values(
        by=["val_R2", "val_RMSE"],
        ascending=[False, True]
    ).reset_index(drop=True)

    print_comparison_table(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    print("\n===== BEST BASE MODEL =====")
    print(f"Selected by validation performance: {best_model_name}")

    # -----------------------------
    # 5) Hyperparameter tuning on validation set
    #    Train on train, select by val
    # -----------------------------
    best_base_model = clone(models[best_model_name])
    param_grid = get_param_grid(best_model_name, random_state=random_state)

    if len(param_grid) == 0:
        print(f"\n[INFO] No hyperparameter grid defined for {best_model_name}. Using base model as final.")
        best_estimator = best_base_model
        best_estimator.fit(X_train, y_train)
        best_params = {}
        best_val_score = None
    else:
        print("\n===== HYPERPARAMETER SEARCH ON VALIDATION SET =====")

        best_estimator = None
        best_params = None
        best_val_rmse = float("inf")
        best_val_score = None

        for params in ParameterGrid(param_grid):
            candidate = clone(best_base_model)
            candidate.set_params(**params)
            candidate.fit(X_train, y_train)

            y_val_pred = candidate.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_r2 = r2_score(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)

            print(
                f"Params: {params} | "
                f"Val RMSE: {val_rmse:.6f} | "
                f"Val MAE: {val_mae:.6f} | "
                f"Val R2: {val_r2:.6f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_val_score = val_r2
                best_params = params
                best_estimator = candidate

        print(f"\nBest validation RMSE: {best_val_rmse:.6f}")
        print(f"Best validation R2: {best_val_score:.6f}")
        print("Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    # -----------------------------
    # 6) Evaluate best tuned model
    # -----------------------------
    y_pred_train = best_estimator.predict(X_train)
    y_pred_val = best_estimator.predict(X_val)
    y_pred_test = best_estimator.predict(X_test)

    train_metrics = regression_metrics(y_train, y_pred_train)
    val_metrics = regression_metrics(y_val, y_pred_val)
    test_metrics = regression_metrics(y_test, y_pred_test)

    print("\n===== FINAL BEST MODEL =====")
    print(f"Model: {best_model_name}")
    if best_params:
        print("Tuned parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    final_perf = pd.DataFrame([
        {
            "Set": "Train",
            "R2": train_metrics["R2"],
            "RMSE": train_metrics["RMSE"],
            "MAE": train_metrics["MAE"],
        },
        {
            "Set": "Validation",
            "R2": val_metrics["R2"],
            "RMSE": val_metrics["RMSE"],
            "MAE": val_metrics["MAE"],
        },
        {
            "Set": "Test",
            "R2": test_metrics["R2"],
            "RMSE": test_metrics["RMSE"],
            "MAE": test_metrics["MAE"],
        },
    ])

    print("\n===== PERFORMANCE OF FINAL BEST MODEL =====")
    print(final_perf.to_string(index=False))

    # -----------------------------
    # 7) Save final model artifact for leadopt
    # -----------------------------
    if not hasattr(best_estimator, "predict"):
        raise TypeError(
            f"Final estimator of type {type(best_estimator)} does not expose predict(...). "
            "Cannot save a leadopt-compatible QSAR model artifact."
        )
    # leadopt RealQSARScorer expects the saved artifact itself to expose .predict(...)
    # Therefore, save the fitted predictor object directly as args.output_model.
    joblib.dump(best_estimator, args.output_model)
    print(f"\nSaved final predictor model to: {args.output_model}")

    # Also save a sidecar metadata package for reproducibility / reporting.
    metadata_path = os.path.splitext(args.output_model)[0] + ".meta.pkl"
    package = {
        "model_name": best_model_name,
        "fingerprint_settings": {
            "kind": "morgan",
            "radius": 2,
            "n_bits": 2048,
            "use_chirality": True,
            "use_features": False,
        },
        "target_column": "pIC50",
        "smiles_column": "Smiles",
        "best_params": best_params,
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "performance": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
    }
    joblib.dump(package, metadata_path)
    print(f"Saved metadata package to: {metadata_path}")


if __name__ == "__main__":
    main()