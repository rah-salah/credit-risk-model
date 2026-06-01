import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import logging
import warnings
import os
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rfm(df):
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )
    snapshot_date = df["TransactionStartTime"].max()
    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime",
                 lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()
    logger.info(f"RFM calculated for {len(rfm):,} customers")
    return rfm


def create_proxy_target(df, n_clusters=3, random_state=42):
    logger.info("Creating proxy target variable using K-Means...")
    rfm = calculate_rfm(df)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    )
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    cluster_summary = rfm.groupby("Cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    })
    cluster_summary["risk_score"] = (
        cluster_summary["Recency"]
        - cluster_summary["Frequency"]
        - cluster_summary["Monetary"]
    )
    high_risk_cluster = cluster_summary["risk_score"].idxmax()
    logger.info(f"High risk cluster: {high_risk_cluster}")
    logger.info(f"Cluster summary:\n{cluster_summary}")
    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)
    count = rfm["is_high_risk"].sum()
    logger.info(
        f"High risk customers: {count:,} "
        f"({count/len(rfm)*100:.1f}%)"
    )
    return rfm[["CustomerId", "Recency", "Frequency",
                "Monetary", "Cluster", "is_high_risk"]]


def merge_target_with_transactions(df, rfm_with_target):
    df = df.merge(
        rfm_with_target[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )
    df["is_high_risk"] = df["is_high_risk"].fillna(0).astype(int)
    logger.info(f"Merged shape: {df.shape}")
    return df


def prepare_features(df):
    drop_cols = [
        "TransactionId", "BatchId", "AccountId",
        "SubscriptionId", "CustomerId", "CurrencyCode",
        "CountryCode", "TransactionStartTime", "Cluster"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    target_col = "is_high_risk"
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            "Run create_proxy_target first."
        )
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and c != target_col
    ]
    X = df[feature_cols].select_dtypes(include=np.number)
    y = df[target_col]
    logger.info(f"Features: {X.shape[1]} columns")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"{'='*50}")
    return metrics


def train_models(df):
    logger.info("Preparing features...")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    mlflow.set_experiment("credit_risk_model")

    with mlflow.start_run(run_name="LogisticRegression"):
        logger.info("Training Logistic Regression...")
        lr = GridSearchCV(
            LogisticRegression(
                class_weight="balanced", random_state=42
            ),
            {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [1000]},
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        lr.fit(X_train, y_train)
        best_lr = lr.best_estimator_
        metrics = evaluate_model(
            best_lr, X_test, y_test, "Logistic Regression"
        )
        mlflow.log_params(lr.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_lr, "logistic_regression")

    with mlflow.start_run(run_name="DecisionTree"):
        logger.info("Training Decision Tree...")
        dt = GridSearchCV(
            DecisionTreeClassifier(
                class_weight="balanced", random_state=42
            ),
            {"max_depth": [3, 5, 7, 10],
             "min_samples_split": [2, 5, 10]},
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        dt.fit(X_train, y_train)
        best_dt = dt.best_estimator_
        metrics = evaluate_model(
            best_dt, X_test, y_test, "Decision Tree"
        )
        mlflow.log_params(dt.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_dt, "decision_tree")

    with mlflow.start_run(run_name="RandomForest"):
        logger.info("Training Random Forest...")
        rf = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42, n_jobs=-1
            ),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
            cv=5, scoring="roc_auc", n_jobs=-1
        )
        rf.fit(X_train, y_train)
        best_rf = rf.best_estimator_
        metrics = evaluate_model(
            best_rf, X_test, y_test, "Random Forest"
        )
        mlflow.log_params(rf.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "random_forest")

    logger.info("All models trained!")
    return best_lr, best_dt, best_rf, X_test, y_test


if __name__ == "__main__":
    data_path = "data/raw/data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape[0]:,} rows")

    rfm_target = create_proxy_target(df)
    df = merge_target_with_transactions(df, rfm_target)

    from src.data_processing import build_pipeline
    pipeline = build_pipeline()
    df_processed = pipeline.fit_transform(df)

    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv(
        "data/processed/processed_data.csv", index=False
    )
    logger.info("Processed data saved!")

    train_models(df_processed)
