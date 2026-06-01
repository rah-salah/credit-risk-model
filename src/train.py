import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import logging
import warnings
import os
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# TASK 4 — RFM CALCULATION AND PROXY TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════

def calculate_rfm(df):
    """
    Calculate Recency, Frequency, Monetary per customer.

    Think of it like scoring every customer on 3 questions:
    1. When did they last buy? (Recency — lower = more recent = better)
    2. How often do they buy? (Frequency — higher = better)
    3. How much do they spend? (Monetary — higher = better)

    We use the most recent transaction date as our snapshot date.
    """
    if "TransactionStartTime" not in df.columns:
        raise ValueError("TransactionStartTime column is required for RFM")
    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId column is required for RFM")

    logger.info("Calculating RFM features...")

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )

    # Snapshot date = the most recent transaction in the dataset
    snapshot_date = df["TransactionStartTime"].max()
    logger.info(f"Snapshot date: {snapshot_date}")

    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime",
                 lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    logger.info(f"RFM calculated for {len(rfm):,} unique customers")
    logger.info(f"RFM Summary:\n{rfm.describe().round(2)}")

    return rfm


def create_proxy_target(df, n_clusters=3, random_state=42):
    """
    Use K-Means clustering on RFM features to create
    the is_high_risk proxy target variable.
    """
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

    logger.info(f"Cluster summary:\n{cluster_summary}")

    cluster_summary["risk_score"] = (
        cluster_summary["Recency"] -
        cluster_summary["Frequency"] -
        cluster_summary["Monetary"]
    )

    high_risk_cluster = cluster_summary["risk_score"].idxmax()
    logger.info(f"High risk cluster identified: Cluster {high_risk_cluster}")

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    high_risk_count = rfm["is_high_risk"].sum()
    total = len(rfm)
    logger.info(
        f"High risk customers: {high_risk_count:,} "
        f"({high_risk_count/total*100:.1f}% of {total:,} customers)"
    )

    return rfm[["CustomerId", "Recency", "Frequency",
                "Monetary", "Cluster", "is_high_risk"]]


def merge_target_with_transactions(df, rfm_with_target):
    """
    Merge the is_high_risk label back into the
    main transaction dataset.
    """
    logger.info("Merging is_high_risk label into transaction dataset...")

    df = df.merge(
        rfm_with_target[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    df["is_high_risk"] = df["is_high_risk"].fillna(0).astype(int)

    logger.info(
        f"Merged dataset shape: {df.shape}"
    )
    logger.info(
        f"High risk transactions: "
        f"{df['is_high_risk'].sum():,} "
        f"({df['is_high_risk'].mean()*100:.1f}%)"
    )

    return df


# ══════════════════════════════════════════════════════════════════
# TASK 5 — MODEL TRAINING WITH MLFLOW TRACKING
# ══════════════════════════════════════════════════════════════════

def prepare_features(df):
    """
    Select the final features and target variable
    for model training.
    """
    drop_cols = [
        "TransactionId", "BatchId", "AccountId",
        "SubscriptionId", "CustomerId", "CurrencyCode",
        "CountryCode", "TransactionStartTime",
        "Cluster"
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

    logger.info(f"Features selected: {X.shape[1]} columns")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculate all evaluation metrics for a trained model.
    """
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
    """
    Train multiple models and track everything with MLflow.
    """
    logger.info("Preparing features for training...")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]:,} rows")
    logger.info(f"Test set: {X_test.shape[0]:,} rows")

    mlflow.set_experiment("credit_risk_model")

    # ── MODEL 1: Logistic Regression ──────────────────────────────
    with mlflow.start_run(run_name="LogisticRegression"):
        logger.info("Training Logistic Regression...")

        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000]
        }

        lr = GridSearchCV(
            LogisticRegression(
                class_weight="balanced",
                random_state=42
            ),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        lr.fit(X_train, y_train)
        best_lr = lr.best_estimator_

        metrics = evaluate_model(
            best_lr, X_test, y_test, "Logistic Regression"
        )

        mlflow.log_params(lr.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_lr, "logistic_regression")

    # ── MODEL 2: Decision Tree ────────────────────────────────────
    with mlflow.start_run(run_name="DecisionTree"):
        logger.info("Training Decision Tree...")

        param_grid = {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10]
        }

        dt = GridSearchCV(
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        dt.fit(X_train, y_train)
        best_dt = dt.best_estimator_

        metrics = evaluate_model(
            best_dt, X_test, y_test, "Decision Tree"
        )

        mlflow.log_params(dt.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_dt, "decision_tree")

    # ── MODEL 3: Random Forest ────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest"):
        logger.info("Training Random Forest...")

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        }

        rf = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        best_rf = rf.best_estimator_

        metrics = evaluate_model(
            best_rf, X_test, y_test, "Random Forest"
        )

        mlflow.log_params(rf.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "random_forest")

    logger.info("All models trained and logged to MLflow!")
    return best_lr, best_dt, best_rf, X_test, y_test


# ══════════════════════════════════════════════════════════════════
# MAIN — Run everything end to end
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Loading data...")

    data_path = "data/raw/data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Please download the Xente dataset from Kaggle."
        )

    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape[0]:,} rows")

    # Step 1: Create proxy target variable
    rfm_target = create_proxy_target(df)
    df = merge_target_with_transactions(df, rfm_target)

    # Step 2: Apply feature engineering pipeline
    from src.data_processing import build_pipeline
    pipeline = build_pipeline()
    df_processed = pipeline.fit_transform(df)

    # Step 3: Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv(
        "data/processed/processed_data.csv", index=False
    )
    logger.info("Processed data saved!")

    # Step 4: Train models
    train_models(df_processed)
ENDOFFILEcd ~/credit-risk-model
cat > src/train.py << 'ENDOFFILE'
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import logging
import warnings
import os
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# TASK 4 — RFM CALCULATION AND PROXY TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════

def calculate_rfm(df):
    """
    Calculate Recency, Frequency, Monetary per customer.

    Think of it like scoring every customer on 3 questions:
    1. When did they last buy? (Recency — lower = more recent = better)
    2. How often do they buy? (Frequency — higher = better)
    3. How much do they spend? (Monetary — higher = better)

    We use the most recent transaction date as our snapshot date.
    """
    if "TransactionStartTime" not in df.columns:
        raise ValueError("TransactionStartTime column is required for RFM")
    if "CustomerId" not in df.columns:
        raise ValueError("CustomerId column is required for RFM")

    logger.info("Calculating RFM features...")

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )

    # Snapshot date = the most recent transaction in the dataset
    snapshot_date = df["TransactionStartTime"].max()
    logger.info(f"Snapshot date: {snapshot_date}")

    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime",
                 lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    logger.info(f"RFM calculated for {len(rfm):,} unique customers")
    logger.info(f"RFM Summary:\n{rfm.describe().round(2)}")

    return rfm


def create_proxy_target(df, n_clusters=3, random_state=42):
    """
    Use K-Means clustering on RFM features to create
    the is_high_risk proxy target variable.
    """
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

    logger.info(f"Cluster summary:\n{cluster_summary}")

    cluster_summary["risk_score"] = (
        cluster_summary["Recency"] -
        cluster_summary["Frequency"] -
        cluster_summary["Monetary"]
    )

    high_risk_cluster = cluster_summary["risk_score"].idxmax()
    logger.info(f"High risk cluster identified: Cluster {high_risk_cluster}")

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    high_risk_count = rfm["is_high_risk"].sum()
    total = len(rfm)
    logger.info(
        f"High risk customers: {high_risk_count:,} "
        f"({high_risk_count/total*100:.1f}% of {total:,} customers)"
    )

    return rfm[["CustomerId", "Recency", "Frequency",
                "Monetary", "Cluster", "is_high_risk"]]


def merge_target_with_transactions(df, rfm_with_target):
    """
    Merge the is_high_risk label back into the
    main transaction dataset.
    """
    logger.info("Merging is_high_risk label into transaction dataset...")

    df = df.merge(
        rfm_with_target[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    df["is_high_risk"] = df["is_high_risk"].fillna(0).astype(int)

    logger.info(
        f"Merged dataset shape: {df.shape}"
    )
    logger.info(
        f"High risk transactions: "
        f"{df['is_high_risk'].sum():,} "
        f"({df['is_high_risk'].mean()*100:.1f}%)"
    )

    return df


# ══════════════════════════════════════════════════════════════════
# TASK 5 — MODEL TRAINING WITH MLFLOW TRACKING
# ══════════════════════════════════════════════════════════════════

def prepare_features(df):
    """
    Select the final features and target variable
    for model training.
    """
    drop_cols = [
        "TransactionId", "BatchId", "AccountId",
        "SubscriptionId", "CustomerId", "CurrencyCode",
        "CountryCode", "TransactionStartTime",
        "Cluster"
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

    logger.info(f"Features selected: {X.shape[1]} columns")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculate all evaluation metrics for a trained model.
    """
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
    """
    Train multiple models and track everything with MLflow.
    """
    logger.info("Preparing features for training...")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]:,} rows")
    logger.info(f"Test set: {X_test.shape[0]:,} rows")

    mlflow.set_experiment("credit_risk_model")

    # ── MODEL 1: Logistic Regression ──────────────────────────────
    with mlflow.start_run(run_name="LogisticRegression"):
        logger.info("Training Logistic Regression...")

        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000]
        }

        lr = GridSearchCV(
            LogisticRegression(
                class_weight="balanced",
                random_state=42
            ),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        lr.fit(X_train, y_train)
        best_lr = lr.best_estimator_

        metrics = evaluate_model(
            best_lr, X_test, y_test, "Logistic Regression"
        )

        mlflow.log_params(lr.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_lr, "logistic_regression")

    # ── MODEL 2: Decision Tree ────────────────────────────────────
    with mlflow.start_run(run_name="DecisionTree"):
        logger.info("Training Decision Tree...")

        param_grid = {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10]
        }

        dt = GridSearchCV(
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        dt.fit(X_train, y_train)
        best_dt = dt.best_estimator_

        metrics = evaluate_model(
            best_dt, X_test, y_test, "Decision Tree"
        )

        mlflow.log_params(dt.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_dt, "decision_tree")

    # ── MODEL 3: Random Forest ────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest"):
        logger.info("Training Random Forest...")

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        }

        rf = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        best_rf = rf.best_estimator_

        metrics = evaluate_model(
            best_rf, X_test, y_test, "Random Forest"
        )

        mlflow.log_params(rf.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, "random_forest")

    logger.info("All models trained and logged to MLflow!")
    return best_lr, best_dt, best_rf, X_test, y_test


# ══════════════════════════════════════════════════════════════════
# MAIN — Run everything end to end
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Loading data...")

    data_path = "data/raw/data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Please download the Xente dataset from Kaggle."
        )

    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape[0]:,} rows")

    # Step 1: Create proxy target variable
    rfm_target = create_proxy_target(df)
    df = merge_target_with_transactions(df, rfm_target)

    # Step 2: Apply feature engineering pipeline
    from src.data_processing import build_pipeline
    pipeline = build_pipeline()
    df_processed = pipeline.fit_transform(df)

    # Step 3: Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv(
        "data/processed/processed_data.csv", index=False
    )
    logger.info("Processed data saved!")

    # Step 4: Train models
    train_models(df_processed)
