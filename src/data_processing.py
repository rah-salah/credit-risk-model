import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# INDIVIDUAL TRANSFORMERS
# Each one does ONE job — this is the modular pipeline approach
# ══════════════════════════════════════════════════════════════════

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    STEP 1: Create customer-level aggregate features.

    EDA showed 3,633 unique customers with high variance
    (avg 26.3 transactions, std 47.2). These features
    capture each customer's overall behavioral pattern
    which is essential for RFM-based risk labeling.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        required = ["CustomerId", "TransactionId", "Amount", "Value"]
        missing_cols = [c for c in required if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for aggregation: {missing_cols}"
            )

        logger.info("Creating customer-level aggregate features...")
        X = X.copy()

        agg = X.groupby("CustomerId").agg(
            transaction_count=("TransactionId", "count"),
            total_spend=("Amount", "sum"),
            avg_value=("Value", "mean"),
            std_value=("Value", "std"),
            max_value=("Value", "max"),
            min_value=("Value", "min"),
        ).reset_index()

        agg["std_value"] = agg["std_value"].fillna(0)
        X = X.merge(agg, on="CustomerId", how="left")
        logger.info(f"Aggregate features created. Shape: {X.shape}")
        return X


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    STEP 2: Extract datetime features.

    EDA showed transaction peaks at 9am-11am and 6pm-8pm
    and higher volumes in Nov-Dec. Hour, day of week, and
    month are therefore strong behavioral signals.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "TransactionStartTime" not in X.columns:
            logger.warning(
                "TransactionStartTime not found. "
                "Skipping datetime extraction."
            )
            return X

        logger.info("Extracting datetime features...")
        X = X.copy()

        try:
            X["TransactionStartTime"] = pd.to_datetime(
                X["TransactionStartTime"], errors="coerce"
            )
            null_dates = X["TransactionStartTime"].isnull().sum()
            if null_dates > 0:
                logger.warning(
                    f"{null_dates} rows could not be parsed as datetime."
                )

            X["tx_hour"] = X["TransactionStartTime"].dt.hour
            X["tx_day"] = X["TransactionStartTime"].dt.day
            X["tx_month"] = X["TransactionStartTime"].dt.month
            X["tx_year"] = X["TransactionStartTime"].dt.year
            X["tx_dayofweek"] = X["TransactionStartTime"].dt.dayofweek

        except Exception as e:
            logger.error(f"Datetime extraction failed: {e}")
            raise

        logger.info("Datetime features extracted.")
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    STEP 3: Handle missing values.

    EDA confirmed zero missing values currently.
    Defensive imputation implemented for future data batches.
    - Numerical: fill with median (robust to outliers)
    - Categorical: fill with mode (most common value)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Handling missing values...")
        X = X.copy()

        for col in X.select_dtypes(include=np.number).columns:
            null_count = X[col].isnull().sum()
            if null_count > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                logger.info(
                    f"Filled {null_count} nulls in '{col}' "
                    f"with median ({median_val:.2f})"
                )

        for col in X.select_dtypes(include="object").columns:
            null_count = X[col].isnull().sum()
            if null_count > 0:
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                logger.info(
                    f"Filled {null_count} nulls in '{col}' "
                    f"with mode ('{mode_val}')"
                )

        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    STEP 4: Log transform skewed columns.

    EDA confirmed Amount and Value are severely right-skewed
    (skewness > 10, mean 4,579 vs median 1,000 UGX).
    log1p compresses extreme values for linear models.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "Value" not in X.columns:
            logger.warning("Value column not found. Skipping log transform.")
            return X

        logger.info("Applying log transformation...")
        X = X.copy()
        X["Amount_log"] = np.log1p(X["Value"].abs())
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    STEP 5: Encode categorical columns.

    EDA showed ProductCategory has 9 unique values dominated
    by airtime (46%). Label encoding converts text to integers.
    Unknown categories at prediction time are mapped to -1.
    """

    def __init__(self):
        self.label_encoders = {}
        self.cat_columns = [
            "ProductCategory", "ChannelId",
            "ProviderId", "PricingStrategy"
        ]

    def fit(self, X, y=None):
        for col in self.cat_columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
                logger.info(
                    f"Fitted encoder for '{col}' — "
                    f"{len(le.classes_)} unique values"
                )
        return self

    def transform(self, X):
        logger.info("Encoding categorical columns...")
        X = X.copy()

        for col in self.cat_columns:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found. Skipping.")
                continue

            try:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X[col] = X[col].astype(str).map(
                        lambda x, le=le: (
                            int(le.transform([x])[0])
                            if x in le.classes_ else -1
                        )
                    )
                else:
                    logger.warning(
                        f"No fitted encoder for '{col}'. Skipping."
                    )
            except Exception as e:
                logger.error(f"Encoding failed for '{col}': {e}")
                raise

        return X


class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    STEP 6: Weight of Evidence (WoE) encoding.

    WoE is a banking industry standard technique that measures
    how strongly each category predicts the target variable.
    IV (Information Value) measures overall column predictiveness:
    - IV < 0.02  → weak predictor → can be dropped
    - IV 0.02-0.1 → weak but useful
    - IV 0.1-0.3  → medium predictor
    - IV > 0.3    → strong predictor

    This is required for Basel II compliant Logistic Regression.
    """

    def __init__(self, target_col="is_high_risk"):
        self.target_col = target_col
        self.woe_maps = {}
        self.iv_scores = {}
        self.cat_columns = [
            "ProductCategory", "ChannelId",
            "ProviderId", "PricingStrategy"
        ]

    def _calculate_woe_iv(self, X, col, target):
        """Calculate WoE and IV for one column."""
        df = pd.DataFrame({col: X[col], "target": target})
        total_events = target.sum()
        total_non_events = (1 - target).sum()

        if total_events == 0 or total_non_events == 0:
            logger.warning(
                f"Column '{col}' has only one class. Skipping WoE."
            )
            return {}, 0

        grouped = df.groupby(col)["target"].agg(
            ["sum", "count"]
        ).reset_index()
        grouped.columns = [col, "events", "total"]
        grouped["non_events"] = grouped["total"] - grouped["events"]

        # Avoid division by zero
        grouped["dist_events"] = (
            grouped["events"] / total_events
        ).replace(0, 0.0001)
        grouped["dist_non_events"] = (
            grouped["non_events"] / total_non_events
        ).replace(0, 0.0001)

        grouped["woe"] = np.log(
            grouped["dist_events"] / grouped["dist_non_events"]
        )
        grouped["iv"] = (
            (grouped["dist_events"] - grouped["dist_non_events"])
            * grouped["woe"]
        )

        woe_map = dict(zip(grouped[col], grouped["woe"]))
        iv_score = grouped["iv"].sum()
        return woe_map, iv_score

    def fit(self, X, y=None):
        if y is None:
            if self.target_col in X.columns:
                y = X[self.target_col]
            else:
                logger.warning(
                    f"Target column '{self.target_col}' not found. "
                    "Skipping WoE fitting."
                )
                return self

        logger.info("Calculating WoE and IV scores...")
        for col in self.cat_columns:
            if col in X.columns:
                woe_map, iv = self._calculate_woe_iv(X, col, y)
                self.woe_maps[col] = woe_map
                self.iv_scores[col] = iv
                logger.info(f"  {col}: IV = {iv:.4f}")

        return self

    def transform(self, X):
        logger.info("Applying WoE encoding...")
        X = X.copy()

        for col in self.cat_columns:
            if col in X.columns and col in self.woe_maps:
                woe_col = f"{col}_woe"
                X[woe_col] = X[col].map(self.woe_maps[col]).fillna(0)
                logger.info(f"WoE encoded: {woe_col}")

        return X

    def get_iv_summary(self):
        """Return a summary of IV scores for all columns."""
        if not self.iv_scores:
            return pd.DataFrame()
        iv_df = pd.DataFrame(
            list(self.iv_scores.items()),
            columns=["Feature", "IV"]
        ).sort_values("IV", ascending=False)
        iv_df["Predictiveness"] = iv_df["IV"].apply(
            lambda x: "Strong" if x > 0.3
            else "Medium" if x > 0.1
            else "Weak" if x > 0.02
            else "Very Weak"
        )
        return iv_df


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    STEP 7: Scale numerical features.

    EDA showed extreme value ranges (Amount: -1M to +9.8M UGX).
    StandardScaler (mean=0, std=1) used for Logistic Regression.
    MinMaxScaler (0 to 1) available for tree-based models.
    """

    def __init__(self, scaler_type="standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.scale_cols = []

    def fit(self, X, y=None):
        candidate_cols = [
            "Amount", "Value", "Amount_log",
            "transaction_count", "total_spend",
            "avg_value", "std_value", "max_value", "min_value",
            "tx_hour", "tx_day", "tx_month", "tx_dayofweek"
        ]
        self.scale_cols = [c for c in candidate_cols if c in X.columns]

        if not self.scale_cols:
            logger.warning("No columns found to scale.")
            return self

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        self.scaler.fit(X[self.scale_cols])
        logger.info(f"Fitted scaler on {len(self.scale_cols)} columns.")
        return self

    def transform(self, X):
        if self.scaler is None or not self.scale_cols:
            logger.warning("Scaler not fitted. Skipping scaling.")
            return X

        logger.info(f"Scaling {len(self.scale_cols)} columns...")
        X = X.copy()

        try:
            X[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise

        return X


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE BUILDER
# This chains all transformers into one sklearn Pipeline
# ══════════════════════════════════════════════════════════════════

def build_pipeline(scaler_type="standard", target_col="is_high_risk"):
    """
    Build and return the full feature engineering pipeline.

    A Pipeline is like a conveyor belt — data goes in one end
    and comes out the other end fully transformed and ready
    for the machine learning model.

    Usage:
        pipeline = build_pipeline()
        X_transformed = pipeline.fit_transform(df)
    """
    pipeline = Pipeline(steps=[
        ("aggregate", AggregateFeatures()),
        ("datetime", DatetimeFeatures()),
        ("missing", MissingValueHandler()),
        ("log_transform", LogTransformer()),
        ("encoder", CategoricalEncoder()),
        ("woe", WoEEncoder(target_col=target_col)),
        ("scaler", FeatureScaler(scaler_type=scaler_type)),
    ])
    return pipeline


# ══════════════════════════════════════════════════════════════════
# LEGACY CLASS — kept for backward compatibility with tests
# ══════════════════════════════════════════════════════════════════

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Wrapper class that uses the full pipeline internally.
    Kept for backward compatibility with existing tests.
    """

    def __init__(self, scaler_type="standard"):
        self.scaler_type = scaler_type
        self._pipeline = None
        self._is_fitted = False

    def fit_transform(self, df, y=None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(df)}"
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        self._pipeline = build_pipeline(
            scaler_type=self.scaler_type
        )
        result = df.copy()

        # Run each step manually to handle the full df
        for name, transformer in self._pipeline.steps:
            result = transformer.fit_transform(result)

        self._is_fitted = True
        return result

    def transform(self, df):
        if not self._is_fitted:
            raise ValueError(
                "Pipeline not fitted. Call fit_transform first."
            )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(df)}"
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        result = df.copy()
        for name, transformer in self._pipeline.steps:
            result = transformer.transform(result)
        return result
