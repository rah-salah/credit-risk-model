import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import warnings
warnings.filterwarnings("ignore")

# Set up logging so we can see what the pipeline is doing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline for the Bati Bank
    Credit Risk Model.

    EDA Findings that drive this pipeline:
    - Amount and Value are heavily right-skewed (skewness > 10)
      → log transformation required
    - No missing values found, but defensive imputation added
      for future data batches
    - Airtime dominates at 46% → label encoding handles this
    - Time-based patterns exist → datetime features extracted
    - High customer-level RFM variance → aggregate features needed
    """

    def __init__(self, scaler_type="standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoders = {}
        self.cat_columns = [
            "ProductCategory", "ChannelId",
            "ProviderId", "PricingStrategy"
        ]
        self.num_columns = ["Amount", "Value"]
        self._is_fitted = False

    # ── STEP 1: Aggregate customer-level features ────────────────────────
    def create_aggregate_features(self, df):
        """
        EDA showed 3,633 unique customers with high variance
        in transaction behavior (avg 26.3 tx, std 47.2).
        These aggregate features capture each customer's
        overall behavioral pattern — essential for RFM-based
        risk labeling in Task 4.
        """
        # Defensive check — required columns must exist
        required = ["CustomerId", "TransactionId", "Amount", "Value"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for aggregation: {missing_cols}"
            )

        logger.info("Creating customer-level aggregate features...")

        agg = df.groupby("CustomerId").agg(
            transaction_count=("TransactionId", "count"),
            total_spend=("Amount", "sum"),
            avg_value=("Value", "mean"),
            std_value=("Value", "std"),
            max_value=("Value", "max"),
            min_value=("Value", "min"),
        ).reset_index()

        # std is NaN for customers with only 1 transaction → fill with 0
        agg["std_value"] = agg["std_value"].fillna(0)

        df = df.merge(agg, on="CustomerId", how="left")
        logger.info(f"Aggregate features created. Shape: {df.shape}")
        return df

    # ── STEP 2: Extract datetime features ───────────────────────────────
    def extract_datetime_features(self, df):
        """
        EDA showed transaction peaks at 9am-11am and 6pm-8pm
        and higher volumes in Nov-Dec. Hour, day of week, and
        month are therefore strong behavioral signals for the model.
        """
        if "TransactionStartTime" not in df.columns:
            logger.warning(
                "TransactionStartTime column not found. "
                "Skipping datetime feature extraction."
            )
            return df

        logger.info("Extracting datetime features...")

        try:
            df["TransactionStartTime"] = pd.to_datetime(
                df["TransactionStartTime"], errors="coerce"
            )

            # Check how many failed to parse
            null_dates = df["TransactionStartTime"].isnull().sum()
            if null_dates > 0:
                logger.warning(
                    f"{null_dates} rows could not be parsed as datetime."
                )

            df["tx_hour"] = df["TransactionStartTime"].dt.hour
            df["tx_day"] = df["TransactionStartTime"].dt.day
            df["tx_month"] = df["TransactionStartTime"].dt.month
            df["tx_year"] = df["TransactionStartTime"].dt.year
            df["tx_dayofweek"] = df["TransactionStartTime"].dt.dayofweek

        except Exception as e:
            logger.error(f"Datetime extraction failed: {e}")
            raise

        logger.info("Datetime features extracted successfully.")
        return df

    # ── STEP 3: Handle missing values ────────────────────────────────────
    def handle_missing_values(self, df):
        """
        EDA confirmed zero missing values in the current dataset.
        However, defensive imputation is implemented here to handle
        future data batches that may contain nulls.
        - Numerical columns: fill with median (robust to outliers)
        - Categorical columns: fill with mode (most common value)
        """
        logger.info("Checking and handling missing values...")

        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        for col in num_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    f"Filled {null_count} missing values in "
                    f"'{col}' with median ({median_val:.2f})"
                )

        for col in cat_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(
                    f"Filled {null_count} missing values in "
                    f"'{col}' with mode ('{mode_val}')"
                )

        logger.info("Missing value handling complete.")
        return df

    # ── STEP 4: Log transform skewed columns ─────────────────────────────
    def log_transform(self, df):
        """
        EDA confirmed Amount and Value are severely right-skewed
        (skewness > 10, mean 4,579 vs median 1,000 UGX).
        log1p(x) = log(x+1) compresses extreme values and
        normalizes the distribution for linear models.
        Value.abs() is used because Amount can be negative.
        """
        if "Value" not in df.columns:
            logger.warning(
                "Value column not found. Skipping log transform."
            )
            return df

        logger.info("Applying log transformation to Value column...")
        df["Amount_log"] = np.log1p(df["Value"].abs())
        return df

    # ── STEP 5: Encode categorical columns ───────────────────────────────
    def encode_categoricals(self, df, fit=True):
        """
        EDA showed ProductCategory has 9 unique values dominated
        by airtime (46%). LabelEncoder converts text to integers
        so models can process them. Unknown categories seen at
        prediction time are safely mapped to -1.
        """
        logger.info("Encoding categorical columns...")

        for col in self.cat_columns:
            if col not in df.columns:
                logger.warning(
                    f"Categorical column '{col}' not found. Skipping."
                )
                continue

            try:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(
                        f"Encoded '{col}' — "
                        f"{len(le.classes_)} unique values"
                    )
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str).map(
                            lambda x: (
                                int(le.transform([x])[0])
                                if x in le.classes_ else -1
                            )
                        )
                    else:
                        logger.warning(
                            f"No fitted encoder found for '{col}'. "
                            f"Skipping."
                        )
            except Exception as e:
                logger.error(f"Encoding failed for column '{col}': {e}")
                raise

        return df

    # ── STEP 6: Scale numerical features ─────────────────────────────────
    def scale_features(self, df, fit=True):
        """
        EDA showed extreme value ranges (Amount: -1M to +9.8M UGX).
        StandardScaler (mean=0, std=1) is used for Logistic Regression
        as it assumes normally distributed inputs.
        MinMaxScaler (0 to 1) is available for tree-based models
        which are scale-invariant but benefit from normalized inputs.
        """
        scale_cols = [
            "Amount", "Value", "Amount_log",
            "transaction_count", "total_spend",
            "avg_value", "std_value", "max_value", "min_value",
            "tx_hour", "tx_day", "tx_month", "tx_dayofweek"
        ]

        # Only scale columns that actually exist in the dataframe
        scale_cols = [c for c in scale_cols if c in df.columns]

        if not scale_cols:
            logger.warning("No columns found to scale.")
            return df

        logger.info(f"Scaling {len(scale_cols)} numerical columns...")

        try:
            if fit:
                if self.scaler_type == "standard":
                    self.scaler = StandardScaler()
                else:
                    self.scaler = MinMaxScaler()
                df[scale_cols] = self.scaler.fit_transform(
                    df[scale_cols]
                )
                self._is_fitted = True
            else:
                if self.scaler is None:
                    raise ValueError(
                        "Scaler not fitted yet. "
                        "Call fit_transform before transform."
                    )
                df[scale_cols] = self.scaler.transform(df[scale_cols])

        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise

        logger.info("Scaling complete.")
        return df

    # ── Main pipeline ─────────────────────────────────────────────────────
    def fit_transform(self, df, y=None):
        """Run full pipeline and fit all transformers."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(df)}"
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        logger.info(
            f"Starting feature engineering pipeline. "
            f"Input shape: {df.shape}"
        )
        df = df.copy()
        df = self.create_aggregate_features(df)
        df = self.extract_datetime_features(df)
        df = self.handle_missing_values(df)
        df = self.log_transform(df)
        df = self.encode_categoricals(df, fit=True)
        df = self.scale_features(df, fit=True)
        logger.info(
            f"Pipeline complete. Output shape: {df.shape}"
        )
        return df

    def transform(self, df):
        """Apply fitted pipeline to new data."""
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

        df = df.copy()
        df = self.create_aggregate_features(df)
        df = self.extract_datetime_features(df)
        df = self.handle_missing_values(df)
        df = self.log_transform(df)
        df = self.encode_categoricals(df, fit=False)
        df = self.scale_features(df, fit=False)
        return df