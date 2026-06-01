from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """
    The shape of data we expect when someone calls our API.
    Every field must be provided and must be the right type.
    If something is missing or wrong, FastAPI automatically
    returns a helpful error message.
    """
    Amount: float = Field(..., description="Transaction amount in UGX")
    Value: float = Field(..., description="Absolute transaction value")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Transaction channel")
    ProviderId: str = Field(..., description="Provider ID")
    PricingStrategy: int = Field(..., description="Pricing strategy code")
    transaction_count: float = Field(..., description="Customer total transactions")
    total_spend: float = Field(..., description="Customer total spend")
    avg_value: float = Field(..., description="Customer average transaction value")
    std_value: float = Field(..., description="Std deviation of customer transactions")
    max_value: float = Field(..., description="Customer max transaction value")
    min_value: float = Field(..., description="Customer min transaction value")
    tx_hour: int = Field(..., description="Hour of transaction (0-23)")
    tx_month: int = Field(..., description="Month of transaction (1-12)")
    tx_dayofweek: int = Field(..., description="Day of week (0=Monday)")

    class Config:
        json_schema_extra = {
            "example": {
                "Amount": 1000.0,
                "Value": 1000.0,
                "ProductCategory": "airtime",
                "ChannelId": "ChannelId_1",
                "ProviderId": "ProviderId_1",
                "PricingStrategy": 2,
                "transaction_count": 5.0,
                "total_spend": 5000.0,
                "avg_value": 1000.0,
                "std_value": 200.0,
                "max_value": 2000.0,
                "min_value": 500.0,
                "tx_hour": 10,
                "tx_month": 11,
                "tx_dayofweek": 2
            }
        }


class PredictionOutput(BaseModel):
    """
    The shape of our response back to the caller.
    """
    is_high_risk: int = Field(..., description="1 = high risk, 0 = low risk")
    risk_probability: float = Field(..., description="Probability of being high risk")
    risk_label: str = Field(..., description="Human readable risk label")
