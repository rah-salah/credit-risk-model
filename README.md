# Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating
a credit risk model for Bati Bank's buy-now-pay-later service.

---

## Project Structure

credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ └── eda.ipynb
├── src/
│ ├── init.py
│ ├── data_processing.py
│ ├── train.py
│ ├── predict.py
│ └── api/
│ ├── main.py
│ └── pydantic_models.py
├── tests/
│ └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

---

## Credit Scoring Business Understanding

### 1. How does Basel II influence the need for an interpretable model?

Basel II is an international banking regulation that requires banks to
measure, document, and explain their risk models clearly. This means a
bank cannot just use a "black box" model that gives answers without
explanation. Regulators need to understand WHY the model flagged a
customer as risky. For example, if a customer is denied a loan, the bank
must be able to explain the decision using clear, traceable factors. This
pushes us toward interpretable models like Logistic Regression with Weight
of Evidence (WoE) encoding, where each feature's contribution to the final
score is transparent and auditable. A complex model like a neural network
may be more accurate, but if it cannot be explained to a regulator, it
cannot be used in a regulated banking environment.

### 2. Why is a proxy variable necessary, and what business risks does it introduce?

The raw dataset from the eCommerce platform contains no column indicating
whether a customer actually defaulted on a loan. This is because the
buy-now-pay-later service is new — there is no historical loan repayment
data yet. Without a target variable, we cannot train a supervised machine
learning model. A proxy variable is a substitute label we engineer from
available behavioral data. We use RFM (Recency, Frequency, Monetary)
patterns to identify customers who appear disengaged — those who shop
rarely and spend little — and label them as high risk.

However, this introduces real business risks:

- The proxy is an assumption, not ground truth. A customer who shops
  infrequently may simply be a low-frequency but reliable payer.
- The model may deny loans to good customers (false positives), causing
  the bank to lose business.
- The model may approve loans for bad customers (false negatives), causing
  financial losses.
- These risks must be clearly communicated to stakeholders and monitored
  after deployment.

### 3. What are the trade-offs between a simple vs complex model in a regulated context?

|                         | Logistic Regression (Simple)                | Gradient Boosting (Complex)                  |
| ----------------------- | ------------------------------------------- | -------------------------------------------- |
| **Interpretability**    | High — each feature has a clear coefficient | Low — hard to explain individual predictions |
| **Performance**         | Moderate                                    | High                                         |
| **Regulatory fit**      | Strong — Basel II friendly                  | Weak — requires extra explainability tools   |
| **Auditability**        | Easy to document                            | Requires SHAP or LIME to explain             |
| **Risk of overfitting** | Low                                         | Higher without careful tuning                |

In a regulated financial context, interpretability often wins over raw
performance. A slightly less accurate model that a regulator can audit
and a customer can understand is more valuable than a highly accurate
black box. In practice, we train both, compare them, and document the
trade-off clearly.

---

## Setup

```bash
pip install -r requirements.txt
```
