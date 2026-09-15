"""Risk Indicator Score — explainable review priority. NOT a fraud score.

Why it exists. The competition PDF lists "Risk Score" among the example core
KPIs (Gate 3) and its Day 2 guide asks for "the exact mathematical formula for
... 'Fraud Risk' in your dataset". The dataset contains no fraud label, so a
supervised or probabilistic fraud score cannot be built, and no score can be
validated as a fraud predictor. What can be built honestly is a transparent
points total over observed conditions that justify a manual review.

    Risk Indicator ≠ Confirmed Fraud

Customer score, 0–100 = four components of up to 25 points each

    dispute_history   = 25 × min(disputed_transactions, 2) / 2
    disputed_value    = 25 × percentile rank of disputed_amount among customers
                        with any disputed value (0 if none)
    kyc_status        = 25 REJECTED · 12.5 PENDING · 0 VERIFIED
                        · 0 and flagged "not assessable" when no KYC record exists
    identity          = 25 when the customer ID is shared by different people

Merchant score, 0–100

    dispute_volume    = 25 × min(disputed_transactions, 3) / 3
    disputed_value    = 25 × percentile rank of disputed_amount among merchants
                        with any disputed value (0 if none)
    merchant_status   = 25 BLOCKED or SUSPENDED · 12.5 INACTIVE · 0 ACTIVE
                        · 0 and flagged when there is no master record
    identity          = 25 when the merchant ID is shared by different businesses

Bands: 0–24 Low · 25–49 Moderate · 50–74 Elevated · 75–100 High review priority.

Design decisions, each tied to evidence in this project:

  * **Equal weights.** There is no label to learn weights from. Unequal weights
    would imply one indicator predicts fraud better than another — a claim this
    data cannot support.
  * **Dispute volume, not dispute rate.** The overdispersion test found no
    merchant-level dispute propensity, so a rate component would score sampling
    noise. Volume and value are real review workload.
  * **UNKNOWN KYC and missing master records score zero.** Absence of a record is
    a coverage gap, not evidence. Scoring it would flag most of the population.
  * **Missing UTR and reporting delay are excluded.** The hypothesis register shows
    neither is associated with disputes on this data.
  * **KYC and merchant status are included as compliance conditions.** A rejected
    customer or a suspended merchant that is still transacting warrants review
    whether or not it predicts disputes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import UNKNOWN_KEY

POINTS = 25.0
SCORE_MAX = 4 * POINTS
CUSTOMER_DISPUTE_CAP = 2
MERCHANT_DISPUTE_CAP = 3

BANDS = ((75.0, "High"), (50.0, "Elevated"), (25.0, "Moderate"), (0.0, "Low"))
BAND_ORDER = ("High", "Elevated", "Moderate", "Low")

KYC_POINTS = {"REJECTED": POINTS, "PENDING": POINTS / 2, "VERIFIED": 0.0}
MERCHANT_STATUS_POINTS = {"BLOCKED": POINTS, "SUSPENDED": POINTS,
                          "INACTIVE": POINTS / 2, "ACTIVE": 0.0}

CUSTOMER_COMPONENTS = {
    "ri_dispute_history": "Dispute history",
    "ri_disputed_value": "Disputed value",
    "ri_kyc_status": "KYC status",
    "ri_identity": "Identity ambiguity",
}
MERCHANT_COMPONENTS = {
    "ri_dispute_volume": "Dispute volume",
    "ri_disputed_value": "Disputed value",
    "ri_merchant_status": "Merchant status",
    "ri_identity": "Identity ambiguity",
}

DISCLAIMER = ("Risk Indicator Score is a review-priority points total built from observed "
              "conditions. It is not a fraud probability and does not indicate confirmed "
              "fraud — the data contains no fraud label.")


def band(score: float) -> str:
    for floor, name in BANDS:
        if score >= floor:
            return name
    return "Low"


def value_points(amounts: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Percentile-rank points for disputed value, and the percentile itself."""
    a = pd.to_numeric(amounts, errors="coerce").fillna(0.0)
    positive = a > 0
    pct = pd.Series(0.0, index=a.index)
    if positive.any():
        pct[positive] = a[positive].rank(method="max", pct=True)
    return (POINTS * pct).round(4), (100.0 * pct).round(1)


def _reasons(row: pd.Series, components: dict, describe) -> str:
    present = [(row[c], describe(c, row)) for c in components if row[c] > 0]
    if not present:
        return "No indicators present"
    present.sort(key=lambda p: -p[0])
    return "; ".join(f"{text} (+{points:.1f})" for points, text in present)


def _finalise(out: pd.DataFrame, components: dict, describe) -> pd.DataFrame:
    out["risk_indicator_score"] = out[list(components)].sum(axis=1).round(2)
    out["risk_indicator_band"] = out["risk_indicator_score"].map(band)
    out["risk_indicator_reasons"] = out.apply(lambda r: _reasons(r, components, describe),
                                              axis=1)
    validate(out, components)
    return out


def score_customers(agg_user: pd.DataFrame) -> pd.DataFrame:
    out = agg_user.copy()
    disputed = pd.to_numeric(out["disputed_transactions"], errors="coerce").fillna(0)
    out["ri_dispute_history"] = (POINTS * np.minimum(disputed, CUSTOMER_DISPUTE_CAP)
                                 / CUSTOMER_DISPUTE_CAP).round(4)
    out["ri_disputed_value"], out["disputed_value_percentile"] = value_points(
        out["disputed_amount"])
    kyc = out["kyc_status_canonical"].fillna(UNKNOWN_KEY).astype(str)
    out["ri_kyc_status"] = kyc.map(KYC_POINTS).fillna(0.0)
    out["kyc_assessable"] = kyc.isin(list(KYC_POINTS))
    out["ri_identity"] = np.where(out["identity_ambiguous"].fillna(False).astype(bool),
                                  POINTS, 0.0)

    def describe(component, row):
        if component == "ri_dispute_history":
            return f"{int(row['disputed_transactions'])} disputed transaction(s)"
        if component == "ri_disputed_value":
            return (f"disputed ₹{row['disputed_amount']:,.0f}, "
                    f"{row['disputed_value_percentile']:.0f}th percentile")
        if component == "ri_kyc_status":
            return f"KYC {row['kyc_status_canonical']}"
        return "customer ID shared by different people"

    return _finalise(out, CUSTOMER_COMPONENTS, describe)


def score_merchants(agg_merchant: pd.DataFrame) -> pd.DataFrame:
    out = agg_merchant.copy()
    disputed = pd.to_numeric(out["disputed_transactions"], errors="coerce").fillna(0)
    out["ri_dispute_volume"] = (POINTS * np.minimum(disputed, MERCHANT_DISPUTE_CAP)
                                / MERCHANT_DISPUTE_CAP).round(4)
    out["ri_disputed_value"], out["disputed_value_percentile"] = value_points(
        out["disputed_amount"])
    status = out["merchant_status_canonical"].fillna(UNKNOWN_KEY).astype(str)
    out["ri_merchant_status"] = status.map(MERCHANT_STATUS_POINTS).fillna(0.0)
    out["status_assessable"] = status.isin(list(MERCHANT_STATUS_POINTS))
    out["ri_identity"] = np.where(out["identity_ambiguous"].fillna(False).astype(bool),
                                  POINTS, 0.0)

    def describe(component, row):
        if component == "ri_dispute_volume":
            return f"{int(row['disputed_transactions'])} disputed transaction(s)"
        if component == "ri_disputed_value":
            return (f"disputed ₹{row['disputed_amount']:,.0f}, "
                    f"{row['disputed_value_percentile']:.0f}th percentile")
        if component == "ri_merchant_status":
            return f"merchant {row['merchant_status_canonical']} while transacting"
        return "merchant ID shared by different businesses"

    return _finalise(out, MERCHANT_COMPONENTS, describe)


def validate(scored: pd.DataFrame, components: dict) -> None:
    """Raise if any score breaks its own definition."""
    parts = scored[list(components)]
    if ((parts < 0) | (parts > POINTS + 1e-9)).any().any():
        raise AssertionError("a risk indicator component is outside 0-25")
    total = parts.sum(axis=1).round(2)
    if not np.allclose(total, scored["risk_indicator_score"], atol=0.011):
        raise AssertionError("risk indicator components do not sum to the score")
    s = scored["risk_indicator_score"]
    if ((s < 0) | (s > SCORE_MAX)).any():
        raise AssertionError("risk indicator score outside 0-100")


def band_summary(scored: pd.DataFrame, entity: str) -> pd.DataFrame:
    n = len(scored)
    rows = []
    for name in BAND_ORDER:
        sub = scored[scored["risk_indicator_band"] == name]
        rows.append({
            "entity": entity, "band": name, "entities": len(sub),
            "share_pct": round(100.0 * len(sub) / n, 2) if n else 0.0,
            "min_score": float(sub["risk_indicator_score"].min()) if len(sub) else np.nan,
            "max_score": float(sub["risk_indicator_score"].max()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def component_summary(scored: pd.DataFrame, entity: str, components: dict) -> pd.DataFrame:
    n = len(scored)
    rows = []
    for column, label in components.items():
        present = scored[column] > 0
        rows.append({
            "entity": entity, "component": column, "label": label,
            "entities_with_points": int(present.sum()),
            "share_pct": round(100.0 * float(present.mean()), 2) if n else 0.0,
            "mean_points_when_present": round(float(scored.loc[present, column].mean()), 2)
            if present.any() else 0.0,
        })
    return pd.DataFrame(rows)
