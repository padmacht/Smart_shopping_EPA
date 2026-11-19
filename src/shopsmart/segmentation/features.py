"""
Customer features for segmentation (one row per customer).

Two feature sets:
  "rfm"       - the notebook 02 features: recency, frequency, monetary, sessions,
                conversion rate, average item price + share of browsing per category
  "extended"  - rfm + the behaviours notebook 02 could not see:
                q4_share          share of activity in Nov-Dec          (Seasonal Gifters)
                premium_share     share of purchases from premium brands (Premium Loyalists)
                brand_loyalty     share of purchases from the customer's favourite brand
                mobile_share      share of events on mobile
                price_index       average price paid vs the typical price of that product type
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The last two brands in each category's list are the premium tier in the generated data.
PREMIUM_BRANDS = {
    "PulseGear", "Lumio", "AuraSkin", "Petal & Co", "Stitch & Fold", "Kairo", "BrewMaster", "Ovenly",
    "LinenLuxe", "SleepWell", "Stride", "IronLeaf", "FunForge", "Wonderbox", "Nutrio", "Farm & Table",
    "Quill House", "Open Page",
}


def build_features(interactions: pd.DataFrame, products: pd.DataFrame, feature_set: str = "extended") -> pd.DataFrame:
    """Return a DataFrame indexed by User ID with one numeric column per feature."""
    if feature_set not in ("rfm", "extended"):
        raise ValueError("feature_set must be 'rfm' or 'extended'")

    prod = products.set_index("Product ID")
    ev = interactions.assign(Category=interactions["Product ID"].map(prod["Category"]))
    end = ev["Timestamp"].max()
    buys = ev[ev["Interaction Type"] == "purchase"].copy()
    buys["spend"] = buys["Price"] * buys["Quantity"].fillna(1).astype(float)

    g = ev.groupby("User ID")
    feat = pd.DataFrame({
        "recency_days": (end - g["Timestamp"].max()).dt.days,
        "sessions": g["Session ID"].nunique(),
        "conv_rate": g["Interaction Type"].apply(lambda s: (s == "purchase").mean()),
    })
    feat["frequency"] = buys.groupby("User ID").size()
    feat["monetary"] = buys.groupby("User ID")["spend"].sum()
    feat["avg_price"] = buys.groupby("User ID")["Price"].mean()
    feat = feat.join(pd.crosstab(ev["User ID"], ev["Category"], normalize="index").add_prefix("cat_"))

    if feature_set == "extended":
        feat["q4_share"] = g["Timestamp"].apply(lambda t: t.dt.month.isin([11, 12]).mean())
        feat["mobile_share"] = g["Device"].apply(lambda d: (d == "Mobile").mean())
        b = buys.assign(Brand=buys["Product ID"].map(prod["Brand"]),
                        ptype=buys["Product ID"].map(prod["Product Type"]))
        feat["premium_share"] = b.groupby("User ID")["Brand"].apply(lambda s: s.isin(PREMIUM_BRANDS).mean())
        feat["brand_loyalty"] = b.groupby("User ID")["Brand"].apply(lambda s: s.value_counts(normalize=True).iloc[0])
        typical = prod.groupby("Product Type")["Price"].median()
        b["price_ratio"] = b["Price"] / b["ptype"].map(typical)
        feat["price_index"] = b.groupby("User ID")["price_ratio"].mean()

    return feat.fillna(0).sort_index()


def transform(feat: pd.DataFrame) -> pd.DataFrame:
    """Reduce skew before scaling: counts and money are log-transformed, shares stay 0-1."""
    out = feat.copy()
    for col in ("recency_days", "sessions", "frequency", "monetary", "avg_price"):
        if col in out:
            out[col] = np.log1p(out[col])
    return out
