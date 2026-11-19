"""
K-Means customer segmentation model.

    from shopsmart.segmentation.model import SegmentationModel
    model = SegmentationModel(n_clusters=8).fit(features)
    labels = model.predict(features)          # -1 = non-buyer, 0..k-1 = segment
    model.save("models/segmentation_kmeans.joblib")

The scaler and K-Means are saved together, so a new customer is scaled exactly like
the training data (no train/serve mismatch).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shopsmart.segmentation.features import transform

NON_BUYER = -1


class SegmentationModel:
    def __init__(self, n_clusters: int = 8, buyers_only: bool = True, random_state: int = 42):
        self.n_clusters = n_clusters
        self.buyers_only = buyers_only
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)),
        ])
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------ training
    def _buyer_mask(self, feat: pd.DataFrame) -> pd.Series:
        return feat["frequency"] > 0 if self.buyers_only else pd.Series(True, index=feat.index)

    def fit(self, feat: pd.DataFrame) -> "SegmentationModel":
        self.feature_names = list(feat.columns)
        X = transform(feat[self._buyer_mask(feat)])
        self.pipeline.fit(X)
        return self

    def predict(self, feat: pd.DataFrame) -> pd.Series:
        feat = feat[self.feature_names]
        labels = pd.Series(NON_BUYER, index=feat.index, name="segment")
        mask = self._buyer_mask(feat)
        if mask.any():
            labels[mask] = self.pipeline.predict(transform(feat[mask]))
        return labels

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, feat: pd.DataFrame, labels: pd.Series) -> dict:
        """Internal quality metrics on the clustered (buyer) customers."""
        mask = labels != NON_BUYER
        X = self.pipeline.named_steps["scale"].transform(transform(feat.loc[mask, self.feature_names]))
        y = labels[mask].to_numpy()
        return {
            "silhouette": float(silhouette_score(X, y, sample_size=min(3000, len(y)), random_state=self.random_state)),
            "davies_bouldin": float(davies_bouldin_score(X, y)),         # lower = better
            "calinski_harabasz": float(calinski_harabasz_score(X, y)),   # higher = better
            "inertia": float(self.pipeline.named_steps["kmeans"].inertia_),
            "n_clustered": int(mask.sum()),
            "n_non_buyers": int((~mask).sum()),
        }

    @staticmethod
    def search_k(feat: pd.DataFrame, k_min: int = 2, k_max: int = 12, buyers_only: bool = True,
                 random_state: int = 42) -> pd.DataFrame:
        """Elbow + silhouette table to help choose the number of segments."""
        rows = []
        for k in range(k_min, k_max + 1):
            m = SegmentationModel(k, buyers_only, random_state).fit(feat)
            rows.append({"k": k, **m.evaluate(feat, m.predict(feat))})
        return pd.DataFrame(rows).set_index("k")

    # ------------------------------------------------------------------ profiles
    @staticmethod
    def profile(feat: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Median behaviour per segment + its strongest category (what marketing reads)."""
        df = feat.assign(segment=labels)
        prof = df.groupby("segment").agg(
            customers=("frequency", "size"),
            recency_days=("recency_days", "median"),
            purchases=("frequency", "median"),
            spend=("monetary", "median"),
            avg_price=("avg_price", "median"),
        )
        cats = df.filter(like="cat_").groupby(df["segment"]).mean()
        prof["top_category"] = cats.idxmax(axis=1).str.replace("cat_", "", regex=False)
        for col in ("q4_share", "premium_share", "mobile_share"):
            if col in df:
                prof[col] = df.groupby("segment")[col].mean().round(2)
        return prof.round(1)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "SegmentationModel":
        return joblib.load(path)
