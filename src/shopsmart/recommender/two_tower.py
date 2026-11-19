"""
Two-tower deep learning recommender (Module B, deep learning).

Two small neural networks are trained side by side:

    customer tower:  customer ID + segment                         -> 32 numbers (embedding)
    product tower:   product ID + category + type + brand + price  -> 32 numbers (embedding)

The match score is the cosine similarity of the two embeddings. Training shows the model
(customer, product) pairs the customer really engaged with (purchases count more than views)
and teaches it to rank that product above the other products in the same batch
(in-batch softmax with a log-frequency correction, so popular items are not over-penalised).

Train and evaluate (same time split and metrics as the ML recommenders):

    python -m shopsmart.recommender.two_tower

Training needs PyTorch (`pip install torch`). The saved model (models/two_tower.npz) holds only
the learned embeddings as NumPy arrays, so the Streamlit app can recommend without PyTorch.

    from shopsmart.recommender.two_tower import TwoTowerRecommender
    model = TwoTowerRecommender.load("models/two_tower.npz", suite.popularity)
    model.recommend(user_id=123, k=10)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shopsmart.recommender.models import EVENT_WEIGHTS, PopularityRecommender


# --------------------------------------------------------------------------- settings
@dataclass
class TwoTowerConfig:
    embedding_dim: int = 32        # size of the final customer / product embeddings
    id_dim: int = 32               # size of the ID embeddings inside each tower
    side_dim: int = 8              # size of segment / category / type / brand embeddings
    hidden_dim: int = 64           # hidden layer of each tower
    epochs: int = 6
    samples_per_epoch: int = 300_000
    batch_size: int = 1024
    learning_rate: float = 3e-3
    weight_decay: float = 1e-6
    temperature: float = 0.2
    logq_correction: bool = True  # subtract log item frequency from in-batch logits
    seed: int = 42

    @classmethod
    def from_config(cls, settings: dict | None) -> "TwoTowerConfig":
        settings = settings or {}
        known = {k: v for k, v in settings.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- inference (NumPy only)
class TwoTowerRecommender:
    """Recommend with embeddings learned by the two-tower network (no PyTorch needed)."""

    def __init__(self, user_ids: np.ndarray, user_vectors: np.ndarray,
                 item_vectors: np.ndarray, popularity: PopularityRecommender) -> None:
        if item_vectors.shape[0] != len(popularity.product_ids):
            raise ValueError("item embeddings do not match the product catalogue; retrain the two-tower model")
        if user_vectors.shape[0] != len(user_ids):
            raise ValueError("user embeddings and user IDs have different lengths")
        self.user_ids = np.asarray(user_ids)
        self.user_index = {u: i for i, u in enumerate(self.user_ids.tolist())}
        self.user_vectors = np.asarray(user_vectors, dtype=np.float32)
        self.item_vectors = np.asarray(item_vectors, dtype=np.float32)
        self.popularity = popularity

    def scores_for_user(self, user_id: object) -> np.ndarray | None:
        index = self.user_index.get(user_id)
        if index is None:
            return None
        return self.item_vectors @ self.user_vectors[index]

    def recommend(self, user_id: object, k: int = 10, segment: object | None = None) -> pd.DataFrame:
        """Top-k products the customer has not interacted with; cold starts use segment best-sellers."""
        scores = self.scores_for_user(user_id)
        if scores is None or not self.popularity.has_history(user_id):
            return self.popularity.recommend(user_id, k, segment)
        tie_breaker = self.popularity.scores_for_user(user_id, segment)
        return self.popularity._recommend_from_scores(user_id, scores.astype(float), k, segment, tie_breaker)

    def save(self, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(file_path, user_ids=self.user_ids, user_vectors=self.user_vectors,
                            item_ids=self.popularity.product_ids, item_vectors=self.item_vectors)
        return file_path

    @classmethod
    def load(cls, file_path: str | Path, popularity: PopularityRecommender) -> "TwoTowerRecommender":
        with np.load(file_path, allow_pickle=False) as saved:
            item_ids = saved["item_ids"]
            if not np.array_equal(item_ids, popularity.product_ids):
                raise ValueError("the two-tower model was trained on a different product catalogue; retrain it")
            return cls(saved["user_ids"], saved["user_vectors"], saved["item_vectors"], popularity)


class TwoTowerSuite:
    """Adapter so evaluate_at_k can score the two-tower model like the other recommenders."""

    def __init__(self, two_tower: TwoTowerRecommender) -> None:
        self.two_tower = two_tower
        self.popularity = two_tower.popularity

    def recommend(self, user_id: object, model: str = "two_tower", k: int = 10,
                  segment: object | None = None) -> pd.DataFrame:
        return self.two_tower.recommend(user_id, k, segment)


# --------------------------------------------------------------------------- training (PyTorch)
def _codes(values: pd.Series) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(values.fillna("unknown").astype(str), sort=True)
    return codes.astype(np.int64), len(uniques)


def train_two_tower(train_events: pd.DataFrame, products: pd.DataFrame, segments: pd.DataFrame,
                    popularity: PopularityRecommender, config: TwoTowerConfig | None = None,
                    verbose: bool = True) -> TwoTowerRecommender:
    """Train both towers on weighted (customer, product) pairs and return the learned embeddings."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Training the two-tower model needs PyTorch: pip install torch") from exc

    config = config or TwoTowerConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    # ---- product features, in the same order as the popularity model's catalogue
    catalogue = products.drop_duplicates("Product ID").set_index("Product ID").loc[popularity.product_ids]
    category, n_category = _codes(catalogue["Category"])
    ptype, n_ptype = _codes(catalogue.get("Product Type", catalogue["Category"]))
    brand, n_brand = _codes(catalogue.get("Brand", catalogue["Category"]))
    log_price = np.log1p(catalogue["Price"].astype(float).to_numpy())
    log_price = ((log_price - log_price.mean()) / (log_price.std() or 1.0)).astype(np.float32)
    n_items = len(popularity.product_ids)

    # ---- weighted (customer, product) training pairs
    events = train_events.loc[train_events["Interaction Type"].isin(EVENT_WEIGHTS)].dropna(
        subset=["User ID", "Product ID"])
    events = events.assign(weight=events["Interaction Type"].map(EVENT_WEIGHTS))
    pairs = events.groupby(["User ID", "Product ID"], sort=True)["weight"].sum().reset_index()
    user_ids = np.sort(pairs["User ID"].unique())
    user_index = {u: i for i, u in enumerate(user_ids.tolist())}
    pair_users = pairs["User ID"].map(user_index).to_numpy(np.int64)
    pair_items = pairs["Product ID"].map(popularity.product_index).to_numpy(np.int64)
    pair_weight = pairs["weight"].to_numpy(dtype=float)
    pair_prob = pair_weight / pair_weight.sum()

    # segment per customer (-1 = never bought, missing = its own "unknown" code)
    seg_map = dict(zip(segments["User ID"], segments["segment"]))
    seg_values = pd.Series([seg_map.get(u, "unknown") for u in user_ids])
    user_segment, n_segment = _codes(seg_values)

    # how often each product is sampled as a positive -> log-frequency correction
    item_freq = np.bincount(pair_items, weights=pair_prob, minlength=n_items) + 1e-9
    log_q = torch.tensor(np.log(item_freq / item_freq.sum()), dtype=torch.float32)

    class Tower(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(in_dim, config.hidden_dim), nn.ReLU(),
                                     nn.Linear(config.hidden_dim, config.embedding_dim))

        def forward(self, x):
            return nn.functional.normalize(self.mlp(x), dim=-1)

    class TwoTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_id = nn.Embedding(len(user_ids), config.id_dim)
            self.user_segment = nn.Embedding(n_segment, config.side_dim)
            self.item_id = nn.Embedding(n_items, config.id_dim)
            self.item_category = nn.Embedding(n_category, config.side_dim)
            self.item_type = nn.Embedding(n_ptype, config.side_dim)
            self.item_brand = nn.Embedding(n_brand, config.side_dim)
            self.user_tower = Tower(config.id_dim + config.side_dim)
            self.item_tower = Tower(config.id_dim + 3 * config.side_dim + 1)
            self.register_buffer("seg", torch.tensor(user_segment))
            self.register_buffer("cat", torch.tensor(category))
            self.register_buffer("typ", torch.tensor(ptype))
            self.register_buffer("brd", torch.tensor(brand))
            self.register_buffer("price", torch.tensor(log_price).unsqueeze(1))

        def users(self, u):
            return self.user_tower(torch.cat([self.user_id(u), self.user_segment(self.seg[u])], dim=1))

        def items(self, i):
            return self.item_tower(torch.cat([self.item_id(i), self.item_category(self.cat[i]),
                                              self.item_type(self.typ[i]), self.item_brand(self.brd[i]),
                                              self.price[i]], dim=1))

    model = TwoTower()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if verbose:
        print(f"Customer tower: ID + segment ({len(user_ids):,} customers) -> {config.embedding_dim} numbers")
        print(f"Product tower:  ID + category + type + brand + price ({n_items:,} products) -> "
              f"{config.embedding_dim} numbers")
        print(f"Training pairs: {len(pairs):,} (customer, product) | {config.epochs} epochs")

    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        sample = rng.choice(len(pairs), size=config.samples_per_epoch, p=pair_prob)
        total, batches = 0.0, 0
        for start in range(0, len(sample), config.batch_size):
            rows = sample[start:start + config.batch_size]
            u = torch.tensor(pair_users[rows])
            i = torch.tensor(pair_items[rows])
            logits = model.users(u) @ model.items(i).T / config.temperature
            if config.logq_correction:
                logits = logits - log_q[i].unsqueeze(0)
            same_item = (i.unsqueeze(0) == i.unsqueeze(1)) & ~torch.eye(len(rows), dtype=torch.bool)
            logits = logits.masked_fill(same_item, -1e9)  # another row's copy of the positive is not a negative
            loss = nn.functional.cross_entropy(logits, torch.arange(len(rows)))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()
            batches += 1
        if verbose:
            done = int(16 * epoch / config.epochs)
            print(f"Epoch {epoch:2d}/{config.epochs}   loss {total / batches:.3f}   "
                  f"{'█' * done}{'░' * (16 - done)}")
    if verbose:
        print(f"Training finished in {time.time() - started:.0f}s")

    model.eval()
    with torch.no_grad():
        user_vectors = model.users(torch.arange(len(user_ids))).numpy()
        item_vectors = model.items(torch.arange(n_items)).numpy()
    return TwoTowerRecommender(user_ids, user_vectors, item_vectors, popularity)


# --------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> dict:
    from shopsmart.config import CONFIG, path
    from shopsmart.data import load
    from shopsmart.recommender.evaluation import evaluate_at_k, time_based_split

    parser = argparse.ArgumentParser(description="Train and evaluate the two-tower deep learning recommender")
    parser.add_argument("--epochs", type=int, help="override the number of training epochs")
    args = parser.parse_args(argv)

    settings = CONFIG.get("two_tower", {})
    config = TwoTowerConfig.from_config(settings)
    if args.epochs:
        config.epochs = args.epochs

    interactions = load("interactions")
    products = load("products")
    segment_file = path("processed_data") / CONFIG["segmentation"]["output_file"]
    if not segment_file.exists():
        raise FileNotFoundError(f"{segment_file} not found. Run `python -m shopsmart.segmentation.train` first.")
    segments = pd.read_csv(segment_file)

    train_events, test_events, cutoff = time_based_split(interactions, CONFIG["split"]["test_fraction"])
    print(f"Loaded {len(interactions):,} events, {products['Product ID'].nunique():,} products")
    print(f"Time split cutoff: {cutoff} | train {len(train_events):,} | test {len(test_events):,}")

    popularity = PopularityRecommender(train_events, products, segments)
    two_tower = train_two_tower(train_events, products, segments, popularity, config)

    k = CONFIG["recommender"]["k"]
    print(f"Evaluating on customers who bought something new after {cutoff:%Y-%m-%d}...")
    scores = evaluate_at_k(TwoTowerSuite(two_tower), "two_tower", test_events, k)
    scores["type"] = "deep learning"
    scores["settings"] = {"embedding_dim": config.embedding_dim, "epochs": config.epochs,
                          "temperature": config.temperature, "seed": config.seed}

    model_file = path("models") / settings.get("model_file", "two_tower.npz")
    two_tower.save(model_file)

    metrics_file = path("reports") / CONFIG["recommender"]["metrics_file"]
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {"models": {}}
    if metrics.get("cutoff") and metrics["cutoff"] != cutoff.isoformat():
        print("Warning: the ML recommenders were evaluated with a different time split; retrain them too.")
    metrics.setdefault("models", {})["two_tower"] = scores
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"two_tower   HitRate@{k}: {scores[f'HitRate@{k}']:.3f} | NDCG@{k}: {scores[f'NDCG@{k}']:.3f} | "
          f"users: {scores['users_evaluated']:,}")
    hybrid = metrics["models"].get("hybrid")
    if hybrid:
        print(f"hybrid      HitRate@{k}: {hybrid[f'HitRate@{k}']:.3f} | NDCG@{k}: {hybrid[f'NDCG@{k}']:.3f}")
        verdict = "beats" if scores[f"HitRate@{k}"] > hybrid[f"HitRate@{k}"] else "does not beat"
        print(f"Result: the two-tower model {verdict} the hybrid on HitRate@{k}.")
    print(f"Saved model   -> {model_file}")
    print(f"Saved metrics -> {metrics_file}")
    return scores


if __name__ == "__main__":
    main()
