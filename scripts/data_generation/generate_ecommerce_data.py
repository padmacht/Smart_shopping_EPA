"""
Generate a realistic synthetic e-commerce dataset (v2: chatbot-ready).

Outputs (in --outdir):
  products.csv     Product ID, Product Name, Category, Product Type, Brand, Description, Price,
                   Colour, Size, Material, Age Range, Tags, Avg Rating, Num Reviews,
                   In Stock, Stock Qty, Delivery Days
  user_data.csv    User ID, Age, Gender, Location, Browsing History, Purchase History
  interaction.csv  User ID, Product ID, Interaction Type, Timestamp, Rating
  faq.csv          Topic, Question, Answer   (shop policies for the chatbot)

Usage:
  python generate_ecommerce_data.py --users 2000 --products 1000 --interactions 15000 --seed 42 --outdir data
"""
import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from catalogue import CATEGORIES, FAQ

LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco", "Seattle", "Boston", "Miami",
             "London", "Manchester", "Birmingham", "Edinburgh", "Dublin", "Toronto", "Sydney", "Melbourne"]
GENDERS, GENDER_W = ["Female", "Male", "Other"], [0.48, 0.48, 0.04]
INTERACTION_TYPES = ["view", "click", "add_to_cart", "purchase", "rating", "wishlist"]
INTERACTION_W = [0.35, 0.20, 0.12, 0.15, 0.10, 0.08]

# Which categories each age group leans towards (adds learnable signal)
AGE_AFFINITY = {
    (18, 25): {"Electronics": 3, "Fashion": 3, "Beauty": 2, "Sports": 2, "Books": 1},
    (26, 40): {"Electronics": 2, "Home & Kitchen": 3, "Fashion": 2, "Sports": 2, "Toys": 3, "Food": 1},
    (41, 55): {"Home & Kitchen": 3, "Bedding & Linens": 3, "Books": 2, "Food": 2, "Toys": 2},
    (56, 90): {"Books": 3, "Bedding & Linens": 2, "Food": 3, "Home & Kitchen": 2, "Beauty": 1},
}
# Brand tier shifts price within the item's range (premium brands cost more)
BRAND_TIER = [0.75, 0.9, 1.0, 1.1, 1.25]

OPENERS = [
    "The {name} {is_are} designed for {use1} and {use2}.",
    "Meet the {name}, {a_adj} {adj_l} {item_l} made for {use1}.",
    "Looking for {a_dep} {item_l}? The {name} {is_are} a great choice for {use1} and {use2}.",
    "Make {use1} easier with the {name} from {brand}.",
    "{brand}'s {adj_l} {item_l} {bring} quality and value to {use1}.",
]
FEATURE_LINES = [
    "It features {f1}, {f2} and {f3}.",
    "Highlights include {f1}, {f2}, and {f3}.",
    "You get {f1} and {f2}, along with {f3}.",
    "Key features: {f1}, {f2} and {f3}.",
]
EXTRA_LINES = ["Plus: {f4}.", "Another highlight is {f4}.", "It also stands out for {f4}."]
CLOSERS = [
    "Backed by {brand}'s 30-day satisfaction promise.",
    "A smart choice whether you're treating yourself or buying for someone else.",
    "Built to last and easy to love from day one.",
    "Great value for everyday use.",
    "Designed and tested by the {brand} team.",
]
PLURAL_ITEMS = {"Wireless Earbuds", "Noise-Cancelling Headphones", "Running Sneakers", "Chino Trousers",
                "Ankle Boots", "Glass Food Containers", "Blackout Curtains", "Adjustable Dumbbells",
                "Mixed Nuts", "Arabica Coffee Beans", "Protein Bars"}
KEEP_CASE = {"ABS", "EPS", "EVA", "TPE", "PU", "German", "Turkish", "Egyptian", "Greece"}


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def soft_lower(text):
    """Lower-case words except acronyms and proper nouns (ABS plastic, German steel)."""
    return " ".join(w if w in KEEP_CASE else w.lower() for w in text.split())


def pick(rng, seq):
    return rng.choice(seq) if seq else ""


def describe(rng, name, brand, adj, item, cat, feats, material, colour, size, age_range, uses, assure):
    u1, u2 = rng.sample(uses, 2)
    s = [rng.choice(OPENERS).format(name=name, brand=brand, a_adj=article(adj),
                                    is_are="are" if item in PLURAL_ITEMS else "is",
                                    bring="bring" if item in PLURAL_ITEMS else "brings",
                                    a_dep="dependable" if item in PLURAL_ITEMS else "a dependable", adj_l=adj.lower(),
                                    item_l=soft_lower(item), use1=u1, use2=u2),
         rng.choice(FEATURE_LINES).format(f1=feats[0], f2=feats[1], f3=feats[2]),
         rng.choice(EXTRA_LINES).format(f4=feats[3])]
    detail = []
    if material:
        detail.append(f"Made from {soft_lower(material)}")
    if colour and colour != "Multicolour":
        detail.append(f"in {colour.lower()}" if detail else f"Shown in {colour.lower()}")
    if detail:
        s.append(" ".join(detail) + ".")
    if size and size != "One size":
        s.append(f"Size: {size}.")
    if age_range not in ("Adult", "All ages"):
        s.append(f"Recommended for ages {age_range}.")
    s.append(rng.choice(assure))
    s.append(rng.choice(CLOSERS).format(brand=brand))
    return " ".join(s)


def make_products(n, rng):
    # every category/item type gets roughly the same number of products
    types = [(cat, it) for cat, c in CATEGORIES.items() for it in c["items"]]
    rows, names = [], set()
    pid, attempts = 0, 0
    while pid < n and attempts < n * 20:
        attempts += 1
        cat, (item, (lo, hi), features, over) = types[pid % len(types)] if pid < len(types) else rng.choice(types)
        c = CATEGORIES[cat]
        bi = rng.randrange(len(c["brands"]))
        brand, adj = c["brands"][bi], rng.choice(c["adj"])
        colour = pick(rng, over.get("colours", c["colours"]))
        size = pick(rng, over.get("size", c.get("size", [""])))
        material = pick(rng, over.get("materials", c["materials"]))
        age_range = over.get("age_range", c["age_range"])
        name = f"{brand} {adj} {item}"
        if name in names:
            variant = colour if colour and colour not in name else size
            name = f"{name} - {variant}" if variant else name
        if name in names:
            continue
        names.add(name)
        feats = rng.sample(features, 4)
        uses = over.get("uses", c["uses"])
        price = round(min(hi, max(lo, rng.uniform(lo, hi) * BRAND_TIER[bi])) - 0.01, 2)
        tags = sorted(set(c["tags"] + over.get("tags", [])))
        rows.append({
            "Product ID": pid, "Product Name": name, "Category": cat, "Product Type": item, "Brand": brand,
            "Description": describe(rng, name, brand, adj, item, cat, feats, material, colour, size, age_range, uses, c["assure"]),
            "Price": price, "Colour": colour, "Size": size, "Material": material, "Age Range": age_range,
            "Tags": ", ".join(tags),
            "In Stock": rng.random() > 0.08,
            "Delivery Days": rng.randint(*c["delivery"]),
        })
        pid += 1
    df = pd.DataFrame(rows)
    df["Stock Qty"] = [rng.randint(5, 250) if s else 0 for s in df["In Stock"]]
    return df


def age_aff(age):
    for (lo, hi), aff in AGE_AFFINITY.items():
        if lo <= age <= hi:
            return aff
    return {}


def make_users(n, rng):
    cats = list(CATEGORIES)
    rows = []
    for uid in range(n):
        age = int(np.clip(rng.gauss(37, 13), 18, 80))
        w = {c: 1 for c in cats}
        for c, extra in age_aff(age).items():
            w[c] += extra
        prefs, pool = [], dict(w)
        for _ in range(rng.randint(2, 5)):
            c = rng.choices(list(pool), weights=list(pool.values()))[0]
            prefs.append(c)
            pool.pop(c)
        rows.append({"User ID": uid, "Age": age, "Gender": rng.choices(GENDERS, GENDER_W)[0],
                     "Location": rng.choice(LOCATIONS), "_prefs": prefs})
    return pd.DataFrame(rows)


def make_interactions(n, users, products, rng, start, end):
    by_cat = products.groupby("Category")["Product ID"].apply(list).to_dict()
    popularity = {p: rng.paretovariate(1.5) for p in products["Product ID"]}
    pop_cat = {c: [popularity[p] for p in ids] for c, ids in by_cat.items()}
    quality = {p: rng.uniform(2.8, 4.8) for p in products["Product ID"]}
    activity = [rng.lognormvariate(0, 0.7) for _ in range(len(users))]
    prefs = users.set_index("User ID")["_prefs"].to_dict()
    pid_cat = products.set_index("Product ID")["Category"].to_dict()
    all_p, all_w = products["Product ID"].tolist(), [popularity[p] for p in products["Product ID"]]
    span = int((end - start).total_seconds())

    uids = users["User ID"].tolist()
    base = uids * 3 if n >= 3 * len(uids) else []
    seq = base + rng.choices(uids, weights=activity, k=n - len(base))
    rng.shuffle(seq)
    rows = []
    for uid in seq:
        if rng.random() < 0.8:
            cat = rng.choice(prefs[uid])
            pid = rng.choices(by_cat[cat], weights=pop_cat[cat])[0]
        else:
            pid = rng.choices(all_p, weights=all_w)[0]
        bonus = 0.4 if pid_cat[pid] in prefs[uid] else -0.3
        rows.append({"User ID": uid, "Product ID": pid,
                     "Interaction Type": rng.choices(INTERACTION_TYPES, INTERACTION_W)[0],
                     "Timestamp": (start + timedelta(seconds=rng.randint(0, span))).strftime("%Y-%m-%d %H:%M:%S"),
                     "Rating": round(float(np.clip(rng.gauss(quality[pid] + bonus, 0.6), 1, 5)), 1)})
    df = pd.DataFrame(rows)
    buyers = set(df.loc[df["Interaction Type"] == "purchase", "User ID"])
    no_buy = df[~df["User ID"].isin(buyers)]
    if len(no_buy):
        df.loc[no_buy.groupby("User ID")["Rating"].idxmax(), "Interaction Type"] = "purchase"
    return df.sort_values("Timestamp").reset_index(drop=True)


def add_review_stats(products, inter):
    rev = inter[inter["Interaction Type"].isin(["rating", "purchase"])]
    stats = rev.groupby("Product ID")["Rating"].agg(["mean", "size"])
    products = products.copy()
    products["Avg Rating"] = products["Product ID"].map(stats["mean"]).round(1)
    products["Num Reviews"] = products["Product ID"].map(stats["size"]).fillna(0).astype(int)
    cols = ["Product ID", "Product Name", "Category", "Product Type", "Brand", "Description", "Price", "Colour",
            "Size", "Material", "Age Range", "Tags", "Avg Rating", "Num Reviews", "In Stock", "Stock Qty",
            "Delivery Days"]
    return products[cols]


def finalise_users(users, inter, products):
    pid_cat = products.set_index("Product ID")["Category"].to_dict()
    browse, buys = {}, {}
    for uid, pid, itype in inter[["User ID", "Product ID", "Interaction Type"]].itertuples(index=False):
        if itype in {"view", "click", "wishlist", "add_to_cart"}:
            browse.setdefault(uid, []).append(pid_cat[pid])
        if itype == "purchase":
            buys.setdefault(uid, []).append(pid)
    uniq = lambda s: list(dict.fromkeys(s))
    users = users.copy()
    users["Browsing History"] = [", ".join(uniq(browse.get(u) or p)) for u, p in zip(users["User ID"], users["_prefs"])]
    users["Purchase History"] = [", ".join(map(str, uniq(buys.get(u, [])))) for u in users["User ID"]]
    return users.drop(columns="_prefs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--products", type=int, default=1000)
    ap.add_argument("--interactions", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    np.random.seed(a.seed)

    products = make_products(a.products, rng)
    users = make_users(a.users, rng)
    inter = make_interactions(a.interactions, users, products, rng, datetime(2024, 11, 1), datetime(2026, 10, 31, 23, 59, 59))
    products = add_review_stats(products, inter)
    users = finalise_users(users, inter, products)

    products.to_csv(f"{a.outdir}/products.csv", index=False)
    users.to_csv(f"{a.outdir}/user_data.csv", index=False)
    inter.to_csv(f"{a.outdir}/interaction.csv", index=False)
    pd.DataFrame(FAQ, columns=["Topic", "Question", "Answer"]).to_csv(f"{a.outdir}/faq.csv", index=False)
    print(f"products.csv {len(products):,} | user_data.csv {len(users):,} | "
          f"interaction.csv {len(inter):,} | faq.csv {len(FAQ)}")


if __name__ == "__main__":
    main()
