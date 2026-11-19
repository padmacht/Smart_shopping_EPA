"""
ShopSmart v3 dataset generator - large and deep-learning ready.

Hidden "behaviour engine" (what the models should discover):
  * 8 customer personas with different category tastes, price sensitivity, brand loyalty,
    activity, device and churn                      -> segmentation, autoencoders
  * latent taste vectors for every customer and product (dot product = affinity)
                                                      -> matrix factorisation, NCF, two-tower
  * shopping sessions with a view -> click -> cart -> purchase funnel
                                                      -> session / funnel features
  * complementary next purchases (yoga mat -> foam roller)  -> sequence models (GRU4Rec, SASRec)
  * seasonality (gifts in Nov-Dec, summer items in Jun-Aug) and churn
  * written reviews whose sentiment follows the rating -> sentiment models (LSTM, BERT)
  * labelled search queries + chatbot utterances     -> semantic search, intent classifiers

Files written to --outdir:
  products.csv, user_data.csv, interaction.csv, reviews.csv, search_queries.csv,
  chatbot_intents.csv, faq.csv, customer_personas_truth.csv (hidden labels - evaluation only!)

Usage:
  python generate_dl_dataset.py --users 5000 --products 1500 --seed 42 --outdir data_v3
"""
import argparse
import json
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from catalogue import CATEGORIES, FAQ
from dl_content import (COMPLEMENTS, INTENT_SLOTS, INTENT_TEMPLATES, PERSONAS, REVIEW_CLOSE, REVIEW_CRITIC, REVIEW_MILD,
                        REVIEW_FILLER, REVIEW_HEDGE,
                        REVIEW_OPEN, REVIEW_PRAISE, REVIEW_TITLES, SEMANTIC)
from generate_ecommerce_data import BRAND_TIER, PLURAL_ITEMS, describe, pick

START, END = datetime(2024, 11, 1), datetime(2026, 10, 31, 23, 59)
LOCATIONS = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Edinburgh", "Bristol", "Cardiff", "Belfast",
             "Dublin", "New York", "Chicago", "San Francisco", "Toronto", "Sydney", "Melbourne"]
DIM = 8                      # size of the hidden taste vectors
SEASONAL_TAGS = {"gift": {11: 1.8, 12: 2.5}, "summer": {6: 2.0, 7: 2.2, 8: 1.8}, "winter": {11: 1.6, 12: 1.8, 1: 1.6, 2: 1.3}}
MONTH_TRAFFIC = {1: .85, 2: .85, 3: .95, 4: .95, 5: 1, 6: 1, 7: 1, 8: 1, 9: .95, 10: 1, 11: 1.35, 12: 1.45}


# =========================================================================== products
def make_products(n, rng, nrng):
    types = [(cat, it) for cat, c in CATEGORIES.items() for it in c["items"]]
    type_vec = {it[0]: nrng.normal(0, 1, DIM) for _, it in types}
    # products of the same category share part of their vector -> category structure in embeddings
    cat_vec = {cat: nrng.normal(0, 1, DIM) for cat in CATEGORIES}
    rows, names, hidden = [], set(), {}
    pid, tries = 0, 0
    while pid < n and tries < n * 30:
        tries += 1
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
        price = round(min(hi, max(lo, rng.uniform(lo, hi) * BRAND_TIER[bi])) - 0.01, 2)
        uses = over.get("uses", c["uses"])
        rows.append({
            "Product ID": pid, "Product Name": name, "Category": cat, "Product Type": item, "Brand": brand,
            "Description": describe(rng, name, brand, adj, item, cat, feats, material, colour, size, age_range, uses, c["assure"]),
            "Price": price, "Colour": colour, "Size": size, "Material": material, "Age Range": age_range,
            "Tags": ", ".join(sorted(set(c["tags"] + over.get("tags", [])))),
            "In Stock": rng.random() > 0.08, "Delivery Days": rng.randint(*c["delivery"]),
            "Launch Date": (START + timedelta(days=rng.randint(-365, 600))).date().isoformat(),
        })
        hidden[pid] = {
            "vec": 0.6 * type_vec[item] + 0.4 * cat_vec[cat] + nrng.normal(0, 0.35, DIM),
            "quality": rng.uniform(-1.0, 0.8) + (0.25 if bi >= 3 else 0),   # premium brands slightly better
            "pop": math.log(rng.paretovariate(1.4)),
            "feats": feats, "uses": uses, "tier": bi,
        }
        pid += 1
    df = pd.DataFrame(rows)
    df["Stock Qty"] = [rng.randint(5, 300) if s else 0 for s in df["In Stock"]]
    return df, hidden


# =========================================================================== users
def make_users(n, rng, nrng):
    names = list(PERSONAS)
    shares = [PERSONAS[p]["share"] for p in names]
    persona_vec = {p: nrng.normal(0, 1, DIM) for p in names}
    rows, hidden = [], {}
    for uid in range(n):
        p = rng.choices(names, shares)[0]
        P = PERSONAS[p]
        lo, hi = P["age"]
        signup = START - timedelta(days=rng.randint(0, 900)) if rng.random() < 0.55 else START + timedelta(days=rng.randint(0, 650))
        churn_date = None
        if rng.random() < P["churn"]:
            churn_date = max(signup, START) + timedelta(days=rng.randint(30, 500))
            if churn_date > END:
                churn_date = None
        cats = list(CATEGORIES)
        w = np.array([P["cat_w"].get(c, 0.25) for c in cats], float)
        w = w * nrng.dirichlet(np.ones(len(cats)) * 2) * len(cats)       # personal variation
        rows.append({
            "User ID": uid, "Age": rng.randint(lo, hi), "Gender": rng.choices(["Female", "Male", "Other"], [.5, .47, .03])[0],
            "Location": rng.choice(LOCATIONS), "Signup Date": signup.date().isoformat(),
            "Preferred Device": rng.choices(list(P["device"]), list(P["device"].values()))[0],
            "Newsletter": rng.random() < (0.6 if p in ("Bargain Hunter", "Premium Loyalist") else 0.35),
        })
        hidden[uid] = {"persona": p, "vec": 0.7 * persona_vec[p] + nrng.normal(0, 0.6, DIM),
                       "cat_w": dict(zip(cats, w / w.sum())), "churn_date": churn_date,
                       "fav_brands": set(), "signup": signup}
    users = pd.DataFrame(rows)
    # Busy parents skew female/mixed, techies skew male: mild realistic correlation
    return users, hidden


# =========================================================================== behaviour
def seasonal_boost(tags, month):
    b = 0.0
    for t, months in SEASONAL_TAGS.items():
        if t in tags:
            b += math.log(months.get(month, 1.0))
    return b


def sample_session_time(rng, a, b, gift_only=False):
    for _ in range(50):
        t = a + timedelta(seconds=rng.randint(0, max(1, int((b - a).total_seconds()))))
        if gift_only and t.month not in (11, 12):
            continue
        if rng.random() < MONTH_TRAFFIC[t.month] / 1.45:
            hour = rng.choices(range(24), [1, .5, .3, .2, .2, .3, .8, 1.5, 2, 2.2, 2.2, 2.4, 3, 2.6, 2.3, 2.3, 2.5,
                                          3, 3.4, 4.2, 4.6, 4.2, 3, 1.8])[0]
            return t.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59))
    return None


def simulate(users, uh, products, ph, rng, nrng):
    pids = products["Product ID"].to_numpy()
    by_cat = {c: g["Product ID"].to_numpy() for c, g in products.groupby("Category")}
    by_type = {t: g["Product ID"].to_numpy() for t, g in products.groupby("Product Type")}
    ptype = products.set_index("Product ID")["Product Type"].to_dict()
    pcat = products.set_index("Product ID")["Category"].to_dict()
    pbrand = products.set_index("Product ID")["Brand"].to_dict()
    price = products.set_index("Product ID")["Price"].to_dict()
    tags = products.set_index("Product ID")["Tags"].to_dict()
    type_median = products.groupby("Product Type")["Price"].median().to_dict()
    launch = pd.to_datetime(products.set_index("Product ID")["Launch Date"]).to_dict()
    V = np.stack([ph[p]["vec"] for p in pids])
    qual = np.array([ph[p]["quality"] for p in pids])
    pop = np.array([ph[p]["pop"] for p in pids])
    logprice = np.array([math.log(price[p] / type_median[ptype[p]]) for p in pids])
    tier = np.array([ph[p]["tier"] for p in pids])
    devices = ["Mobile", "Desktop", "Tablet"]

    events, sid = [], 0
    for uid in users["User ID"]:
        h, P = uh[uid], PERSONAS[uh[uid]["persona"]]
        a = max(h["signup"], START)
        b = h["churn_date"] or END
        months = max((b - a).days / 30.4, 0.3)
        n_sessions = nrng.poisson(P["rate"] * months * (0.25 if P.get("gift_only") else 1)) + 1
        times = sorted(t for t in (sample_session_time(rng, a, b, P.get("gift_only")) for _ in range(n_sessions)) if t)
        if P["loyalty"] > 0:
            # favourite brands = brands of products the user has highest affinity with
            h["fav_brands"] = {pbrand[p] for p in pids[np.argsort(-(V @ h["vec"]))[:3]]}
        last_purchase = None
        for t in times:
            sid += 1
            dev = h.get("dev") or rng.choices(devices, [PERSONAS[h["persona"]]["device"].get(d, 0) for d in devices])[0]
            # --- session intent: complement of last purchase, gift shopping, or a favourite category
            if last_purchase and (t - last_purchase[1]).days < 75 and ptype[last_purchase[0]] in COMPLEMENTS and rng.random() < 0.6:
                cand = np.concatenate([by_type[x] for x in COMPLEMENTS[ptype[last_purchase[0]]] if x in by_type])
                comp = 0.8          # customers completing a set are keen to buy
            else:
                comp = 0.0
                cat = rng.choices(list(h["cat_w"]), list(h["cat_w"].values()))[0]
                cand = by_cat[cat]
            cand = np.array([p for p in cand if launch[p] <= t]) if len(cand) else cand
            if len(cand) == 0:
                continue
            idx = cand  # product ids equal row positions
            score = (1.1 * (V[idx] @ h["vec"]) / math.sqrt(DIM) + 0.6 * pop[idx]
                     - P["price_sens"] * logprice[idx]
                     + np.array([seasonal_boost(tags[p], t.month) for p in idx]))
            if P["loyalty"]:
                score += P["loyalty"] * np.array([pbrand[p] in h["fav_brands"] for p in idx])
            if P["price_sens"] < 0:
                score += 0.3 * (tier[idx] >= 3)
            prob = np.exp(score - score.max()); prob /= prob.sum()
            n_view = min(len(idx), 1 + nrng.geometric(0.3))
            viewed = nrng.choice(idx, size=n_view, replace=False, p=prob)
            clock = t
            for p in viewed:
                s = float(score[np.where(idx == p)[0][0]])
                clock += timedelta(seconds=rng.randint(15, 240))
                events.append((uid, p, sid, "view", clock, None, dev, None))
                aff = 1 / (1 + math.exp(-(s - 0.4 + P["buy_bias"] + comp)))
                if rng.random() < 0.22 + 0.4 * aff:
                    clock += timedelta(seconds=rng.randint(5, 90))
                    events.append((uid, p, sid, "click", clock, None, dev, None))
                    if rng.random() < 0.06:
                        events.append((uid, p, sid, "wishlist", clock + timedelta(seconds=20), None, dev, None))
                    if rng.random() < 0.10 + 0.45 * aff:
                        clock += timedelta(seconds=rng.randint(10, 200))
                        events.append((uid, p, sid, "add_to_cart", clock, None, dev, None))
                        if rng.random() < 0.30 + 0.35 * aff:
                            clock += timedelta(seconds=rng.randint(30, 400))
                            qty = rng.choice([1, 1, 1, 2, 3]) if pcat[p] == "Food" else 1
                            events.append((uid, p, sid, "purchase", clock, None, dev, qty))
                            last_purchase = (p, clock)
                            # some buyers rate later; rating follows hidden affinity + product quality
                            if rng.random() < 0.45:
                                true = 3.35 + 1.1 * (V[p] @ h["vec"]) / math.sqrt(DIM) + 1.3 * qual[p] + nrng.normal(0, 0.9)
                                r = int(min(5, max(1, round(true))))
                                rt = clock + timedelta(days=rng.randint(2, 21))
                                if rt <= END:
                                    events.append((uid, p, sid, "rating", rt, r, dev, None))
    ev = pd.DataFrame(events, columns=["User ID", "Product ID", "Session ID", "Interaction Type", "Timestamp", "Rating",
                                       "Device", "Quantity"])
    ev["Price"] = ev["Product ID"].map(price).where(ev["Interaction Type"] == "purchase")
    ev = ev.sort_values(["Timestamp", "Session ID"]).reset_index(drop=True)
    ev.insert(0, "Event ID", np.arange(1, len(ev) + 1))
    ev["Timestamp"] = ev["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ev["Rating"] = ev["Rating"].astype("Int64")
    ev["Quantity"] = ev["Quantity"].astype("Int64")
    return ev


# =========================================================================== reviews
def strip_article(f):
    for art in ("a ", "an ", "the "):
        if f.startswith(art):
            return f[len(art):]
    return f


def make_reviews(ev, products, ph, rng):
    rated = ev[ev["Interaction Type"] == "rating"]
    ptype = products.set_index("Product ID")["Product Type"].to_dict()
    brand = products.set_index("Product ID")["Brand"].to_dict()
    rows = []
    for r in rated.itertuples(index=False):
        if rng.random() > 0.75:           # not every rating has text
            continue
        rating, p = int(r.Rating), r[2]
        # people don't write consistently: ~15% of texts read like a neighbouring star rating
        tr = min(5, max(1, rating + rng.choice([-1, 1]))) if rng.random() < 0.15 else rating
        item_l = ptype[p].lower()
        f1, f2 = [strip_article(f) for f in rng.sample(ph[p]["feats"], 2)]
        use = rng.choice(ph[p]["uses"])
        pl = ptype[p] in PLURAL_ITEMS
        words = {"is": "are" if pl else "is", "was": "were" if pl else "was",
                 "this": "these" if pl else "this", "This": "These" if pl else "This"}
        parts = [rng.choice(REVIEW_OPEN[tr]).format(item_l=item_l, use=use, **words)]
        if rng.random() < 0.5:
            parts.append(rng.choice(REVIEW_FILLER))
        if tr >= 4:
            parts.append(rng.choice(REVIEW_PRAISE).format(feature=f1))
            if rng.random() < (0.45 if tr == 4 else 0.15):
                parts.append(rng.choice(REVIEW_MILD))
        elif tr == 3:
            parts += [rng.choice(REVIEW_PRAISE).format(feature=f1), rng.choice(REVIEW_CRITIC).format(feature=f2)]
        else:
            if rng.random() < 0.3:          # "looks nice, but..." style mixed reviews
                parts.append(rng.choice(REVIEW_PRAISE).format(feature=f1).rstrip(".") + ", but that's about it.")
            parts += [x.format(feature=f2) for x in rng.sample(REVIEW_CRITIC, 2 if tr == 1 else 1)]
        if rng.random() < 0.35:
            mood = "positive" if tr >= 4 else "negative" if tr <= 2 else "neutral"
            parts.append(rng.choice(REVIEW_HEDGE[mood]))
        if rng.random() < 0.8:
            parts.append(rng.choice(REVIEW_CLOSE[tr]).format(brand=brand[p]))
        title = rng.choice(REVIEW_TITLES[tr if rng.random() < 0.85 else rating])
        text = " ".join(parts)
        if rng.random() < 0.2:
            text = text.lower()
        rows.append({"Review ID": len(rows) + 1, "User ID": r[1], "Product ID": p, "Rating": rating,
                     "Review Title": title, "Review Text": text,
                     "Review Date": r.Timestamp[:10], "Verified Purchase": True})
    return pd.DataFrame(rows)


# =========================================================================== search queries
def make_search_queries(products, rng):
    by_type = products.groupby("Product Type")["Product ID"].apply(list).to_dict()
    rows = []
    def add(q, ids, qtype):
        if ids:
            rows.append({"Query ID": len(rows) + 1, "Query": q, "Query Type": qtype,
                         "Relevant Product IDs": "|".join(map(str, sorted(ids)))})
    for t, phrases in SEMANTIC.items():
        for q in phrases:
            add(q, by_type.get(t, []), "semantic")                      # meaning, few shared words
        add(t.lower(), by_type.get(t, []), "keyword")                   # exact product type
    for t, ids in by_type.items():                                     # filtered queries
        sub = products[products["Product ID"].isin(ids)]
        cap = int(sub["Price"].median() // 5 * 5) or 5
        add(f"{t.lower()} under £{cap}", sub[sub["Price"] < cap]["Product ID"].tolist(), "filtered_price")
        b = rng.choice(sub["Brand"].unique().tolist())
        add(f"{b} {t.lower()}", sub[sub["Brand"] == b]["Product ID"].tolist(), "filtered_brand")
        col = sub["Colour"].dropna()
        col = col[col.str.len() > 0]
        if len(col):
            c = rng.choice(col.unique().tolist())
            add(f"{c.lower()} {t.lower()}", sub[sub["Colour"] == c]["Product ID"].tolist(), "filtered_colour")
    kids = products[products["Age Range"].str.match(r"^\d")]
    for age in [3, 5, 8, 10, 12]:
        def ok(r):
            a = r.replace("+", "-99").split("-")
            return int(a[0]) <= age <= int(a[1])
        ids = kids[kids["Age Range"].apply(ok) & (kids["Category"] == "Toys")]["Product ID"].tolist()
        add(f"toy for a {age} year old", ids, "gift")
    return pd.DataFrame(rows)


# =========================================================================== chatbot intents
PREFIXES = ["hi, ", "hey ", "um ", "ok so ", "quick question: ", "excuse me, ", "hello, ", "sorry, ", "right, "]
SUFFIXES = [" please", " thanks", " :)", "?", "!!", " asap", " thank you", " x"]


def make_intents(products, rng, per_intent=160):
    names = products["Product Name"].tolist()
    rows = []
    for intent, temps in INTENT_TEMPLATES.items():
        seen = set()
        for _ in range(per_intent * 30):
            if len(seen) >= per_intent:
                break
            ti = rng.randrange(len(temps))
            t = temps[ti]
            ents = {}
            def fill(m):
                key = m.group(1)
                if key in ("product", "product2"):
                    v = rng.choice(names)
                    v = v.split(" - ")[0] if rng.random() < 0.7 else " ".join(v.split()[-2:])  # partial names too
                else:
                    v = rng.choice(INTENT_SLOTS[key])
                ents[key] = v
                return v
            import re
            text = re.sub(r"\{(\w+)\}", fill, t)
            # realistic noise: fillers, casing, missing punctuation, typos
            if rng.random() < 0.3: text = rng.choice(PREFIXES) + text[0].lower() + text[1:]
            if rng.random() < 0.25: text = text.rstrip("?!.") + rng.choice(SUFFIXES)
            if rng.random() < 0.4: text = text.lower()
            if rng.random() < 0.3: text = text.rstrip("?!.")
            if rng.random() < 0.12 and len(text) > 6:
                i = rng.randrange(1, len(text) - 1)
                text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
            if text in seen:
                continue
            seen.add(text)
            rows.append({"Text": text, "Intent": intent, "Entities": json.dumps(ents), "Template ID": f"{intent}_{ti}"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    df.insert(0, "Utterance ID", np.arange(1, len(df) + 1))
    return df


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--products", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data_v3")
    a = ap.parse_args()
    rng, nrng = random.Random(a.seed), np.random.default_rng(a.seed)
    import os
    os.makedirs(a.outdir, exist_ok=True)

    products, ph = make_products(a.products, rng, nrng)
    users, uh = make_users(a.users, rng, nrng)
    ev = simulate(users, uh, products, ph, rng, nrng)
    reviews = make_reviews(ev, products, ph, rng)

    # product review stats + user history columns
    stats = ev[ev["Interaction Type"] == "rating"].groupby("Product ID")["Rating"].agg(["mean", "size"])
    products["Avg Rating"] = products["Product ID"].map(stats["mean"]).astype(float).round(1)
    products["Num Reviews"] = products["Product ID"].map(stats["size"]).fillna(0).astype(int)
    products = products[["Product ID", "Product Name", "Category", "Product Type", "Brand", "Description", "Price",
                         "Colour", "Size", "Material", "Age Range", "Tags", "Avg Rating", "Num Reviews", "In Stock",
                         "Stock Qty", "Delivery Days", "Launch Date"]]
    pcat = products.set_index("Product ID")["Category"]
    browse = ev[ev["Interaction Type"].isin(["view", "click"])].groupby("User ID")["Product ID"].apply(
        lambda s: ", ".join(dict.fromkeys(pcat[s].tolist())))
    buys = ev[ev["Interaction Type"] == "purchase"].groupby("User ID")["Product ID"].apply(
        lambda s: ", ".join(map(str, dict.fromkeys(s.tolist()))))
    users["Browsing History"] = users["User ID"].map(browse).fillna("")
    users["Purchase History"] = users["User ID"].map(buys).fillna("")
    truth = pd.DataFrame([{"User ID": u, "Persona": h["persona"], "Churned": h["churn_date"] is not None,
                           "Churn Date": h["churn_date"].date().isoformat() if h["churn_date"] else ""}
                          for u, h in uh.items()])

    out = {
        "products.csv": products, "user_data.csv": users, "interaction.csv": ev, "reviews.csv": reviews,
        "search_queries.csv": make_search_queries(products, rng), "chatbot_intents.csv": make_intents(products, rng),
        "faq.csv": pd.DataFrame(FAQ, columns=["Topic", "Question", "Answer"]),
        "customer_personas_truth.csv": truth,
    }
    for f, df in out.items():
        df.to_csv(f"{a.outdir}/{f}", index=False)
        print(f"{f:30s} {len(df):>9,} rows  {df.shape[1]:>3} cols")


if __name__ == "__main__":
    main()
