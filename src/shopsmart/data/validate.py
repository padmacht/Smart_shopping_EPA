"""
Data validation checks (run before any training).

    python -m shopsmart.data.validate

Checks for every dataset in config.yaml:
  - file exists and is not empty
  - primary key is unique and has no missing values
  - foreign keys point to existing rows (e.g. every interaction's Product ID is a real product)
  - date columns parse correctly
Plus a few business rules (prices > 0, ratings 1-5, known interaction types).
Exit code 1 if any check fails, so it can be used in CI/CD.
"""
from __future__ import annotations

import sys

from shopsmart.config import CONFIG
from shopsmart.data.load import load

INTERACTION_TYPES = {"view", "click", "add_to_cart", "purchase", "rating", "wishlist"}


def run_checks() -> list[tuple[str, str, bool, str]]:
    results = []

    def check(dataset, name, ok, detail=""):
        results.append((dataset, name, bool(ok), detail))

    frames = {}
    for name, info in CONFIG["datasets"].items():
        try:
            df = load(name)
        except FileNotFoundError as e:
            check(name, "file exists", False, str(e))
            continue
        frames[name] = df
        check(name, "not empty", len(df) > 0, f"{len(df):,} rows")
        pk = info.get("primary_key")
        if pk:
            check(name, f"primary key '{pk}' unique", df[pk].is_unique, f"{df[pk].duplicated().sum()} duplicates")
            check(name, f"primary key '{pk}' complete", df[pk].notna().all(), f"{df[pk].isna().sum()} missing")
        for col in info.get("parse_dates", []):
            bad = df[col].isna().sum()
            check(name, f"dates in '{col}' parse", bad == 0, f"{bad} unparseable")

    for name, info in CONFIG["datasets"].items():
        for col, ref in (info.get("foreign_keys") or {}).items():
            if name in frames and ref in frames:
                ref_pk = CONFIG["datasets"][ref]["primary_key"]
                orphans = (~frames[name][col].isin(frames[ref][ref_pk])).sum()
                check(name, f"'{col}' exists in {ref}", orphans == 0, f"{orphans} orphan rows")

    if "products" in frames:
        check("products", "prices > 0", (frames["products"]["Price"] > 0).all())
    if "interactions" in frames:
        it = frames["interactions"]
        check("interactions", "known interaction types", it["Interaction Type"].isin(INTERACTION_TYPES).all(),
              str(sorted(set(it["Interaction Type"]) - INTERACTION_TYPES)))
        r = it["Rating"].dropna()
        check("interactions", "ratings between 1 and 5", r.between(1, 5).all())
        check("interactions", "ratings only on rating events",
              it.loc[it["Rating"].notna(), "Interaction Type"].eq("rating").all())
    if "reviews" in frames:
        check("reviews", "ratings between 1 and 5", frames["reviews"]["Rating"].between(1, 5).all())
    return results


def main() -> int:
    results = run_checks()
    width = max(len(f"{d} | {n}") for d, n, _, _ in results)
    for d, n, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {f'{d} | {n}':<{width}}  {detail}")
    failed = sum(not ok for *_, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
