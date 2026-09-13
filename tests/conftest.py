import json
from pathlib import Path

import duckdb
import pytest

from hodge_rank.ingest import ingest


@pytest.fixture
def tiny_yelp_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "yelp"
    directory.mkdir()
    businesses = [
        {"business_id": bid, "name": name, "city": "Seattle", "state": "WA", "categories": category}
        for bid, name, category in [
            ("b_a", "Cafe A", "Coffee & Tea"),
            ("b_b", "Cafe B", "Coffee & Tea"),
            ("b_c", "Cafe C", "Coffee & Tea"),
            ("b_d", "Cafe D", "Coffee & Tea"),
            ("b_bakery", "Corner Bakery", "Bakeries, Coffee & Tea"),
            ("b_pizza", "Pizza Place", "Pizza"),
        ]
    ]
    ratings = {
        "u_active1": {"b_a": 5, "b_b": 3, "b_c": 4, "b_d": 2, "b_bakery": 4},
        "u_active2": {"b_a": 4, "b_b": 3, "b_c": 5, "b_d": 2, "b_bakery": 3},
        "u_active3": {"b_a": 5, "b_b": 2, "b_c": 4, "b_d": 3, "b_bakery": 4},
        "u_active4": {"b_a": 4, "b_b": 3, "b_c": 5, "b_d": 2, "b_bakery": 5},
        "u_inactive1": {"b_a": 5, "b_pizza": 4},
        "u_inactive2": {"b_b": 3, "b_pizza": 5},
    }
    reviews = [
        {
            "review_id": f"{uid}-{bid}",
            "user_id": uid,
            "business_id": bid,
            "stars": stars,
            "date": "2023-05-01 12:00:00",
        }
        for uid, shops in ratings.items()
        for bid, stars in shops.items()
    ]
    for name, rows in [("business", businesses), ("review", reviews)]:
        (directory / f"yelp_academic_dataset_{name}.json").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    return directory


@pytest.fixture
def sample_db(tiny_yelp_dir: Path, tmp_path: Path) -> Path:
    path = tmp_path / "coffee.duckdb"
    with duckdb.connect(str(path)) as con:
        ingest(con, tiny_yelp_dir)
    return path
