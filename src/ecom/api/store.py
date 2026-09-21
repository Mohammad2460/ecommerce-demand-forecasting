"""Read-only access to pipeline artifacts, cached per process."""

from __future__ import annotations

import json
from functools import cache, lru_cache

import joblib
import pandas as pd

from ecom.config import models_dir, processed_dir


class ArtifactMissing(RuntimeError):
    pass


@cache
def table(name: str) -> pd.DataFrame:
    path = processed_dir() / f"{name}.parquet"
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found - run `make pipeline` first")
    return pd.read_parquet(path)


@cache
def model(name: str):
    path = models_dir() / f"{name}.joblib"
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found - run `make pipeline` first")
    return joblib.load(path)


@cache
def json_artifact(name: str) -> dict:
    path = processed_dir() / f"{name}.json"
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found - run `make pipeline` first")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def customers() -> pd.DataFrame:
    return table("customers").set_index("customer_unique_id")


def clear() -> None:
    for fn in (table, model, json_artifact, customers):
        fn.cache_clear()
