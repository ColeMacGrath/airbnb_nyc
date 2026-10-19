import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "datos" / "datos_ini" / "AB_NYC_2019.csv"
CLEAN_PATH = ROOT / "datos" / "datos_limp" / "airbnb_nyc_clean.csv"
RAW_URL = "https://raw.githubusercontent.com/ManarOmar/New-York-Airbnb-2019/master/AB_NYC_2019.csv"
RAW_SHA256 = "e420db40ff10fcb40efc1b5b1648ee0b18a48f4e4537155cecc59fe95d18783a"
DATA_VERSION = "1.0.0"

PRICE_MIN = 10
PRICE_MAX = 1000
MAX_NIGHTS = 365
ID_COLUMNS = ["id", "name", "host_id", "host_name"]
TEXT_COLUMNS = ["name", "host_name", "neighbourhood_group", "neighbourhood", "room_type"]
LOGICAL_KEY = ["host_id", "latitude", "longitude", "room_type", "price"]
CATEGORICAL = ["neighbourhood_group", "neighbourhood", "room_type"]
NUMERIC = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "days_since_last_review",
    "has_reviews",
]
TARGET = "price"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_raw(path=RAW_PATH, url=RAW_URL):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    digest = sha256(path)
    if digest != RAW_SHA256:
        raise ValueError(f"Hash inesperado para {path.name}: {digest}")
    return path


def load_raw(path=RAW_PATH):
    return pd.read_csv(download_raw(path))


def quality_summary(df):
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_id": int(df["id"].duplicated().sum()) if "id" in df else 0,
        "duplicados_logicos": int(df.duplicated(LOGICAL_KEY).sum()) if "host_id" in df else 0,
        "precio_cero": int((df[TARGET] == 0).sum()),
        "precio_fuera_rango": int((~df[TARGET].between(PRICE_MIN, PRICE_MAX)).sum()),
        "noches_mayor_365": int((df["minimum_nights"] > MAX_NIGHTS).sum()),
    }


def clean(df):
    df = df.copy()
    log = []

    def step(name, before):
        log.append({"paso": name, "filas_antes": before, "filas_despues": len(df), "eliminadas": before - len(df)})

    for column in TEXT_COLUMNS:
        df[column] = df[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["name"] = df["name"].fillna("unknown")
    df["host_name"] = df["host_name"].fillna("unknown")

    before = len(df)
    df = df.drop_duplicates()
    step("Duplicados exactos", before)

    before = len(df)
    df = df.drop_duplicates("id")
    step("Duplicados por id", before)

    before = len(df)
    df = df.sort_values("number_of_reviews", ascending=False).drop_duplicates(LOGICAL_KEY).sort_index()
    step("Duplicados logicos (host, coordenadas, tipo, precio)", before)

    before = len(df)
    df = df[df[TARGET].between(PRICE_MIN, PRICE_MAX)]
    step(f"Precio fuera de {PRICE_MIN} a {PRICE_MAX} USD", before)

    df["minimum_nights"] = df["minimum_nights"].clip(upper=MAX_NIGHTS)

    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    reference_date = df["last_review"].max()
    df["has_reviews"] = df["last_review"].notna().astype(int)
    days = (reference_date - df["last_review"]).dt.days
    df["days_since_last_review"] = days.fillna(days.max()).astype(int)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)

    df = df.drop(columns=ID_COLUMNS + ["last_review"])
    df = df[CATEGORICAL + NUMERIC + [TARGET]].reset_index(drop=True)
    return df, pd.DataFrame(log)


def save_clean(df, path=CLEAN_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path=CLEAN_PATH):
    return pd.read_csv(path)


def split_features(df):
    return df[CATEGORICAL + NUMERIC], np.log1p(df[TARGET])


def run():
    raw = load_raw()
    clean_df, log = clean(raw)
    path = save_clean(clean_df)
    print(log.to_string(index=False))
    print(f"Dataset limpio guardado en {path} ({len(clean_df)} filas)")
    print(f"SHA-256 limpio: {sha256(path)}")
    return clean_df, log


if __name__ == "__main__":
    run()
