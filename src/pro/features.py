import pandas as pd
import numpy as np
from typing import List

# Глобальный список признаков, который используют модели
ALL_FEATURES: List[str] = [
    # статические/категориальные (как есть; модели могут онехешивать/оэкодить)
    "sku", "category",
    # календарь
    "month", "month_sin", "month_cos", "quarter",
    # лаги и роллинги индивидуально по SKU
    "y_lag1", "y_lag3", "y_lag6", "y_lag12",
    "mean_3m", "mean_6m", "mean_12m",
    # промо
    "promo_flag",
    "promo_share_3m", "promo_share_6m", "promo_share_12m",
    # групповые (leave-one-out) по category
    "grp_qty_excl_lag1", "grp_qty_excl_lag3", "grp_qty_excl_lag6", "grp_qty_excl_lag12",
    "grp_mean_excl_3m", "grp_mean_excl_6m", "grp_mean_excl_12m",
    "grp_trend_excl_3m",
]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "promo_flag" not in out.columns:
        out["promo_flag"] = 0
    if "category" not in out.columns:
        out["category"] = "ALL"
    return out

def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Ожидает колонки: date (pd.Timestamp), sku (str), qty (float/int), promo_flag (0/1), category (str, опц.).
    Возвращает те же строки + фичи. Никаких .dropna() внутри — это делает вызывающая сторона при обучении.
    """
    df = _ensure_columns(df_in).copy()
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    # Календарь
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    # годовая сезонность
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Лаги по SKU
    for L in [1, 3, 6, 12]:
        df[f"y_lag{L}"] = df.groupby("sku")["qty"].shift(L)

    # Роллинги по SKU (mean по предыдущим W месяцам, не включая текущий)
    for W in [3, 6, 12]:
        df[f"mean_{W}m"] = (
            df.groupby("sku")["qty"]
              .shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    # Промо-окна по SKU
    for W in [3, 6, 12]:
        df[f"promo_share_{W}m"] = (
            df.groupby("sku")["promo_flag"]
              .shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    # --- ГРУППОВЫЕ ПРИЗНАКИ (leave-one-out) по category ---
    # Суммарные продажи и доля промо по группе на дату
    g = (df.groupby(["category", "date"], as_index=False)
            .agg(grp_qty=("qty", "sum"),
                 grp_promo_share=("promo_flag", "mean")))
    df = df.merge(g, on=["category", "date"], how="left")
    # исключаем текущий SKU из групповой суммы
    df["grp_qty_excl"] = df["grp_qty"] - df["qty"]
    # лаги по группе (без текущего SKU)
    df = df.sort_values(["category", "date", "sku"])  # устойчивость
    for L in [1, 3, 6, 12]:
        df[f"grp_qty_excl_lag{L}"] = (
            df.groupby("category")["grp_qty_excl"].shift(L)
        )
    # скользящая средняя по группе
    for W in [3, 6, 12]:
        df[f"grp_mean_excl_{W}m"] = (
            df.groupby("category")["grp_qty_excl"]
              .shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
    # простой тренд группы (3м)
    df["grp_trend_excl_3m"] = df["grp_qty_excl_lag1"] - df["grp_qty_excl_lag3"]

    # Вернём в порядке по SKU/дате (так удобнее в вызовах)
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    return df
