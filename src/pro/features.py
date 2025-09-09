# src/pro/features.py
import pandas as pd
import numpy as np
from typing import List

# ====== Списки признаков ======
ALL_FEATURES: List[str] = [  # monthly
    "sku", "category",
    "month", "month_sin", "month_cos", "quarter",
    "y_lag1", "y_lag3", "y_lag6", "y_lag12",
    "mean_3m", "mean_6m", "mean_12m",
    "promo_flag",
    "promo_share_3m", "promo_share_6m", "promo_share_12m",
    "oos_flag",
    "grp_qty_excl_lag1", "grp_qty_excl_lag3", "grp_qty_excl_lag6", "grp_qty_excl_lag12",
    "grp_mean_excl_3m", "grp_mean_excl_6m", "grp_mean_excl_12m",
    "grp_trend_excl_3m",
    "comp_price_index", "comp_qty_index", "comp_promo_flag",
    "comp_price_index_lag1", "comp_price_index_lag3", "comp_price_index_lag6", "comp_price_index_lag12",
    "comp_qty_index_lag1",   "comp_qty_index_lag3",   "comp_qty_index_lag6",   "comp_qty_index_lag12",
    "comp_promo_flag_lag1",  "comp_promo_flag_lag3",  "comp_promo_flag_lag6",  "comp_promo_flag_lag12",
    "comp_price_index_mean_3m", "comp_price_index_mean_6m", "comp_price_index_mean_12m",
    "comp_qty_index_mean_3m",   "comp_qty_index_mean_6m",   "comp_qty_index_mean_12m",
    "comp_promo_share_3m",      "comp_promo_share_6m",      "comp_promo_share_12m",
]

DAILY_ALL: List[str] = [  # daily
    "sku", "category",
    "dow", "dom", "week", "month",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "y_lag1", "y_lag7", "y_lag14", "y_lag28",
    "mean_7d", "mean_14d", "mean_28d",
    "promo_flag",
    "promo_share_7d", "promo_share_28d",
    "oos_flag",
    "grp_qty_excl_lag1", "grp_qty_excl_lag7", "grp_qty_excl_lag14", "grp_qty_excl_lag28",
    "grp_mean_excl_7d", "grp_mean_excl_14d", "grp_mean_excl_28d",
    "grp_trend_excl_7d",
    "comp_price_index", "comp_qty_index", "comp_promo_flag",
    "comp_price_index_lag1",  "comp_price_index_lag7",  "comp_price_index_lag14", "comp_price_index_lag28",
    "comp_qty_index_lag1",    "comp_qty_index_lag7",    "comp_qty_index_lag14",   "comp_qty_index_lag28",
    "comp_promo_flag_lag1",   "comp_promo_flag_lag7",   "comp_promo_flag_lag14",  "comp_promo_flag_lag28",
    "comp_price_index_mean_7d", "comp_price_index_mean_28d",
    "comp_qty_index_mean_7d",   "comp_qty_index_mean_28d",
    "comp_promo_share_7d",      "comp_promo_share_28d",
]

# ====== Служебное ======
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "promo_flag" not in out.columns:
        out["promo_flag"] = 0
    if "category" not in out.columns:
        out["category"] = "ALL"
    if "oos_flag" not in out.columns:
        out["oos_flag"] = 0
    # конкуренты: если нет — безопасные дефолты
    if "comp_price_index" not in out.columns:
        out["comp_price_index"] = 1.0
    if "comp_qty_index" not in out.columns:
        out["comp_qty_index"] = 1.0
    if "comp_promo_flag" not in out.columns:
        out["comp_promo_flag"] = 0
    return out

# ====== MONTHLY ======
def add_features(df_in: pd.DataFrame, id_col: str = "sku", **kwargs) -> pd.DataFrame:

    """
    MONTHLY: ожидает date (Timestamp месячный/конец месяца допустим), sku, qty.
    Без .dropna() — пропуски закрываются снаружи ffill в прогнозе.
    """
    df = _ensure_columns(df_in).copy()
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    if "oos_flag" in df.columns:
        df["oos_flag_lag1"] = df.groupby(id_col)["oos_flag"].shift(1)
        df["oos_flag_lag2"] = df.groupby(id_col)["oos_flag"].shift(2)


    # Календарь
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Лаги по SKU
    for L in [1, 3, 6, 12]:
        df[f"y_lag{L}"] = df.groupby("sku")["qty"].shift(L)

    # Скользящие mean по SKU
    for W in [3, 6, 12]:
        df[f"mean_{W}m"] = (
            df.groupby("sku")["qty"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    # Промо окна по SKU
    for W in [3, 6, 12]:
        df[f"promo_share_{W}m"] = (
            df.groupby("sku")["promo_flag"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    # Групповые признаки (leave-one-out) по category
    g = (df.groupby(["category", "date"], as_index=False)
            .agg(grp_qty=("qty", "sum"),
                 grp_promo_share=("promo_flag", "mean")))
    df = df.merge(g, on=["category", "date"], how="left")
    df["grp_qty_excl"] = df["grp_qty"] - df["qty"]

    df = df.sort_values(["category", "date", "sku"])

    if "oos_flag" in df.columns:
        df["oos_flag_lag1"] = df.groupby(id_col)["oos_flag"].shift(1)
        df["oos_flag_lag2"] = df.groupby(id_col)["oos_flag"].shift(2)

    for L in [1, 3, 6, 12]:
        df[f"grp_qty_excl_lag{L}"] = df.groupby("category")["grp_qty_excl"].shift(L)

    for W in [3, 6, 12]:
        df[f"grp_mean_excl_{W}m"] = (
            df.groupby("category")["grp_qty_excl"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
    df["grp_trend_excl_3m"] = df["grp_qty_excl_lag1"] - df["grp_qty_excl_lag3"]

    # Конкуренты: лаги и окна по категории
    for L in [1, 3, 6, 12]:
        df[f"comp_price_index_lag{L}"] = df.groupby("category")["comp_price_index"].shift(L)
        df[f"comp_qty_index_lag{L}"]   = df.groupby("category")["comp_qty_index"].shift(L)
        df[f"comp_promo_flag_lag{L}"]  = df.groupby("category")["comp_promo_flag"].shift(L)
    for W in [3, 6, 12]:
        df[f"comp_price_index_mean_{W}m"] = (
            df.groupby("category")["comp_price_index"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
        df[f"comp_qty_index_mean_{W}m"] = (
            df.groupby("category")["comp_qty_index"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
        df[f"comp_promo_share_{W}m"] = (
            df.groupby("category")["comp_promo_flag"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    if "oos_flag" in df.columns:
        df["oos_flag_lag1"] = df.groupby(id_col)["oos_flag"].shift(1)
        df["oos_flag_lag2"] = df.groupby(id_col)["oos_flag"].shift(2)

    return df

# ====== DAILY ======
def add_features_daily(df_in: pd.DataFrame, id_col: str = "sku", **kwargs) -> pd.DataFrame:
    """
    DAILY: ожидает date (день), sku, qty. Аналог monthly, но с дневными лагами/окнами.
    """
    df = _ensure_columns(df_in).copy()
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    if "oos_flag" in df.columns:
        df["oos_flag_lag1"] = df.groupby(id_col)["oos_flag"].shift(1)
        df["oos_flag_lag2"] = df.groupby(id_col)["oos_flag"].shift(2)


    # Календарь (день)
    dt = df["date"]
    df["dow"] = dt.dt.dayofweek       # 0..6
    df["dom"] = dt.dt.day             # 1..31
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # Лаги по SKU (дневные)
    for L in [1, 7, 14, 28]:
        df[f"y_lag{L}"] = df.groupby("sku")["qty"].shift(L)

    # Скользящие mean по SKU
    for W in [7, 14, 28]:
        df[f"mean_{W}d"] = (
            df.groupby("sku")["qty"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    # Промо окна по SKU
    for W in [7, 28]:
        df[f"promo_share_{W}d"] = (
            df.groupby("sku")["promo_flag"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    # Групповые признаки (leave-one-out) по category
    g = (df.groupby(["category", "date"], as_index=False)
            .agg(grp_qty=("qty", "sum"),
                 grp_promo_share=("promo_flag", "mean")))
    df = df.merge(g, on=["category", "date"], how="left")
    df["grp_qty_excl"] = df["grp_qty"] - df["qty"]

    df = df.sort_values(["category", "date", "sku"])

    if "oos_flag" in df.columns:
        df["oos_flag_lag1"] = df.groupby(id_col)["oos_flag"].shift(1)
        df["oos_flag_lag2"] = df.groupby(id_col)["oos_flag"].shift(2)

    for L in [1, 7, 14, 28]:
        df[f"grp_qty_excl_lag{L}"] = df.groupby("category")["grp_qty_excl"].shift(L)

    for W in [7, 14, 28]:
        df[f"grp_mean_excl_{W}d"] = (
            df.groupby("category")["grp_qty_excl"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
    df["grp_trend_excl_7d"] = df["grp_qty_excl_lag1"] - df["grp_qty_excl_lag7"]

    # Конкуренты: лаги/окна по категории
    for L in [1, 7, 14, 28]:
        df[f"comp_price_index_lag{L}"] = df.groupby("category")["comp_price_index"].shift(L)
        df[f"comp_qty_index_lag{L}"]   = df.groupby("category")["comp_qty_index"].shift(L)
        df[f"comp_promo_flag_lag{L}"]  = df.groupby("category")["comp_promo_flag"].shift(L)
    for W in [7, 28]:
        df[f"comp_price_index_mean_{W}d"] = (
            df.groupby("category")["comp_price_index"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
        df[f"comp_qty_index_mean_{W}d"] = (
            df.groupby("category")["comp_qty_index"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )
        df[f"comp_promo_share_{W}d"] = (
            df.groupby("category")["comp_promo_flag"].shift(1).rolling(W).mean().reset_index(level=0, drop=True)
        )

    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    if "oos_flag" in df.columns:
        df["oos_flag_lag1"] = df.groupby(id_col)["oos_flag"].shift(1)
        df["oos_flag_lag2"] = df.groupby(id_col)["oos_flag"].shift(2)

    return df
