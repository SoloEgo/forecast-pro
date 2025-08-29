import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from typing import Optional, Dict

from .features import add_features

def _parse_dates(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "monthly":
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")\
                        .dt.to_period("M").dt.to_timestamp("M")
    else:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    return df

def _next_date(d: pd.Timestamp, freq: str) -> pd.Timestamp:
    return (d + pd.offsets.MonthEnd(1)) if freq == "monthly" else (d + pd.Timedelta(days=1))

def _load_promo_plan(path: Optional[str], freq: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    plan = pd.read_csv(path)
    plan = _parse_dates(plan, freq)
    for col in ["sku", "promo_flag"]:
        if col not in plan.columns:
            raise ValueError(f"promo_plan must contain column '{col}'")
    if "category" not in plan.columns:
        plan["category"] = "ALL"
    return plan[["date", "sku", "category", "promo_flag"]].copy()

def _get_promo_flag(plan: Optional[pd.DataFrame], date: pd.Timestamp, sku) -> int:
    if plan is None:
        return 0
    row = plan[(plan["date"] == date) & (plan["sku"] == sku)]
    if len(row) == 0:
        return 0
    return int(row["promo_flag"].iloc[0])

# ---------- ОДИНОЧНЫЙ ПРОГНОЗ ДЛЯ ОДНОГО SKU (как раньше, но устойчивый) ----------
def forecast_one_sku(df_hist: pd.DataFrame,
                     m,
                     horizon: int,
                     freq: str,
                     promo_plan: Optional[pd.DataFrame],
                     calibrate_last_k: int = 0,
                     calibrate_clip_low: float = 0.6,
                     calibrate_clip_high: float = 1.4) -> pd.DataFrame:
    sku = df_hist["sku"].iloc[0]
    cat = df_hist["category"].iloc[0] if "category" in df_hist.columns else "ALL"
    df_hist = df_hist.sort_values("date").copy()

    # 1) Исторические фичи
    hist_feat = add_features(df_hist)

    # 2) Калибровка per-SKU
    ratio = 1.0
    if calibrate_last_k and calibrate_last_k > 0 and not hist_feat.empty:
        tail = hist_feat.tail(calibrate_last_k)
        if len(tail) >= 2:
            X_tail = tail.drop(columns=["qty"])
            q_tail = m.predict_quantiles(X_tail)
            ratio = np.median(tail["qty"].values / np.maximum(q_tail["p50"].values, 1e-6))
            ratio = float(np.clip(ratio, calibrate_clip_low, calibrate_clip_high))

    # 3) Рекурсивная генерация
    last_date = df_hist["date"].max()
    rows = []
    work = df_hist.copy()

    for _ in range(horizon):
        next_d = _next_date(last_date, freq)
        stub = pd.DataFrame({
            "date": [next_d],
            "sku": [sku],
            "category": [cat],
            "qty": [np.nan],
            "promo_flag": [_get_promo_flag(promo_plan, next_d, sku)]
        })
        work = pd.concat([work, stub], ignore_index=True)

        feat_all = add_features(work).sort_values(["sku", "date"])
        feat_cols = [c for c in feat_all.columns if c not in ("date", "qty")]
        # ffill пропусков по SKU
        feat_all[feat_cols] = feat_all.groupby("sku")[feat_cols].ffill()
        x_next = feat_all.iloc[[-1]].drop(columns=["qty"])

        q = m.predict_quantiles(x_next)
        p10 = float(q["p10"].iloc[0] * ratio)
        p50 = float(q["p50"].iloc[0] * ratio)
        p90 = float(q["p90"].iloc[0] * ratio)

        rows.append({"date": next_d, "sku": sku, "promo_flag": int(stub["promo_flag"].iloc[0]),
                     "p10": p10, "p50": p50, "p90": p90})
        work.loc[work["date"] == next_d, "qty"] = p50
        last_date = next_d

    return pd.DataFrame(rows)

# ---------- СОВМЕСТНЫЙ ПРОГНОЗ ДЛЯ ВСЕХ SKU (JOINT) ----------
def forecast_joint(df_hist_all: pd.DataFrame,
                   m,
                   horizon: int,
                   freq: str,
                   promo_plan: Optional[pd.DataFrame],
                   calibrate_last_k: int = 0,
                   calibrate_clip_low: float = 0.6,
                   calibrate_clip_high: float = 1.4) -> pd.DataFrame:
    # подготовка
    df_hist_all = df_hist_all.copy()
    if "category" not in df_hist_all.columns:
        df_hist_all["category"] = "ALL"
    df_hist_all = df_hist_all.sort_values(["sku", "date"]).reset_index(drop=True)

    # калибровка per-SKU
    ratios: Dict[str, float] = {}
    for sku, df_sku in df_hist_all.groupby("sku"):
        hist_feat = add_features(df_sku)
        ratio = 1.0
        if calibrate_last_k and calibrate_last_k > 0 and not hist_feat.empty:
            tail = hist_feat.tail(calibrate_last_k)
            if len(tail) >= 2:
                X_tail = tail.drop(columns=["qty"])
                q_tail = m.predict_quantiles(X_tail)
                r = np.median(tail["qty"].values / np.maximum(q_tail["p50"].values, 1e-6))
                ratio = float(np.clip(r, calibrate_clip_low, calibrate_clip_high))
        ratios[sku] = ratio

    # общий work-фрейм
    work = df_hist_all.copy()
    rows = []
    last_date = work["date"].max()

    for _ in range(horizon):
        next_d = _next_date(last_date, freq)

        # добавляем "заготовки" по всем SKU за next_d
        stubs = []
        for sku, grp in work.groupby("sku"):
            cat = grp["category"].iloc[0] if "category" in grp.columns else "ALL"
            stubs.append({
                "date": next_d, "sku": sku, "category": cat,
                "qty": np.nan,
                "promo_flag": _get_promo_flag(promo_plan, next_d, sku)
            })
        work = pd.concat([work, pd.DataFrame(stubs)], ignore_index=True)

        # пересчитываем фичи на всем фрейме (групповые признаки теперь учитывают всю историю)
        feat_all = add_features(work).sort_values(["sku", "date"]).reset_index(drop=True)
        feat_cols = [c for c in feat_all.columns if c not in ("date", "qty")]

        # аккуратный ffill по SKU для пропусков (лагам и роллингам иногда не хватает истории на ранних шагах)
        feat_all[feat_cols] = feat_all.groupby("sku")[feat_cols].ffill()

        # берём все строки на next_d для предсказаний
        mask_next = feat_all["date"] == next_d
        X_next = feat_all.loc[mask_next].drop(columns=["qty"])

        # предсказываем квантили батчем
        Q = m.predict_quantiles(X_next)  # индекс соответствует X_next

        # применяем per-SKU калибровку
        out_batch = []
        for (idx, row), (_, qrow) in zip(X_next.iterrows(), Q.iterrows()):
            sku = row["sku"]
            ratio = ratios.get(sku, 1.0)
            out_batch.append({
                "date": next_d,
                "sku": sku,
                "promo_flag": int(row["promo_flag"]),
                "p10": float(qrow["p10"] * ratio),
                "p50": float(qrow["p50"] * ratio),
                "p90": float(qrow["p90"] * ratio),
            })
        rows.extend(out_batch)

        # обновляем qty=p50 для всех SKU на next_d (чтобы лаги/группа на следующем шаге учитывали текущие предсказания)
        upd = pd.DataFrame({
            "sku": [r["sku"] for r in out_batch],
            "date": [r["date"] for r in out_batch],
            "qty":  [r["p50"] for r in out_batch],
        })
        work = work.merge(upd, on=["sku", "date"], how="left", suffixes=("", "_pred"))
        work["qty"] = np.where(work["qty"].isna(), work["qty_pred"], work["qty"])
        work = work.drop(columns=["qty_pred"])

        last_date = next_d

    return pd.DataFrame(rows)

def main(input_path: str,
         model_path: str,
         horizon: int,
         outdir: str,
         promo_scenario: Optional[str],
         freq: str,
         joint: bool,
         calibrate_last_k: int,
         calibrate_clip_low: float,
         calibrate_clip_high: float):
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # История
    df = pd.read_csv(input_path)
    df = _parse_dates(df, freq)
    if "sku" not in df.columns:
        df["sku"] = "ALL"
    if "promo_flag" not in df.columns:
        df["promo_flag"] = 0
    if "category" not in df.columns:
        df["category"] = "ALL"

    m = load(model_path)
    promo_plan = _load_promo_plan(promo_scenario, freq)

    if joint:
        forecast = forecast_joint(df, m, horizon, freq, promo_plan,
                                  calibrate_last_k=calibrate_last_k,
                                  calibrate_clip_low=calibrate_clip_low,
                                  calibrate_clip_high=calibrate_clip_high)
    else:
        out = []
        for sku, df_sku in df.groupby("sku"):
            fut = forecast_one_sku(df_sku, m, horizon, freq, promo_plan,
                                   calibrate_last_k=calibrate_last_k,
                                   calibrate_clip_low=calibrate_clip_low,
                                   calibrate_clip_high=calibrate_clip_high)
            out.append(fut)
        forecast = pd.concat(out, ignore_index=True)

    forecast[["p10","p50","p90"]] = forecast[["p10","p50","p90"]].clip(lower=0).round(2)
    out_csv = outdir_p / f"forecast_{freq}.csv"
    forecast.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--promo_scenario", default=None, help="CSV: date,sku,category(опц.),promo_flag")
    ap.add_argument("--freq", choices=["monthly","daily"], required=True)

    # новый флаг: совместный прогноз всех SKU
    ap.add_argument("--joint", action="store_true", help="Совместный прогноз всех SKU по шагам.")

    # калибровка bias
    ap.add_argument("--calibrate_last_k", type=int, default=0,
                    help="Сколько последних исторических точек использовать для пост-калибровки (0=выкл).")
    ap.add_argument("--calibrate_clip_low", type=float, default=0.6)
    ap.add_argument("--calibrate_clip_high", type=float, default=1.4)

    args = ap.parse_args()
    main(args.input, args.model_path, args.horizon, args.outdir,
         args.promo_scenario, args.freq, args.joint,
         args.calibrate_last_k, args.calibrate_clip_low, args.calibrate_clip_high)
