from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import numpy as np
import pandas as pd
from joblib import load

try:
    from .features import add_features
except Exception:
    from src.pro.features import add_features  # type: ignore


def _parse_dates(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    if freq == "monthly":
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _next_date(d: pd.Timestamp, freq: str) -> pd.Timestamp:
    return (d + pd.offsets.MonthEnd(1)) if freq == "monthly" else (d + pd.Timedelta(days=1))


def _load_promo_plan(path: Optional[str], freq: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    plan = pd.read_csv(path)
    plan = _parse_dates(plan, freq)
    req = ["sku", "promo_flag"]
    miss = [c for c in req if c not in plan.columns]
    if miss:
        raise ValueError(f"promo_plan is missing required columns: {miss}")
    if "date" not in plan.columns:
        raise ValueError("promo_plan must contain 'date' column aligned to the forecast horizon.")
    plan["promo_flag"] = plan["promo_flag"].astype(int)
    return plan[["sku", "date", "promo_flag"]]


class QuantileModelAdapter:
    """
    Tries multiple calling conventions to fetch p10/p50/p90:
      - dict {'p10','p50','p90'} of estimators with .predict
      - .predict_quantiles(X, quantiles=[...])   (kw-arg)
      - .predict_quantiles(X, [...])             (positional)
      - .predict_quantiles(X)                    (no-arg; model holds quantiles inside)
      - .predict(X) returning DataFrame/dict/np with three columns
    """
    def __init__(self, obj: Any):
        self.obj = obj

    @staticmethod
    def _as_array(y) -> np.ndarray:
        if isinstance(y, (list, tuple)):
            return np.asarray(y).ravel()
        if isinstance(y, pd.Series):
            return y.to_numpy().ravel()
        if isinstance(y, pd.DataFrame):
            return y.to_numpy().ravel()
        return np.asarray(y).ravel()

    @staticmethod
    def _parse_result(res) -> Optional[Tuple[float, float, float]]:
        # ndarray / list-like
        if isinstance(res, (list, tuple, np.ndarray)):
            arr = np.asarray(res).astype(float).ravel()
            if arr.size >= 3:
                return float(arr[0]), float(arr[1]), float(arr[2])
        # DataFrame
        if isinstance(res, pd.DataFrame) and len(res) >= 1:
            cols = {str(c).lower(): c for c in res.columns}
            def pick(keys, idx):
                for k in keys:
                    kl = k.lower()
                    if kl in cols:
                        return float(res[cols[kl]].iloc[0])
                return float(res.iloc[0, idx])
            p10 = pick(("p10","q10","0.1","10%"), 0)
            p50 = pick(("p50","median","q50","0.5","50%"), 1)
            p90 = pick(("p90","q90","0.9","90%"), 2)
            return p10, p50, p90
        # dict
        if isinstance(res, dict):
            def pick(d, keys, fallback_idx):
                for k in keys:
                    if k in d: return float(np.ravel(d[k])[0])
                return float(np.ravel(list(d.values()))[fallback_idx])
            p10 = pick(res, ("p10","P10","q10","Q10","0.1","10%"), 0)
            p50 = pick(res, ("p50","P50","median","50%","0.5"), 1)
            p90 = pick(res, ("p90","P90","q90","Q90","0.9","90%"), 2)
            return p10, p50, p90
        return None

    def predict_row(self, X_row: pd.DataFrame) -> Tuple[float, float, float]:
        obj = self.obj

        # Case 1: dict of estimators (common in our codebase)
        if isinstance(obj, dict) and all(k in obj for k in ("p10", "p50", "p90")):
            p10 = self._as_array(obj["p10"].predict(X_row))[0]
            p50 = self._as_array(obj["p50"].predict(X_row))[0]
            p90 = self._as_array(obj["p90"].predict(X_row))[0]
            return float(p10), float(p50), float(p90)

        # Case 2: dedicated predict_quantiles with various signatures
        if hasattr(obj, "predict_quantiles"):
            for mode in ("kw", "pos", "none"):
                try:
                    if mode == "kw":
                        res = obj.predict_quantiles(X_row, quantiles=[0.1, 0.5, 0.9])
                    elif mode == "pos":
                        res = obj.predict_quantiles(X_row, [0.1, 0.5, 0.9])
                    else:
                        res = obj.predict_quantiles(X_row)
                    parsed = self._parse_result(res)
                    if parsed is not None:
                        return parsed
                except TypeError:
                    # try the next calling convention
                    continue
                except Exception:
                    continue

        # Case 3: generic predict returning DataFrame/dict/np with three columns
        if hasattr(obj, "predict"):
            try:
                res = obj.predict(X_row)
                parsed = self._parse_result(res)
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        raise TypeError("Unsupported model type: can't obtain p10/p50/p90 from the provided model.")

def _build_future_calendar(last_date: pd.Timestamp, horizon: int, freq: str) -> pd.DatetimeIndex:
    dates = []
    d = last_date
    for _ in range(horizon):
        d = _next_date(d, freq)
        dates.append(d)
    return pd.DatetimeIndex(dates)


def _get_future_promo(stub: pd.DataFrame, date: pd.Timestamp, plan: Optional[pd.DataFrame]) -> int:
    if plan is None:
        if "promo_flag" in stub.columns and not stub["promo_flag"].isna().all():
            return int(stub["promo_flag"].iloc[-1])
        return 0
    row = plan[(plan["date"] == date)]
    if row.empty:
        return 0
    if "sku" in stub.columns and "sku" in plan.columns:
        sku = stub["sku"].iloc[0]
        r = plan[(plan["date"] == date) & (plan["sku"] == sku)]
        if not r.empty:
            return int(r["promo_flag"].iloc[0])
    return int(row["promo_flag"].iloc[0])


def forecast_per_sku(df_hist: pd.DataFrame,
                     model: QuantileModelAdapter,
                     horizon: int,
                     freq: str,
                     promo_plan: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_hist = df_hist.sort_values("date").reset_index(drop=True)
    last_date = df_hist["date"].max()
    out_rows = []
    work = df_hist.copy()

    for _ in range(horizon):
        next_d = _next_date(last_date, freq)
        stub = pd.DataFrame({
            "date": [next_d],
            "sku": [df_hist["sku"].iloc[0]],
            "qty": [np.nan],
            "promo_flag": [0]
        })
        stub.loc[0, "promo_flag"] = _get_future_promo(df_hist[["sku"]].assign(promo_flag=df_hist.get("promo_flag", 0)), next_d, promo_plan)

        work_ext = pd.concat([work, stub], ignore_index=True)
        feat = add_features(work_ext, id_col="sku", date_col="date", y_col="qty", freq=freq)

        X = feat.loc[feat["date"] == next_d].copy()
        if "qty" in X.columns:
            X = X.drop(columns=["qty"])

        p10, p50, p90 = model.predict_row(X)
        p10, p50, p90 = max(p10, 0.0), max(p50, 0.0), max(p90, 0.0)
        arr = np.array([p10, p50, p90], dtype=float); arr.sort()
        p10, p50, p90 = float(arr[0]), float(arr[1]), float(arr[2])

        out_rows.append({
            "date": next_d, "sku": df_hist["sku"].iloc[0],
            "promo_flag": int(stub["promo_flag"].iloc[0]),
            "p10": p10, "p50": p50, "p90": p90
        })

        next_hist = {"date": next_d, "sku": df_hist["sku"].iloc[0], "qty": p50, "promo_flag": int(stub["promo_flag"].iloc[0])}
        work = pd.concat([work, pd.DataFrame([next_hist])], ignore_index=True)
        last_date = next_d

    return pd.DataFrame(out_rows)


def forecast_joint(df_hist_all: pd.DataFrame,
                   model_obj: Any,
                   horizon: int,
                   freq: str,
                   promo_plan: Optional[pd.DataFrame]) -> pd.DataFrame:
    adapter = QuantileModelAdapter(model_obj)
    rows = []
    for _, g in df_hist_all.groupby("sku"):
        rows.append(forecast_per_sku(g, adapter, horizon, freq, promo_plan))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date","sku","p10","p50","p90","promo_flag"])


def apply_quantile_calibration_all(df_forecast: pd.DataFrame,
                                   df_history: pd.DataFrame,
                                   horizon: int,
                                   k: int,
                                   clip_low: float,
                                   clip_high: float) -> pd.DataFrame:
    if k <= 0:
        return df_forecast.copy()

    f = df_forecast.copy()
    h = df_history.copy()

    def _cal_one(sku_df_f: pd.DataFrame, sku_df_h: pd.DataFrame) -> pd.DataFrame:
        sku_df_f = sku_df_f.sort_values("date").reset_index(drop=True)
        sku_df_h = sku_df_h.sort_values("date").reset_index(drop=True)
        y_last = sku_df_h["qty"].dropna().tail(k)
        p50_first = sku_df_f["p50"].head(min(k, horizon))
        if y_last.empty or p50_first.empty or float(p50_first.mean()) == 0.0:
            g = 1.0
        else:
            g_raw = float(y_last.mean()) / float(p50_first.mean())
            g = float(np.clip(g_raw, clip_low, clip_high))
        sku_df_f[["p10","p50","p90"]] = sku_df_f[["p10","p50","p90"]] * g
        return sku_df_f

    res = []
    for sku, g in f.groupby("sku"):
        res.append(_cal_one(g, h[h["sku"] == sku]))
    f2 = pd.concat(res, ignore_index=True) if res else f

    arr = np.vstack([f2["p10"].values, f2["p50"].values, f2["p90"].values])
    arr.sort(axis=0)
    f2["p10"], f2["p50"], f2["p90"] = arr[0], arr[1], arr[2]
    return f2


def enforce_quantile_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    arr = np.vstack([df["p10"].values, df["p50"].values, df["p90"].values])
    arr.sort(axis=0)
    df["p10"], df["p50"], df["p90"] = arr[0], arr[1], arr[2]
    return df


def main(input_csv: str,
         model_path: str,
         horizon: int,
         outdir: str,
         promo_scenario: Optional[str],
         freq: str,
         joint: bool,
         calibrate_last_k: int,
         calibrate_clip_low: float,
         calibrate_clip_high: float):
    df = pd.read_csv(input_csv)
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain column 'date'.")
    if "sku" not in df.columns:
        df["sku"] = "ALL"
    if "qty" not in df.columns:
        raise ValueError("Input CSV must contain column 'qty'.")

    df = _parse_dates(df, freq)
    keep_cols = [c for c in ["date", "sku", "qty", "promo_flag", "SK", "category"] if c in df.columns]
    df = df[keep_cols].sort_values(["sku", "date"]).reset_index(drop=True)

    plan = _load_promo_plan(promo_scenario, freq) if promo_scenario else None

    mp = Path(model_path)
    if mp.is_dir():
        joblibs = sorted(Path(model_path).glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not joblibs:
            raise FileNotFoundError(f"No model files (*.joblib) found in directory: {model_path}")
        mdl_obj = load(joblibs[0])
    else:
        mdl_obj = load(mp)

    if joint:
        df_forecast = forecast_joint(df, mdl_obj, horizon, freq, plan)
    else:
        df_forecast = forecast_joint(df, mdl_obj, horizon, freq, plan)

    for c in ("p10","p50","p90"):
        df_forecast[c] = df_forecast[c].astype(float).clip(lower=0.0)

    meta_cols = [c for c in ["SK", "category"] if c in df.columns]
    if meta_cols:
        meta = df.drop_duplicates(subset=["sku"])[["sku"] + meta_cols]
        df_forecast = df_forecast.merge(meta, on="sku", how="left")

    df_forecast = apply_quantile_calibration_all(
        df_forecast=df_forecast,
        df_history=df[["date","sku","qty"] + (["promo_flag"] if "promo_flag" in df.columns else [])],
        horizon=horizon,
        k=calibrate_last_k,
        clip_low=calibrate_clip_low,
        clip_high=calibrate_clip_high,
    )

    df_forecast = enforce_quantile_monotonicity(df_forecast)

    outdir_p = Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    out_csv = outdir_p / f"forecast_{freq}.csv"
    col_order = ["date", "sku"] + (["SK"] if "SK" in df_forecast.columns else []) \
                + ["promo_flag" if "promo_flag" in df_forecast.columns else None, "p10","p50","p90"]
    col_order = [c for c in col_order if c is not None and c in df_forecast.columns]
    df_forecast = df_forecast.sort_values(["sku", "date"]).reset_index(drop=True)
    df_forecast[col_order].to_csv(out_csv, index=False)
    print(f"Saved forecast to: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("forecast")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--promo_scenario", type=str, default=None)
    ap.add_argument("--freq", choices=["monthly", "daily"], required=True)
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--calibrate_last_k", type=int, default=0)
    ap.add_argument("--calibrate_clip_low", type=float, default=0.6)
    ap.add_argument("--calibrate_clip_high", type=float, default=1.4)
    args = ap.parse_args()
    main(args.input, args.model_path, args.horizon, args.outdir, args.promo_scenario, args.freq,
         args.joint, args.calibrate_last_k, args.calibrate_clip_low, args.calibrate_clip_high)
