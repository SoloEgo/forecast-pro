# src/pro/smart_train.py
import argparse
from pathlib import Path
import json
import pandas as pd
from joblib import dump
from typing import Optional

from .features import add_features, ALL_FEATURES
from .ensemble import EnsembleModel
from .models.lgbm import LGBMQuantile
from .models.catboost import CatBoostQuantile

# Резервный сезонный наивный (если нет собственного файла models/snaive.py)
try:
    from .models.snaive import SeasonalNaiveQuantile  # type: ignore
except Exception:
    class SeasonalNaiveQuantile:
        """Простейший сезонный наивный бэкап: p10=p50=p90=последнее значение."""
        def __init__(self, freq: str = "monthly", season: Optional[int] = None):
            self.freq = freq
            self.season = season or (12 if freq == "monthly" else 7)
            self.last_: Optional[float] = None
        def fit(self, X, y, sample_weight=None):
            self.last_ = float(y[-1]) if len(y) else 0.0
            return self
        def predict_quantiles(self, X):
            import pandas as pd
            n = len(X)
            val = self.last_ if self.last_ is not None else 0.0
            return pd.DataFrame({"p10": [val]*n, "p50": [val]*n, "p90": [val]*n}, index=X.index)

def _load_best_params(outdir: Path, algo: str, freq: str) -> Optional[dict]:
    fp = outdir / f"best_params_{algo}_{freq}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def main(input_path: str, freq: str, horizon: int, cv: int,
         eta: float, outdir: str, lookback_months: Optional[int], halflife: Optional[int]):

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # --- читаем CSV и ПАРСИМ ДАТЫ (важно: без dayfirst) ---
    df = pd.read_csv(input_path)
    if freq == "monthly":
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce") \
                        .dt.to_period("M").dt.to_timestamp("M")
    elif freq == "daily":
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    else:
        raise ValueError("freq must be 'monthly' or 'daily'")

    # страховка по колонкам
    if "sku" not in df.columns:
        if "SKU" in df.columns:
            df = df.rename(columns={"SKU": "sku"})
        else:
            df["sku"] = "ALL"

    # --- LOOKBACK: оставляем только последние N месяцев/≈дней ---
    if lookback_months is not None:
        if freq == "monthly":
            cutoff = df["date"].max() - pd.DateOffset(months=lookback_months)
        else:  # daily
            cutoff = df["date"].max() - pd.Timedelta(days=lookback_months * 30)
        df = df[df["date"] >= cutoff].copy()

    # --- Генерация признаков ---
    df_feat = add_features(df).dropna()

    # Список фич: берём из ALL_FEATURES то, что реально есть
    feature_cols = [c for c in ALL_FEATURES if c in df_feat.columns]
    cat_cols_present = [c for c in ["sku"] if c in feature_cols]

    # --- Подхват лучших гиперов (если ранее запускался optimize) ---
    best_lgbm = _load_best_params(outdir_p, "lgbm", freq) or {}
    best_cat  = _load_best_params(outdir_p, "catboost", freq) or {}

    # --- Эксперты ---
    lgbm = LGBMQuantile(feature_names=feature_cols, cat_cols=cat_cols_present, **best_lgbm)
    cat  = CatBoostQuantile(feature_names=feature_cols, cat_cols=cat_cols_present, **best_cat)
    snaive = SeasonalNaiveQuantile(freq=freq)

    ens = EnsembleModel(
        experts=[("lgbm", lgbm), ("cat", cat), ("snaive", snaive)],
        feature_names=feature_cols,
        eta=eta
    )

    # обучаем ансамбль (с time-decay весами)
    ens.fit_with_backtest(
        df_feat,
        y_col="qty",
        date_col="date",
        k=cv,
        horizon=horizon,
        halflife=halflife,   # период полураспада весов (None = без весов)
        freq=freq            # "monthly" или "daily"
    )

    # --- Сохранение ансамбля и отчёта ---
    model_path = outdir_p / f"model_ensemble_{freq}.joblib"
    dump(ens, model_path)
    report_path = outdir_p / f"ensemble_report_{freq}.json"
    ens.save_report(str(report_path))

    print(f"Saved model → {model_path}")
    print(f"Saved report → {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Путь к CSV: date, sku, qty, promo_flag")
    ap.add_argument("--freq", choices=["monthly", "daily"], required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--eta", type=float, default=5.0, help="температура ансамбля: выше — жестче штрафуем слабых")
    ap.add_argument("--outdir", default="outputs")
    # новые аргументы
    ap.add_argument("--lookback_months", type=int, default=None,
                    help="Оставить только последние N месяцев (для daily ~N*30 дней).")
    ap.add_argument("--halflife", type=int, default=None,
                    help="Период полураспада временных весов: monthly — в месяцах, daily — в днях (напр. 24).")
    args = ap.parse_args()

    main(args.input, args.freq, args.horizon, args.cv,
         args.eta, args.outdir, args.lookback_months, args.halflife)
