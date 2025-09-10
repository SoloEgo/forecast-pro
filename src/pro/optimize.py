import argparse
import json
import re
from pathlib import Path
from typing import Optional

import optuna
import pandas as pd
from joblib import dump

from .config import Config
from .features import add_features, ALL_FEATURES, add_features_daily, DAILY_ALL
from .backtest import evaluate_quantile_model
from .models.lgbm import LGBMQuantile
from .models.catboost import CatBoostQuantile

try:
    from .models.snaive import SeasonalNaiveQuantile  # type: ignore
except Exception:
    SeasonalNaiveQuantile = None  # type: ignore


def _san(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(s))[:80] or "ALL"


def make_objective_lgbm(df_feat: pd.DataFrame, used_cols, horizon: int, cv: int, naive_pinball: Optional[float]):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1400),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 200),
        )

        def factory():
            return LGBMQuantile(feature_names=used_cols, **params)

        metrics = evaluate_quantile_model(df_feat, factory, k=cv, horizon=horizon)
        val = float(metrics.loc["Pinball_p50", "value"])
        if naive_pinball is not None and naive_pinball > 0:
            imp = 1 - val / naive_pinball
            print(f"[trial {trial.number}] pinball={val:.3f}; naive={naive_pinball:.3f}; improvement={imp:.1%}")
        return val

    return objective


def make_objective_cat(df_feat: pd.DataFrame, used_cols, horizon: int, cv: int, naive_pinball: Optional[float]):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            iterations=trial.suggest_int("iterations", 600, 2500),
        )

        def factory():
            return CatBoostQuantile(feature_names=used_cols, **params)

        metrics = evaluate_quantile_model(df_feat, factory, k=cv, horizon=horizon)
        val = float(metrics.loc["Pinball_p50", "value"])
        if naive_pinball is not None and naive_pinball > 0:
            imp = 1 - val / naive_pinball
            print(f"[trial {trial.number}] pinball={val:.3f}; naive={naive_pinball:.3f}; improvement={imp:.1%}")
        return val

    return objective


def main(input_csv, algo, freq, horizon, cv, trials, outdir):
    cfg = Config(horizon=horizon)
    df = pd.read_csv(input_csv)
    if "sku" not in df.columns:
        df["sku"] = "ALL"

    # --- parse dates & build features ---
    if freq == "monthly":
        df["date"] = (
            pd.to_datetime(df["date"], errors="coerce")
            .dt.to_period("M").dt.to_timestamp("M")
        )
        df_feat = add_features(df, id_col=cfg.id_col, date_col=cfg.date_col, y_col=cfg.y_col)
        feature_cols = ALL_FEATURES
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df_feat = add_features_daily(df, id_col=cfg.id_col, date_col=cfg.date_col, y_col=cfg.y_col)
        feature_cols = DAILY_ALL

    # --- restrict to actually present features ---
    used_cols = [c for c in feature_cols if c in df_feat.columns]
    missing = [c for c in feature_cols if c not in df_feat.columns]
    print(f"Features used: {len(used_cols)}; missing: {len(missing)}")
    if missing[:10]:
        print("Missing sample:", missing[:10])
    if not used_cols:
        raise RuntimeError("Нет доступных фич — X пустой. Проверь генерацию признаков.")

    # --- Baseline (Seasonal Naive) pinball for reference ---
    naive_pinball = None  # type: Optional[float]
    try:
        if SeasonalNaiveQuantile is not None:
            def _naive_factory():
                return SeasonalNaiveQuantile(freq=freq)
            _m = evaluate_quantile_model(df_feat, _naive_factory, k=cv, horizon=horizon)
            naive_pinball = float(_m.loc["Pinball_p50", "value"])
    except Exception:
        naive_pinball = None
    if naive_pinball is not None:
        print(f"Baseline (sNaive) Pinball_p50 = {naive_pinball:.3f}")

    # --- normalize algo and build objective ---
    algo_norm = (algo or "lgbm").strip().lower()
    if algo_norm in ("lgbm", "lightgbm", "lgbmquantile"):
        obj_fn = make_objective_lgbm(df_feat, used_cols, horizon, cv, naive_pinball)
    elif algo_norm in ("catboost", "cat"):
        obj_fn = make_objective_cat(df_feat, used_cols, horizon, cv, naive_pinball)
    else:
        print(f"Unknown algo '{algo}'. Falling back to lgbm.")
        obj_fn = make_objective_lgbm(df_feat, used_cols, horizon, cv, naive_pinball)

    # --- HPO ---
    study = optuna.create_study(direction="minimize")
    study.optimize(obj_fn, n_trials=trials)

    # --- Save trials with baseline columns ---
    try:
        df_trials = study.trials_dataframe()
        if naive_pinball is not None:
            df_trials["naive_pinball_p50"] = naive_pinball
            df_trials["improvement_vs_naive"] = 1 - df_trials["value"] / naive_pinball
        Path(outdir).mkdir(parents=True, exist_ok=True)
        df_trials.to_csv(Path(outdir) / f"hpo_trials_{algo_norm}_{freq}.csv", index=False)
        print(f"Saved HPO log → {Path(outdir)/f'hpo_trials_{algo_norm}_{freq}.csv'}")
    except Exception:
        pass

    print("Best value:", study.best_value)
    try:
        if naive_pinball is not None:
            print(
                f"Baseline (sNaive) Pinball_p50 = {naive_pinball:.3f} | "
                f"Best = {study.best_value:.3f} | "
                f"Improvement = {(1 - study.best_value/naive_pinball):.1%}"
            )
    except Exception:
        pass
    print("Best params:", study.best_params)

    # --- Save best params (generic and per-SK) ---
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    (out / f"best_params_{algo_norm}_{freq}.json").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")

    # per-SK tagging
    try:
        df_in = pd.read_csv(input_csv)
        if "SK" in df_in.columns:
            vals = df_in["SK"].dropna().astype(str).unique()
            sk = vals[0] if len(vals) == 1 else "ALL"
        else:
            sk = "ALL"
    except Exception:
        sk = "ALL"
    sk_dir = out / "best_params_by_sk"; sk_dir.mkdir(parents=True, exist_ok=True)
    (sk_dir / f"best_params_{algo_norm}_{freq}__sk={_san(sk)}.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )

    # --- Train final model on all data ---
    if algo_norm in ("lgbm","lightgbm","lgbmquantile"):
        model = LGBMQuantile(feature_names=used_cols, **study.best_params)
    else:
        model = CatBoostQuantile(feature_names=used_cols, **study.best_params)
    X, y = df_feat.drop(columns=[cfg.y_col]), df_feat[cfg.y_col]
    model.fit(X, y)
    dump(model, out / f"model_{algo_norm}_{freq}_opt.joblib")
    print(f"Saved model → {Path(out)/f'model_{algo_norm}_{freq}_opt.joblib'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--algo", default="lgbm", choices=["lgbm", "catboost"])
    ap.add_argument("--freq", default="monthly", choices=["monthly", "daily"])
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--cv", type=int, default=4)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    main(args.input, args.algo, args.freq, args.horizon, args.cv, args.trials, args.outdir)
