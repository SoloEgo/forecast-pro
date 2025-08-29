import argparse, pandas as pd, optuna, json
from pathlib import Path
from .config import Config
from .features import add_features, ALL_FEATURES, add_features_daily, DAILY_ALL
from .backtest import evaluate_quantile_model
from .models.lgbm import LGBMQuantile
from .models.catboost import CatBoostQuantile

def objective_lgbm(trial, df_feat, feature_cols, horizon, cv):
    params = dict(
        n_estimators = trial.suggest_int('n_estimators', 300, 1400),
        num_leaves   = trial.suggest_int('num_leaves', 15, 127),
        learning_rate= trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 200),
    )
    def factory():
        return LGBMQuantile(feature_names=feature_cols, **params)
    metrics = evaluate_quantile_model(df_feat, factory, k=cv, horizon=horizon)
    return float(metrics.loc['Pinball_p50','value'])

def objective_cat(trial, df_feat, feature_cols, horizon, cv):
    params = dict(
        depth         = trial.suggest_int('depth', 4, 8),
        learning_rate = trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        iterations    = trial.suggest_int('iterations', 600, 2500),
    )
    def factory():
        return CatBoostQuantile(feature_names=feature_cols, **params)
    metrics = evaluate_quantile_model(df_feat, factory, k=cv, horizon=horizon)
    return float(metrics.loc['Pinball_p50','value'])

def main(input_csv, algo, freq, horizon, cv, trials, outdir):
    cfg = Config(horizon=horizon)
    df = pd.read_csv(input_csv)
    if 'sku' not in df.columns:
        df['sku'] = 'ALL'

    if freq == 'monthly':
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')\
                        .dt.to_period('M').dt.to_timestamp('M')
        df_feat = add_features(df, id_col=cfg.id_col, date_col=cfg.date_col, y_col=cfg.y_col)
        feature_cols = ALL_FEATURES
    else:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df_feat = add_features_daily(df, id_col=cfg.id_col, date_col=cfg.date_col, y_col=cfg.y_col)
        feature_cols = DAILY_ALL

    df_feat = df_feat.dropna(subset=feature_cols + [cfg.y_col]).copy()

    if algo == 'lgbm':
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective_lgbm(t, df_feat, feature_cols, horizon, cv), n_trials=trials)
        df_trials = study.trials_dataframe()
        df_trials.to_csv(Path(outdir)/f'hpo_trials_{algo}_{freq}.csv', index=False)
        print(f"Saved HPO log → {Path(outdir)/f'hpo_trials_{algo}_{freq}.csv'}")
    else:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective_cat(t, df_feat, feature_cols, horizon, cv), n_trials=trials)
        df_trials = study.trials_dataframe()
        df_trials.to_csv(Path(outdir)/f'hpo_trials_{algo}_{freq}.csv', index=False)
        print(f"Saved HPO log → {Path(outdir)/f'hpo_trials_{algo}_{freq}.csv'}")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir)/f'best_params_{algo}_{freq}.json').write_text(json.dumps(study.best_params, indent=2), encoding='utf-8')
    print('Best value:', study.best_value)
    print('Best params:', study.best_params)

    # финальное обучение лучшей моделью на всей истории
    if algo == 'lgbm':
        m = LGBMQuantile(feature_names=feature_cols, **study.best_params)
    else:
        m = CatBoostQuantile(feature_names=feature_cols, **study.best_params)
    X, y = df_feat.drop(columns=[cfg.y_col]), df_feat[cfg.y_col]
    m.fit(X, y)
    import joblib
    joblib.dump(m, Path(outdir)/f'model_{algo}_{freq}_opt.joblib')
    print(f"Saved model → {Path(outdir)/f'model_{algo}_{freq}_opt.joblib'}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--algo', default='lgbm', choices=['lgbm','catboost'])
    ap.add_argument('--freq', default='monthly', choices=['monthly','daily'])
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--cv', type=int, default=4)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()
    main(args.input, args.algo, args.freq, args.horizon, args.cv, args.trials, args.outdir)
