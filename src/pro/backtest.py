import pandas as pd, numpy as np
from typing import Callable
from .metrics import wape, mase, smape, pinball
import re

def _max_required_lag(cols):
    lags = []
    for c in cols:
        m = re.match(r'^y_lag(\d+)$', c)
        if m:
            lags.append(int(m.group(1)))
    return max(lags) if lags else 0

def rolling_splits(dates: pd.DatetimeIndex, k: int, horizon: int):
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    n = len(dates)
    if n < 3:
        raise ValueError(f"Слишком мало временных точек: {n}. Нужны ≥3.")
    valid_max_anchor = n - horizon - 1
    if valid_max_anchor < 0:
        a = 0
        yield (dates[0], dates[a]), (dates[a+1], dates[-1])
        return
    k_eff = max(1, min(k, valid_max_anchor + 1))
    anchors = np.linspace(0, valid_max_anchor, k_eff, dtype=int)
    for a in anchors:
        yield (dates[0], dates[a]), (dates[a+1], dates[a + horizon])

def evaluate_quantile_model(df_feat: pd.DataFrame, model_factory: Callable[[], object], k: int, horizon: int,
                            id_col='sku', date_col='date', y_col='qty') -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(df_feat[date_col].unique()))
    max_lag = _max_required_lag(df_feat.columns)
    rows = []
    for (tr0, tr1), (te0, te1) in rolling_splits(dates, k=k, horizon=horizon):
        tr = df_feat[(df_feat[date_col] <= tr1)].dropna()
        te = df_feat[(df_feat[date_col] > tr1) & (df_feat[date_col] <= te1)].dropna()
        if len(tr) < max(2, max_lag + 1) or len(te) == 0:
            continue
        Xtr, ytr = tr.drop(columns=[y_col]), tr[y_col].values
        Xte, yte = te.drop(columns=[y_col]), te[y_col].values
        if len(Xtr) < 2:
            continue
        m = model_factory(); m.fit(Xtr, ytr)
        q = m.predict_quantiles(Xte)
        p50 = q['p50'] if 'p50' in q.columns else q.iloc[:,1]
        rows.append({
            'wMAPE': wape(yte, p50.values),
            'sMAPE': smape(yte, p50.values),
            'MASE' : mase(yte, p50.values),
            'Pinball_p10': pinball(yte, q['p10'].values, 0.1),
            'Pinball_p50': pinball(yte, p50.values, 0.5),
            'Pinball_p90': pinball(yte, q['p90'].values, 0.9),
        })
    if not rows:
        return pd.DataFrame({'value':[1e6,1e6,1e6,1e6,1e6,1e6]},
                            index=['wMAPE','sMAPE','MASE','Pinball_p10','Pinball_p50','Pinball_p90'])
    return pd.DataFrame(rows).mean(numeric_only=True).to_frame('value')
