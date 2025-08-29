import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from .metrics import pinball
from .backtest import rolling_splits

def _spawn_like(proto):
    """
    Создаём новый экземпляр эксперта того же класса и переносим ключевые настройки,
    чтобы НЕ терять feature_names / cat_cols / params / base_params.
    """
    cls = type(proto)
    try:
        new = cls()  # конструктор без аргументов
    except Exception:
        # на крайний случай (не должен понадобиться)
        new = cls
    for attr in ("feature_names", "cat_cols", "params", "base_params"):
        if hasattr(proto, attr):
            setattr(new, attr, getattr(proto, attr))
    return new

def _time_decay_weights(dates: pd.Series, halflife: int = None, freq: str = "monthly") -> Optional[np.ndarray]:
    """
    Экспоненциальное затухание во времени:
    w = 0.5 ** (age / halflife), где age — разница в месяцах (monthly) или днях (daily).
    Если halflife=None, возвращает None (без весов).
    """
    if halflife is None:
        return None
    d = pd.to_datetime(dates)
    dmax = pd.to_datetime(d.max())
    if freq == "monthly":
        age = (dmax.year - d.dt.year) * 12 + (dmax.month - d.dt.month)
    else:
        age = (dmax - d).dt.days
    w = np.power(0.5, age / max(1, halflife))
    return w.values.astype(float)

class EnsembleModel:
    """Ансамбль экспертов с экспоненциальными весами по walk-forward pinball(p50)."""
    def __init__(self, experts: List[Tuple[str, object]], feature_names: List[str], eta: float = 5.0):
        self.experts = experts
        self.feature_names = feature_names
        self.eta = eta
        self.weights_: Dict[str, float] = {}
        self.fold_log: List[Dict[str, Any]] = []

    def fit_with_backtest(self, df_feat: pd.DataFrame, y_col: str, date_col: str,
                          k: int, horizon: int, halflife: int = None, freq: str = "monthly"):
        names = [n for n,_ in self.experts]
        losses = np.zeros(len(self.experts), dtype=float)
        self.fold_log = []

        dates = pd.DatetimeIndex(sorted(df_feat[date_col].unique()))
        fold_id = 0
        for (tr0, tr1), (te0, te1) in rolling_splits(dates, k=k, horizon=horizon):
            tr = df_feat[(df_feat[date_col] <= tr1)].dropna()
            te = df_feat[(df_feat[date_col] > tr1) & (df_feat[date_col] <= te1)].dropna()
            if len(tr) < 2 or len(te) == 0:
                continue

            Xtr, ytr = tr.drop(columns=[y_col]), tr[y_col].values
            Xte, yte = te.drop(columns=[y_col]), te[y_col].values
            if len(Xtr) < 2:
                continue

            # временные веса (дают больший вес свежим точкам)
            sw = _time_decay_weights(tr[date_col], halflife=halflife, freq=freq)

            rec = {
                "fold": int(fold_id),
                "train_start": pd.Timestamp(tr[date_col].min()).isoformat(),
                "train_end":   pd.Timestamp(tr[date_col].max()).isoformat(),
                "test_start":  pd.Timestamp(te[date_col].min()).isoformat(),
                "test_end":    pd.Timestamp(te[date_col].max()).isoformat(),
                "n_train": int(len(tr)),
                "n_test":  int(len(te)),
                "experts": []
            }

            best_name = None
            best_loss = None

            for i,(name, proto) in enumerate(self.experts):
                model = _spawn_like(proto)
                try:
                    model.fit(Xtr, ytr, sample_weight=sw)
                    q = model.predict_quantiles(Xte)
                    p50 = q['p50'] if 'p50' in q.columns else q.iloc[:,1]
                    loss = pinball(yte, p50.values, 0.5)
                    losses[i] += loss
                    rec["experts"].append({"name": name, "pinball_p50": float(loss), "status": "ok"})
                    if best_loss is None or loss < best_loss:
                        best_loss, best_name = loss, name
                except Exception as e:
                    # если эксперт не смог обучиться на этом фолде — штрафуем, но не падаем
                    losses[i] += 1e5
                    rec["experts"].append({"name": name, "pinball_p50": 1e5, "status": f"fail: {type(e).__name__}"})

            rec["winner"] = best_name
            self.fold_log.append(rec)
            fold_id += 1

        # вычисляем веса экспертов по накопленным лоссам
        if not np.isfinite(losses).any() or np.all(losses == 0):
            self.weights_ = {n: 1.0/len(self.experts) for n in names}
        else:
            good = np.isfinite(losses) & (losses > 0)
            mean_loss = np.nanmean(losses[good]) if good.any() else 1.0
            losses = np.where(good, losses, mean_loss)
            weights = np.exp(-self.eta * (losses / (mean_loss + 1e-9)))
            weights = weights / np.sum(weights)
            self.weights_ = {n: float(w) for n,w in zip(names, weights)}

        # финальное дообучение экспертов на всей истории (тоже с decay-весами)
        X, y = df_feat.drop(columns=[y_col]), df_feat[y_col].values
        sw_all = _time_decay_weights(df_feat[date_col], halflife=halflife, freq=freq)
        self.experts = [(n, _spawn_like(m).fit(X, y, sample_weight=sw_all)) for (n,m) in self.experts]
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = []
        for name, m in self.experts:
            q = m.predict_quantiles(X)
            q = q.rename(columns={c: f"{name}_{c}" for c in q.columns})
            preds.append(q)
        P = pd.concat(preds, axis=1)
        out = pd.DataFrame(index=X.index)
        for qn in ['p10','p50','p90']:
            cols = [c for c in P.columns if c.endswith(qn)]
            mat = P[cols].values
            w = np.array([self.weights_.get(c.split('_')[0], 0.0) for c in cols])
            out[qn] = mat.dot(w)
        return out

    def save_report(self, path: str):
        Path(path).write_text(
            json.dumps({'weights': self.weights_, 'folds': self.fold_log}, indent=2),
            encoding='utf-8'
        )
