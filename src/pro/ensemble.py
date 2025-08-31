# src/pro/ensemble.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Эксперт: либо уже созданный объект с .fit/.predict_quantiles,
# либо фабрика (callable), возвращающая такой объект.
ExpertT = Union[Any, Callable[[], Any]]

def _spawn_like(m: ExpertT) -> Any:
    """Вернуть НОВЫЙ экземпляр эксперта, поддерживая фабрики и готовые инстансы."""
    # фабрика?
    if callable(m) and not hasattr(m, "fit"):
        return m()
    # инстанс -> воссоздаём по его полям
    cls = m.__class__
    kwargs = {}
    for attr in ("feature_names", "cat_cols", "params"):
        if hasattr(m, attr):
            kwargs[attr] = getattr(m, attr)
    params = kwargs.pop("params", {})
    obj = cls(**kwargs, **params)
    return obj

def _time_decay_weights(dates: pd.Series,
                        halflife: Optional[int] = None,
                        freq: str = "monthly") -> Optional[np.ndarray]:
    if halflife is None or halflife <= 0:
        return None
    order = dates.sort_values().reset_index(drop=True)
    t = np.arange(len(order), dtype=float)
    lam = np.log(2.0) / float(halflife)
    w = np.exp(lam * (t - t.max()))
    w_series = pd.Series(w, index=order.index)
    w_full = np.zeros(len(dates), dtype=float)
    w_full[order.index.values] = w_series.values
    return w_full

@dataclass
class EnsembleModel:
    experts: List[Tuple[str, ExpertT]]
    eta: float = 5.0
    halflife: Optional[int] = None
    freq: str = "monthly"
    # для совместимости со smart_train.py
    feature_names: Optional[List[str]] = None
    cat_cols: Optional[List[str]] = None

    fitted: List[Tuple[str, Any]] = field(default_factory=list)
    _report: Dict[str, Any] = field(default_factory=dict)

    def fit_with_backtest(self,
                          df_feat: pd.DataFrame,
                          y_col: str,
                          date_col: str,
                          k: int,
                          horizon: int,
                          oos_col: str = "oos_flag",
                          # допускаем прокидывание параметров по имени
                          halflife: Optional[int] = None,
                          freq: Optional[str] = None,
                          **kwargs) -> "EnsembleModel":
        if halflife is not None:
            self.halflife = halflife
        if freq is not None:
            self.freq = freq

        X = df_feat.copy()
        y = X[y_col].values
        dates = pd.to_datetime(X[date_col])

        # time-decay
        sw_time = _time_decay_weights(dates, self.halflife, self.freq)
        if sw_time is None:
            sw_time = np.ones(len(X), dtype=float)

        # downweight OOS
        if oos_col in X.columns:
            sw = sw_time * (1.0 - 0.9 * X[oos_col].astype(float).values)
        else:
            sw = sw_time

        # финальное обучение всех экспертов на всей истории
        fitted: List[Tuple[str, Any]] = []
        for name, proto in self.experts:
            model = _spawn_like(proto)
            if not hasattr(model, "fit") or not hasattr(model, "predict_quantiles"):
                raise TypeError(f"Expert '{name}' не имеет методов fit/predict_quantiles")
            model.fit(X, y, sample_weight=sw)
            fitted.append((name, model))
        self.fitted = fitted

        # соберём краткий отчёт (метаданные)
        self._report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "freq": self.freq,
            "eta": self.eta,
            "halflife": self.halflife,
            "rows": int(len(X)),
            "features_total": int(len(X.columns)),
            "y_col": y_col,
            "date_min": str(dates.min()) if len(dates) else None,
            "date_max": str(dates.max()) if len(dates) else None,
            "cv_folds": int(k),
            "horizon": int(horizon),
            "experts": [
                {
                    "name": name,
                    "class": m.__class__.__name__,
                    # если у моделей есть эти поля — добавим
                    "feature_names": getattr(m, "feature_names", None),
                    "cat_cols": getattr(m, "cat_cols", None),
                }
                for name, m in self.fitted
            ],
        }
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Ансамбль не обучен: вызовите fit_with_backtest()")
        preds = []
        for _, m in self.fitted:
            preds.append(m.predict_quantiles(X))
        out = {col: np.mean([p[col].values for p in preds], axis=0) for col in ("p10", "p50", "p90")}
        return pd.DataFrame(out, index=X.index)

    def save_report(self, path: str) -> str:
        """
        Сохранить краткий отчёт об ансамбле (метаданные обучения).
        Формат: .json (если расширение .json), иначе .md
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._report, f, ensure_ascii=False, indent=2)
        else:
            # Markdown-версия
            lines = []
            lines.append(f"# Forecast Ensemble Report")
            lines.append("")
            for k, v in self._report.items():
                if k != "experts":
                    lines.append(f"- **{k}**: {v}")
            lines.append("")
            lines.append("## Experts")
            for e in self._report.get("experts", []):
                lines.append(f"- **{e['name']}**: {e['class']}")
                if e.get("feature_names") is not None:
                    lines.append(f"  - feature_names: {len(e['feature_names'])} cols")
                if e.get("cat_cols") is not None:
                    lines.append(f"  - cat_cols: {e['cat_cols']}")
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        return str(p)
