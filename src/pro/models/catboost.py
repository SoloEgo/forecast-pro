# src/pro/models/catboost.py
from typing import List, Optional, Dict
import pandas as pd
from catboost import CatBoostRegressor, Pool

def _is_cat_dtype(s: pd.Series) -> bool:
    return (s.dtype == object) or str(s.dtype).startswith("category")

class CatBoostQuantile:
    """
    Обёртка CatBoost для квантильной регрессии (p10/p50/p90) с
    автодетектом категориальных признаков и поддержкой явного cat_cols.
    """

    def __init__(self,
                 feature_names: List[str],
                 cat_cols: Optional[List[str]] = None,
                 **params):
        self.feature_names = feature_names
        self.cat_cols = list(cat_cols or [])  # может быть пустым
        base = dict(
            depth=6,
            learning_rate=0.08,
            iterations=1200,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        base.update(params)
        self.params: Dict = base
        self.models: Dict[float, CatBoostRegressor] = {}

    def _infer_cat_idx(self, Xc: pd.DataFrame) -> List[int]:
        """Объединяем явный список cat_cols + авто-детект по dtype=object/category."""
        # 1) явные названия, которые реально присутствуют
        explicit = [c for c in self.cat_cols if c in Xc.columns]
        # 2) авто-детект по типу
        auto = [c for c in Xc.columns if _is_cat_dtype(Xc[c])]
        # объединяем и убираем дубли, затем переводим в индексы
        cols = []
        seen = set()
        for c in explicit + auto:
            if c in Xc.columns and c not in seen:
                seen.add(c); cols.append(c)
        return [Xc.columns.get_loc(c) for c in cols]

    def _make_pool(self, X: pd.DataFrame, y=None, sample_weight=None) -> Pool:
        # оставляем только выбранные фичи, копия чтобы не портить исходный df
        Xc = X[self.feature_names].copy()

        # приведение потенциально категориальных к строкам
        for c in Xc.columns:
            if _is_cat_dtype(Xc[c]) or c in self.cat_cols:
                Xc[c] = Xc[c].astype(str)

        cat_idx = self._infer_cat_idx(Xc)

        return Pool(
            data=Xc,
            label=y,
            weight=sample_weight,
            cat_features=cat_idx if cat_idx else None
        )

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self.models = {}
        for a in (0.10, 0.50, 0.90):
            m = CatBoostRegressor(
                loss_function=f"Quantile:alpha={a}",
                **self.params
            )
            pool = self._make_pool(X, y, sample_weight)
            m.fit(pool)
            self.models[a] = m
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        pool = self._make_pool(X)
        out = {}
        for a, m in self.models.items():
            out[f"p{int(a*100)}"] = m.predict(pool)
        return pd.DataFrame(out, index=X.index)
