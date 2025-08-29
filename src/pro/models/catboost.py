import pandas as pd
from typing import List, Optional
from catboost import CatBoostRegressor, Pool

class CatBoostQuantile:
    """Квантильный бустинг (CatBoost), аккуратно работает с категориальными и sample_weight."""
    def __init__(
        self,
        depth: int = 6,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        feature_names: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ):
        self.feature_names = feature_names or []
        self.cat_cols = cat_cols or ['sku']
        self.params = dict(
            depth=depth,
            iterations=iterations,
            learning_rate=learning_rate,
            random_seed=42,
            allow_writing_files=False,
            logging_level='Silent',
        )
        self.m = {}

    def _make_pool(self, X: pd.DataFrame, y=None, sample_weight=None):
        use_feats = [c for c in self.feature_names if c in X.columns]
        Xc = X[use_feats].copy()
        # категории -> строковый тип
        for c in self.cat_cols:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype(str)
        cat_present = [Xc.columns.get_loc(c) for c in self.cat_cols if c in Xc.columns]
        # ВАЖНО: веса передаём ВОВНУТРЬ Pool (weight=sample_weight),
        # чтобы не было ошибки "sample_weight must be None when X is Pool"
        return Pool(Xc, label=y, weight=sample_weight, cat_features=cat_present)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        for a in [0.1, 0.5, 0.9]:
            m = CatBoostRegressor(loss_function=f'Quantile:alpha={a}', **self.params)
            pool = self._make_pool(X, y, sample_weight=sample_weight)
            m.fit(pool, verbose=False)   # <- НЕ передаём sample_weight отдельно!
            self.m[a] = m
        self._feature_names_used = [c for c in self.feature_names if c in X.columns]
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        use_feats = [c for c in getattr(self, '_feature_names_used', self.feature_names) if c in X.columns]
        Xc = X[use_feats].copy()
        for c in self.cat_cols:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype(str)
        pool = Pool(Xc, cat_features=[Xc.columns.get_loc(c) for c in self.cat_cols if c in Xc.columns])
        out = pd.DataFrame(index=X.index)
        for a in [0.1, 0.5, 0.9]:
            out[f'p{int(a*100)}'] = self.m[a].predict(pool)
        return out
