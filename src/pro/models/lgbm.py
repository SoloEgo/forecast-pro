import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor, log_evaluation

class _ConstModel:
    """Простой бэкап: предсказывает константную квантиль по трейну."""
    def __init__(self, value: float):
        self.value = float(value)
    def predict(self, X: pd.DataFrame):
        return np.full(len(X), self.value, dtype=float)

class LGBMQuantile:
    """Квантильный бустинг (LightGBM) с автоподстройкой гиперов и бэкапом на крошечных окнах."""
    def __init__(
        self,
        n_estimators: int = 800,
        num_leaves: int = 64,
        learning_rate: float = 0.05,
        min_data_in_leaf: int = 50,
        feature_names: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ):
        self.m = {}
        self.feature_names = feature_names or []
        self.cat_cols = cat_cols or ['sku']
        self.base_params = dict(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_data_in_leaf=min_data_in_leaf,
            random_state=42,
            verbosity=-1,          # тише
            force_col_wise=True,   # убираем подсказки
        )

    def _make_preprocessor(self, X: pd.DataFrame):
        feat_present = [c for c in self.feature_names if c in X.columns]
        cats_present = [c for c in self.cat_cols if c in feat_present]
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        if cats_present:
            prep = ColumnTransformer([('ohe', ohe, cats_present)], remainder='passthrough')
        else:
            from sklearn.preprocessing import FunctionTransformer
            prep = FunctionTransformer(lambda x: x, feature_names_out='one-to-one')
        return prep, feat_present

    def _adaptive_params(self, n_train: int):
        """Смягчаем параметры под маленькое окно."""
        p = self.base_params.copy()
        p['num_leaves'] = max(2, min(p.get('num_leaves', 64), n_train - 1 if n_train > 2 else 2))
        p['min_data_in_leaf'] = max(1, min(p.get('min_data_in_leaf', 20), max(1, n_train // 10)))
        p['min_data_in_bin'] = max(1, n_train // 10)
        return p

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        prep, feat_present = self._make_preprocessor(X)
        n_train = len(y)
        params = self._adaptive_params(n_train)

        for a in [0.1, 0.5, 0.9]:
            const_fallback = _ConstModel(np.quantile(y, a))
            model = LGBMRegressor(objective='quantile', alpha=a, **params)
            pipe = Pipeline(steps=[('prep', prep), ('m', model)])
            try:
                pipe.fit(
                    X[feat_present], y,
                    m__sample_weight=sample_weight,
                    m__callbacks=[log_evaluation(period=0)]
                )
                self.m[a] = pipe
            except Exception:
                self.m[a] = const_fallback
        self._feat_present = feat_present
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        feat_present = [c for c in getattr(self, '_feat_present', self.feature_names) if c in X.columns]
        out = pd.DataFrame(index=X.index)
        for a in [0.1, 0.5, 0.9]:
            m = self.m[a]
            if isinstance(m, Pipeline):
                out[f'p{int(a*100)}'] = m.predict(X[feat_present])
            else:
                out[f'p{int(a*100)}'] = m.predict(X)
        return out
