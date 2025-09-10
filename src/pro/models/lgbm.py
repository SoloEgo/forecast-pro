# src/pro/models/lgbm.py
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

class LGBMQuantile:
    """
    Три отдельные модели LightGBM под p10/p50/p90.
    - Параметры из Optuna пробрасываются в каждый регрессор.
    - Признаки берутся только из feature_names.
    - object-колонки -> pandas 'category' (и передаются как categorical_feature).
    - fit поддерживает sample_weight.
    - predict_quantiles возвращает DataFrame с колонками ['p10','p50','p90'].
    - Встроенная страховка: неотрицательность и монотонность квантилей.
    """

    def __init__(self, feature_names, **params):
        self.feature_names = list(feature_names)
        self.params = dict(params)
        self.m_10 = None
        self.m_50 = None
        self.m_90 = None
        self.categorical_cols: list[str] = []

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        Xsub = X[self.feature_names].copy()
        self.categorical_cols = []
        for c in Xsub.columns:
            s = Xsub[c]
            if s.dtype == object:
                Xsub[c] = s.astype('category')
                self.categorical_cols.append(c)
            elif is_categorical_dtype(s):
                self.categorical_cols.append(c)
            elif not is_numeric_dtype(s):
                Xsub[c] = pd.to_numeric(s, errors='coerce').astype(float).fillna(0.0)
        return Xsub

    def _make(self, alpha: float):
        # молчим по умолчанию
        params = dict(objective='quantile', alpha=alpha, **self.params)
        params.setdefault('verbosity', -1)
        return lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        Xsub = self._prepare_X(X)
        self.m_10 = self._make(0.1).fit(
            Xsub, y, sample_weight=sample_weight,
            categorical_feature=self.categorical_cols or 'auto'
        )
        self.m_50 = self._make(0.5).fit(
            Xsub, y, sample_weight=sample_weight,
            categorical_feature=self.categorical_cols or 'auto'
        )
        self.m_90 = self._make(0.9).fit(
            Xsub, y, sample_weight=sample_weight,
            categorical_feature=self.categorical_cols or 'auto'
        )
        return self

    def _prepare_X_infer(self, X: pd.DataFrame) -> pd.DataFrame:
        Xsub = X[self.feature_names].copy()
        # те же категории на инференсе
        for c in self.categorical_cols:
            if c in Xsub.columns:
                Xsub[c] = Xsub[c].astype('category')
        # и защита от внезапных нечисловых типов
        for c in Xsub.columns:
            s = Xsub[c]
            if (c not in self.categorical_cols) and (not is_numeric_dtype(s)):
                Xsub[c] = pd.to_numeric(s, errors='coerce').astype(float).fillna(0.0)
        return Xsub

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> pd.DataFrame:
        Xsub = self._prepare_X_infer(X)
        p10 = np.asarray(self.m_10.predict(Xsub))
        p50 = np.asarray(self.m_50.predict(Xsub))
        p90 = np.asarray(self.m_90.predict(Xsub))
        # построчная монотонность + неотрицательность
        arr = np.vstack([p10, p50, p90]).T.astype(float)  # (n, 3)
        arr[arr < 0.0] = 0.0
        arr = np.sort(arr, axis=1)
        return pd.DataFrame(arr, columns=["p10", "p50", "p90"])

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.predict_quantiles(X)
