import pandas as pd, numpy as np
class SeasonalNaiveQuantile:
    """Сезонный наивный прогноз + квантили по остаткам (lag12 для monthly, lag7 для daily)."""
    def __init__(self, freq: str = 'monthly'):
        self.freq = freq
        self.q10 = {}; self.q90 = {}

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        lag_col = 'y_lag12' if self.freq == 'monthly' else 'y_lag7'
        df = X.copy(); df['y'] = y
        if lag_col not in df.columns:
            raise ValueError(f"Не найден {lag_col} в признаках.")
        df = df.dropna(subset=[lag_col, 'y'])
        for sku, g in df.groupby('sku'):
            r = (g['y'] - g[lag_col]).values
            if len(r) < 5:
                self.q10[sku] = np.quantile(r, 0.1) if len(r)>0 else 0.0
                self.q90[sku] = np.quantile(r, 0.9) if len(r)>0 else 0.0
            else:
                self.q10[sku] = float(np.quantile(r, 0.1))
                self.q90[sku] = float(np.quantile(r, 0.9))
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        lag_col = 'y_lag12' if self.freq == 'monthly' else 'y_lag7'
        base = X[lag_col].fillna(0.0).astype(float)
        sku = X['sku'].astype(str)
        p50 = base.values
        p10 = p50 + sku.map(self.q10).fillna(0.0).values
        p90 = p50 + sku.map(self.q90).fillna(0.0).values
        return pd.DataFrame({'p10': p10, 'p50': p50, 'p90': p90}, index=X.index)
