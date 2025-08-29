import numpy as np, pandas as pd

# MONTHLY
NUMERIC_FEATURES = [
    'promo_flag','month_sin','month_cos',
    'y_lag1','y_lag3','y_lag6','y_lag12',
    'mean_3m','mean_6m','mean_12m'
]
CAT_FEATURES = ['sku']
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

def add_features(df: pd.DataFrame, id_col='sku', date_col='date', y_col='qty') -> pd.DataFrame:
    f = df.copy().sort_values([id_col, date_col])
    f[date_col] = pd.to_datetime(f[date_col]).dt.to_period('M').dt.to_timestamp('M')
    f['month'] = f[date_col].dt.month
    f['month_sin'] = np.sin(2*np.pi*f['month']/12)
    f['month_cos'] = np.cos(2*np.pi*f['month']/12)
    for L in [1,3,6,12]:
        f[f'y_lag{L}'] = f.groupby(id_col, group_keys=False)[y_col].shift(L)
    for W in [3,6,12]:
        f[f'mean_{W}m'] = f.groupby(id_col, group_keys=False)[y_col].apply(
            lambda s: s.shift(1).rolling(W, min_periods=1).mean())
    if 'promo_flag' not in f.columns:
        f['promo_flag'] = 0
    return f

# DAILY
DAILY_NUMERIC = [
    'promo_flag','dow_sin','dow_cos','month_sin','month_cos',
    'is_weekend','is_month_end',
    'y_lag1','y_lag2','y_lag3','y_lag7','y_lag14','y_lag28',
    'mean_7d','mean_14d','mean_28d'
]
DAILY_CATS = ['sku']
DAILY_ALL = DAILY_NUMERIC + DAILY_CATS

def add_features_daily(df: pd.DataFrame, id_col='sku', date_col='date', y_col='qty') -> pd.DataFrame:
    f = df.copy().sort_values([id_col, date_col])
    f[date_col] = pd.to_datetime(f[date_col])
    f['dow'] = f[date_col].dt.weekday
    f['dow_sin'] = np.sin(2*np.pi*f['dow']/7)
    f['dow_cos'] = np.cos(2*np.pi*f['dow']/7)
    f['month'] = f[date_col].dt.month
    f['month_sin'] = np.sin(2*np.pi*f['month']/12)
    f['month_cos'] = np.cos(2*np.pi*f['month']/12)
    f['is_weekend'] = (f['dow']>=5).astype(int)
    f['is_month_end'] = f[date_col].dt.is_month_end.astype(int)
    for L in [1,2,3,7,14,28]:
        f[f'y_lag{L}'] = f.groupby(id_col, group_keys=False)[y_col].shift(L)
    for W, name in [(7,'mean_7d'),(14,'mean_14d'),(28,'mean_28d')]:
        f[name] = f.groupby(id_col, group_keys=False)[y_col].apply(
            lambda s: s.shift(1).rolling(W, min_periods=1).mean())
    if 'promo_flag' not in f.columns:
        f['promo_flag'] = 0
    return f
