import argparse, numpy as np, pandas as pd
from pathlib import Path

def winsorize_per_sku_day(df, sku_col, qty_col, p=0.99):
    caps = (df.groupby(sku_col)[qty_col].quantile(p).rename('cap')).reset_index()
    df = df.merge(caps, on=sku_col, how='left')
    df[qty_col] = np.minimum(df[qty_col].values, df['cap'].values)
    return df.drop(columns=['cap'])

def main(inp, out):
    df = pd.read_csv(inp)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    if 'sku' not in df.columns: df['sku'] = 'ALL'
    if 'promo_flag' not in df.columns: df['promo_flag'] = 0

    df = winsorize_per_sku_day(df, 'sku', 'qty', p=0.99)

    df['month'] = df['date'].dt.to_period('M')
    agg = (df.groupby(['sku','month'], as_index=False)
             .agg(qty=('qty','sum'),
                  sale_days=('qty', lambda s: (s>0).sum()),
                  promo_days=('promo_flag','sum')))
    agg['promo_share'] = (agg['promo_days'] /
                          agg['month'].dt.to_timestamp('M').dt.daysinmonth).clip(0,1)
    agg['date'] = agg['month'].dt.to_timestamp('M')
    out_df = agg[['date','sku','qty','promo_share','sale_days']].sort_values(['sku','date'])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Saved → {out} ; months={out_df['date'].dt.to_period('M').nunique()} ; skus={out_df['sku'].nunique()}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default='data/sales_monthly.csv')
    args = ap.parse_args()
    main(args.input, args.out)
