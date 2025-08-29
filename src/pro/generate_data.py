import numpy as np, pandas as pd
from pathlib import Path

def main(out: str = 'data/sales_monthly.csv'):
    rng = np.random.default_rng(7)
    dates = pd.date_range('2019-01-31','2024-12-31',freq='ME')
    skus = ['A','B','C']
    recs = []
    for sku in skus:
        base = {'A':90,'B':35,'C':60}[sku]
        for i,d in enumerate(dates):
            m = d.month
            seas = 12*np.sin(2*np.pi*m/12) + 6*np.cos(2*np.pi*m/6)
            trend = (i-36)*0.3 if sku!='B' else (i-18)*0.5
            promo = 1 if (sku=='B' and m in [11,12]) or (sku=='A' and m in [3,9]) else 0
            boost = 10 if promo else 0
            qty = max(0.0, base + trend + seas + boost + rng.normal(0,5))
            recs.append({'date':d,'sku':sku,'qty':round(float(qty),2),'promo_flag':promo})
    df = pd.DataFrame.from_records(recs).sort_values(['sku','date']).reset_index(drop=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out,index=False)
    print(f"Saved: {out} rows={len(df)}")

if __name__=='__main__':
    main()
