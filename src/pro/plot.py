import argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def main(history_csv: str, forecast_csv: str, outdir: str='outputs'):
    hist = pd.read_csv(history_csv); hist['date'] = pd.to_datetime(hist['date'])
    fc = pd.read_csv(forecast_csv);   fc['date'] = pd.to_datetime(fc['date'])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for sku, g in hist.groupby('sku'):
        f = fc[fc['sku']==sku]
        if f.empty: continue
        plt.figure(figsize=(9,4.5))
        plt.plot(g['date'], g['qty'], label='История (qty)')
        plt.plot(f['date'], f['p50'], label='Прогноз p50')
        if 'p10' in f.columns and 'p90' in f.columns:
            plt.fill_between(f['date'].values, f['p10'].values, f['p90'].values, alpha=0.2, label='[p10;p90]')
        plt.title(f'SKU {sku}: история и прогноз')
        plt.xlabel('Дата'); plt.ylabel('Количество'); plt.legend(); plt.tight_layout()
        out_path = Path(outdir)/f'plot_{sku}.png'
        plt.savefig(out_path, dpi=160); plt.close()
        print(f'Saved plot → {out_path}')

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', required=True)
    ap.add_argument('--forecast', required=True)
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()
    main(args.history, args.forecast, args.outdir)
