# Forecast SMART (Monthly & Daily) — HPO + Ensemble

**Что внутри**
- Частота: `--freq monthly|daily`
- Квантильные модели: LightGBM, CatBoost
- Бенчмарк-эксперт: Seasonal Naive (lag 12/7) с квантилями по остаткам
- Walk-forward бэктест (rolling) с wMAPE / sMAPE / MASE / Pinball (p10/p50/p90)
- Байесовская оптимизация гиперпараметров (Optuna) по Pinball(p50)
- Ансамбль экспертов с экспоненциальными весами по walk-forward лоссам
- Прогноз p10/p50/p90, сценарии промо, графики

## Установка
```bash
python3 -m venv .venv && source .venv/bin/activate
# macOS (Apple Silicon) для LightGBM:
#   brew install libomp
pip install -U pip
pip install -r requirements.txt
```

## Данные
- **Месячные:** CSV `date, sku, qty[, promo_flag]` (date конвертируется к концу месяца).
- **Дневные:** CSV `date, sku, qty[, promo_flag]` (по дням).
- Агрегировать дни → месяцы:
  ```bash
  python -m src.pro.prepare_monthly --input data/sales_daily.csv --out data/sales_monthly.csv
  ```

## 1) Подбор гиперпараметров (Optuna)
```bash
python -m src.pro.optimize --input data/sales_monthly.csv --algo lgbm --freq monthly --horizon 12 --cv 4 --trials 60
python -m src.pro.optimize --input data/sales.csv --algo catboost --freq daily --horizon 14 --cv 6 --trials 60
```
Результат: `outputs/best_params_<algo>_<freq>.json` и `model_<algo>_<freq>_opt.joblib`.

## 2) Обучение «умного» ансамбля
```bash
python -m src.pro.smart_train --input data/sales_monthly.csv --freq monthly --horizon 12 --cv 4 --eta 5.0
```
Результат: `outputs/model_ensemble_<freq>.joblib` и `outputs/ensemble_report_<freq>.json` (веса экспертов).

## 3) Прогноз и графики
```bash
python -m src.pro.forecast --input data/sales_monthly.csv --model_path outputs/model_ensemble_monthly.joblib --freq monthly --horizon 12
python -m src.pro.plot --history data/sales_monthly.csv --forecast outputs/forecast_monthly.csv
```

## Советы
- Дневные горизонты: 7–14.
- Месячные: лучше 36+ мес истории.
- Если хочешь сильнее учитывать недавние точки — можно добавить time-decay веса (см. комментарии в smart_train.py).
