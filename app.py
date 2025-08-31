# app.py
import io
import os
from pathlib import Path
import traceback
from typing import Optional
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
# ====== наши модули ======
from src.pro.optimize import main as optimize_main
from src.pro.smart_train import main as smart_train_main
from src.pro.forecast import main as forecast_main
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    _rerun = st.rerun            # новые версии (>=1.30)
except AttributeError:
    _rerun = st.experimental_rerun  # старые версии

def rerun():
    _rerun()
err_box = st.empty()


# ====== логотипы ======
LOGO_BIG = "assets/logo_forecastpro_transparent.png"
LOGO_ICO = "assets/logo_forecastpro_128.png"

# ====== set_page_config (один раз) ======
page_icon = "📈"
try:
    if Path(LOGO_ICO).exists():
        page_icon = Image.open(LOGO_ICO)
except Exception:
    pass
st.set_page_config(page_title="ForecastPro", page_icon=page_icon, layout="wide")

# ====== CSS: блюр + оверлей ======
st.markdown("""
<style>
[data-testid="stSidebarUserContent"]{
margin-top: -3rem;
}
[data-testid="stStatusWidget"], .stToolbarActions{
    display: none;
}
header.stAppHeader {
    background: none;
}
.stMainBlockContainer.block-container {
    padding: 0.5rem 2rem;
}
.stAppDeployButton { display: none !important; }
.blurred {
  filter: blur(5px) grayscale(12%);
  transition: filter .2s ease-in-out;
  pointer-events: none;
}
.app-overlay{
  position: fixed; inset: 0;
  background: rgba(0,0,0,.28);
  z-index: 10000;
  display:flex; align-items:center; justify-content:center;
  backdrop-filter: blur(2px);
}
.app-overlay .card{
  background: #111; color: #fff;
  padding: 14px 18px; border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0,0,0,.3);
  text-align: center; min-width: 280px; max-width: 80vw;
  font-size: 15px;
}
.spinner {
  width: 46px; height: 46px;
  border: 4px solid rgba(255,255,255,0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 10px auto;
}

.st-key-eta_ui { display: none !important; }

@keyframes spin { to { transform: rotate(360deg); } }


</style>
""", unsafe_allow_html=True)

# ====== session_state ======
ss = st.session_state
def _init(k, v):
    if k not in ss: ss[k] = v

_init("is_running", False)
_init("busy_msg", "")
_init("step", None)               # 'optimize' -> 'train' -> 'forecast' -> None
_init("params", {})               # сюда кладём все параметры запуска
_init("input_path", None)
_init("promo_path", None)
_init("hist_df", None)
_init("forecast_df", None)
_init("freq", None)
_init("sku_sel", None)

# ====== Header ======
col1, col2 = st.columns([1, 12])
with col1:
    if Path(LOGO_BIG).exists():
        st.image(LOGO_BIG)
    else:
        st.markdown("<div style='font-size:48px'>📈</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin:0;margin-left: -2rem;'>orecastPro</h1>", unsafe_allow_html=True)

# ====== Рабочие директории ======
WORKDIR = Path("ui_runtime"); WORKDIR.mkdir(exist_ok=True, parents=True)
OUTDIR = "outputs"

# ====== helpers ======
def save_uploaded_file(uploaded, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return target_path

def run_optimize(input_csv: str, algo: str, freq: str, horizon: int, cv: int, trials: int):
    optimize_main(input_csv, algo, freq, horizon, cv, trials, OUTDIR)

def run_train(input_csv: str, freq: str, horizon: int, cv: int, eta: float,
              lookback_months: Optional[int], halflife: Optional[int]):
    smart_train_main(input_csv, freq, horizon, cv, eta, OUTDIR, lookback_months, halflife)

def run_forecast(input_csv: str, model_path: str, freq: str, horizon: int,
                 joint: bool, calibrate_last_k: int, cal_clip_low: float, cal_clip_high: float,
                 promo_scenario: Optional[str]):
    forecast_main(input_csv, model_path, horizon, OUTDIR, promo_scenario, freq,
                  joint, calibrate_last_k, cal_clip_low, cal_clip_high)

def load_forecast_csv(freq: str) -> pd.DataFrame:
    f = Path(OUTDIR) / f"forecast_{freq}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Не найден {f}")
    return pd.read_csv(f)

def plot_sku_interactive(history: pd.DataFrame,
                         forecast: pd.DataFrame,
                         sku: str,
                         show_band: bool = True,
                         show_points: bool = True,
                         show_promo: bool = True):
    # фильтрация и даты
    h = history[history["sku"] == sku].copy()
    f = forecast[forecast["sku"] == sku].copy()
    if h.empty and f.empty:
        st.info("Нет данных для выбранного SKU.")
        return
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    h.sort_values("date", inplace=True)
    f.sort_values("date", inplace=True)

    fig = go.Figure()

    # История
    fig.add_trace(go.Scatter(
        x=h["date"], y=h["qty"],
        mode="lines+markers" if show_points else "lines",
        name="История (qty)",
        hovertemplate="Дата: %{x|%Y-%m-%d}<br>Продажи: %{y:,}<extra></extra>"
    ))

    # Прогноз p50
    if "p50" in f.columns and len(f):
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["p50"],
            mode="lines",
            name="Прогноз p50",
            line=dict(dash="dash"),
            hovertemplate="Дата: %{x|%Y-%m-%d}<br>p50: %{y:,}<extra></extra>"
        ))

    # Диапазон p10–p90 как заливка
    if show_band and {"p10", "p90"}.issubset(f.columns) and len(f):
        # верхняя граница
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["p90"],
            mode="lines",
            line=dict(width=0),
            name="p90",
            hoverinfo="skip",
            showlegend=False
        ))
        # нижняя граница с fill='tonexty'
        fig.add_trace(go.Scatter(
            x=f["date"], y=f["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Диапазон p10–p90",
            hovertemplate="Дата: %{x|%Y-%m-%d}<br>p10: %{y:,}<extra></extra>"
        ))

    # Подсветка промо (если есть колонка)
    if show_promo and "promo_flag" in h.columns and h["promo_flag"].sum() > 0:
        # аккуратно подсветим вертикальными полупрозрачными прямоугольниками
        shapes = []
        for dt in h.loc[h["promo_flag"] == 1, "date"]:
            # на месячной частоте подсветим весь месяц,
            # на дневной – один день
            if ss.freq == "monthly":
                start = pd.to_datetime(dt).replace(day=1)
                # конец месяца:
                end = (start + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1)
            else:  # daily
                start = pd.to_datetime(dt)
                end = start + pd.Timedelta(days=1)
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=start, x1=end, y0=0, y1=1,
                fillcolor="rgba(255,165,0,0.15)", line=dict(width=0)
            ))
        fig.update_layout(shapes=shapes)

    # Оси, легенда, слайдер, кнопки выбора диапазона
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"SKU: {sku}",
        xaxis=dict(
            title="Дата",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        ),
        yaxis=dict(title="Продажи"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Настройки тулбара: без логотипа, с "скачать PNG"
    config = dict(displaylogo=False, toImageButtonOptions=dict(filename=f"forecast_{sku}"))

    st.plotly_chart(fig, use_container_width=True, config=config)


# ====== Sidebar ======
# ------------- Sidebar -------------
st.sidebar.title("⚙️ Параметры")
#st.sidebar.header("⚙️ Параметры")

uploaded_csv = st.sidebar.file_uploader(
    "Загрузить исходный CSV (monthly/daily)",
    type=["csv"],
    help=(
        "CSV с колонками: date (YYYY-MM-DD), sku, qty, promo_flag (0/1), "
        "category (опц.). Частота выбирается ниже."
    ),
)

freq = st.sidebar.selectbox(
    "Частота",
    ["monthly", "daily"],
    index=0,
    help=(
        "Гранулярность данных и прогноза:\n"
        "• monthly — шаг 1 месяц (ожидается одна запись на SKU/месяц)\n"
        "• daily — шаг 1 день (ожидается одна запись на SKU/день)"
    ),
)

model_algo = st.sidebar.selectbox(
    "Модель для Optimize",
    ["lgbm", "catboost"],
    index=0,
    help=(
        "Какой алгоритм настраивать на этапе HPO (подбора гиперпараметров).\n"
        "Ансамбль обучится после HPO; этот выбор влияет именно на то, "
        "какую модель оптимизируем (LightGBM или CatBoost)."
    ),
)

do_optimize = st.sidebar.checkbox(
    "Запустить Optimize (HPO)",
    value=True,
    help=(
        "Включить этап Hyper-Parameter Optimization (Optuna): перебор гиперпараметров "
        "с кросс-валидацией по времени.\n"
        "Даёт шанс заметно улучшить качество, но увеличивает время расчёта."
    ),
)

trials = st.sidebar.number_input(
    "HPO trials",
    min_value=10, max_value=500, value=60, step=10,
    help=(
        "Сколько попыток (трейалов) выполнить при подборе гиперпараметров.\n"
        "Больше — дольше, но выше шанс найти лучшее качество.\n"
        "Ориентиры: 30–80 для небольших данных; 100–200 для серьёзного поиска."
    ),
)

st.sidebar.markdown("---")

horizon = st.sidebar.number_input(
    "Horizon",
    min_value=1, max_value=60, value=12,
    help=(
        "Горизонт прогноза в периодах (месяцах/днях — по выбранной частоте).\n"
        "Чем длиннее горизонт, тем выше неопределённость и ниже точность."
    ),
)

cv = st.sidebar.number_input(
    "CV folds",
    min_value=2, max_value=10, value=3,
    help=(
        "Сколько раз проверяем модель, сдвигая окно вперёд по времени.\n"
        "На каждом шаге учимся на прошлом и прогнозируем ровно на Horizon периодов.\n"
        "Больше фолдов — надёжнее оценка, но дольше расчёт.\n"
        "Нужно ≈ (CV folds + 1) × Horizon периодов истории."
    ),
)

st.sidebar.markdown('<span id="eta_anchor"></span>', unsafe_allow_html=True)
eta = st.sidebar.number_input(
    "Ensemble η (жесткость штрафа)",
    min_value=0.1, max_value=20.0, value=5.0, step=0.1,
    help=(
        "Параметр для взвешивания экспертов ансамбля: чем больше η, тем сильнее "
        "штрафуются модели, хуже работавшие на недавних периодах.\n"
        "⚠️ В текущей сборке ансамбль усредняет равными весами, поэтому η не влияет. "
        "Оставьте по умолчанию; будет задействован в версии со взвешиванием."
    ),
    key="eta_ui",
)

lookback_months_val = st.sidebar.number_input(
    "Lookback (мес, ~дни для daily)",
    min_value=0, max_value=240, value=60,
    help=(
        "Использовать для обучения только последние N периодов (месяцев/дней). "
        "0 — брать всю историю.\n"
        "Полезно, если старые годы уже не репрезентативны: уменьшает влияние давних данных."
    ),
)

halflife_val = st.sidebar.number_input(
    "Halflife (мес/дни)",
    min_value=0, max_value=120, value=18,
    help=(
        "Период полураспада для экспоненциального затухания весов обучающих точек. "
        "Каждые Halflife периодов вклад снижается в 2 раза.\n"
        "0 — отключить. Типичные значения: 12–24 мес (monthly) или 30–90 дней (daily)."
    ),
)

st.sidebar.markdown("---")

joint = st.sidebar.checkbox(
    "Совместный прогноз по всем SKU",
    value=True,
    help=(
        "Одна модель обучается сразу на всех товарах. "
        "Она видит идентификатор SKU и категорию, поэтому может перенимать "
        "паттерны между похожими товарами (это помогает редким SKU).\n\n"
        "Если выключить — для каждого SKU строится отдельная модель.\n\n"
        "Плюсы: редкие SKU начинают прогнозироваться лучше. "
        "Минусы: выше требования к качеству признаков и разметки — "
        "ошибки в одном товаре могут немного влиять на другие."
    ),
)

calibrate_last_k = st.sidebar.number_input(
    "Калибровка: последние k точек",
    min_value=0, max_value=24, value=6,
    help=(
        "Пост-калибровка p50: масштабируем уровень прогноза, чтобы p50 совпадал со средним "
        "последних k фактов. Устраняет «level shift».\n"
        "0 — отключить. Ориентиры: 3–6 для monthly; 7–28 для daily."
    ),
)

cal_clip_low = st.sidebar.number_input(
    "Calibrate clip low",
    min_value=0.1, max_value=2.0, value=0.6, step=0.05,
    help=(
        "Нижняя граница *коэффициента* масштабирования при калибровке p50.\n"
        "По умолчанию 0.60 = максимум −40% к уровню прогноза.\n\n"
        "Зачем: не дать калибровке чрезмерно занизить прогноз, если последние k периодов были аномально низкими.\n"
        "Рекомендации: daily/волатильные — можно опустить до 0.5; стабильные ряды — поднять до 0.8."
    ),
)

cal_clip_high = st.sidebar.number_input(
    "Calibrate clip high",
    min_value=0.1, max_value=3.0, value=1.4, step=0.05,
    help=(
        "Верхняя граница *коэффициента* масштабирования при калибровке p50.\n"
        "По умолчанию 1.40 = максимум +40% к уровню прогноза.\n\n"
        "Зачем: ограничить завышение прогноза, если последние k периодов были всплеском.\n"
        "Рекомендации: daily/волатильные — можно поднять до 1.6; стабильные — снизить до 1.2–1.3."
    ),
)

promo_file = st.sidebar.file_uploader(
    "План промо (опц.) CSV",
    type=["csv"],
    help=(
        "Необязательный файл с будущими промо-периодами для сценарного прогноза. "
        "Ожидаемые колонки: date, sku (или ALL), promo_flag=1."
    ),
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button(
    "🚀 Запустить Forecast",
    help="Выполнить подряд все этапы: подбор гиперпараметров (если включён), обучение ансамбля и расчёт прогноза."
)


# ====== Оверлей (если идёт шаг) ======
if ss.is_running:
    st.markdown(f"""
    <div class="app-overlay">
      <div class="card">
        <div class="spinner"></div>
        <div>{ss.busy_msg}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ====== Обработка клика: ставим шаг и делаем rerun ======
if run_btn:
    try:
        # Сохраняем/обновляем входной CSV
        if uploaded_csv is not None:
            input_path = save_uploaded_file(uploaded_csv, WORKDIR / "data.csv")
            ss.input_path = str(input_path)
        elif ss.input_path:
            input_path = Path(ss.input_path)
            if not input_path.exists():
                st.warning("Ранее загруженный CSV не найден, загрузите файл заново.")
                st.stop()
        else:
            st.warning("Загрузите исходный CSV.")
            st.stop()

        # Нормализуем колонки
        df_hist = pd.read_csv(ss.input_path)
        if "sku" not in df_hist.columns: df_hist["sku"] = "ALL"
        if "qty" not in df_hist.columns: raise ValueError("В CSV должна быть колонка 'qty'.")
        if "date" not in df_hist.columns: raise ValueError("В CSV должна быть колонка 'date' (YYYY-MM-DD).")
        if "promo_flag" not in df_hist.columns: df_hist["promo_flag"] = 0
        if "category" not in df_hist.columns: df_hist["category"] = "ALL"
        df_hist.to_csv(ss.input_path, index=False)

        # Промо-план (если есть)
        if promo_file is not None:
            ss.promo_path = str(save_uploaded_file(promo_file, WORKDIR / "promo.csv"))

        # Сохраняем все параметры запуска в ss.params
        ss.params = dict(
            freq=freq, model_algo=model_algo, do_optimize=bool(do_optimize),
            trials=int(trials), horizon=int(horizon), cv=int(cv), eta=float(eta),
            lookback=int(lookback_months_val) if lookback_months_val > 0 else None,
            halflife=int(halflife_val) if halflife_val > 0 else None,
            joint=bool(joint), calibrate_last_k=int(calibrate_last_k),
            cal_clip_low=float(cal_clip_low), cal_clip_high=float(cal_clip_high),
        )

        # Устанавливаем первый шаг и перерисовываем
        ss.is_running = True
        if ss.params["do_optimize"]:
            ss.step = "optimize"; ss.busy_msg = f"Запуск HPO ({ss.params['model_algo']})…"
        else:
            ss.step = "train"; ss.busy_msg = "Обучение ансамбля…"
        st.rerun()

    except Exception:
        err_box.error("Ошибка на этапе подготовки:\n\n" + "".join(traceback.format_exc()))

# ====== Машина состояний: выполняем шаги между перерисовками ======
try:
    if ss.is_running and ss.step:
        p = ss.params
        if ss.step == "optimize":
            run_optimize(ss.input_path, p["model_algo"], p["freq"], p["horizon"], p["cv"], p["trials"])
            ss.step = "train"; ss.busy_msg = "Обучение ансамбля…"
            st.rerun()

        elif ss.step == "train":
            run_train(ss.input_path, p["freq"], p["horizon"], p["cv"], p["eta"], p["lookback"], p["halflife"])
            ss.step = "forecast"; ss.busy_msg = "Расчёт прогноза…"
            st.rerun()

        elif ss.step == "forecast":
            model_path = str(Path(OUTDIR) / f"model_ensemble_{p['freq']}.joblib")
            run_forecast(ss.input_path, model_path, p["freq"], p["horizon"],
                         p["joint"], p["calibrate_last_k"], p["cal_clip_low"], p["cal_clip_high"],
                         ss.promo_path)

            # Результат → session_state
            ss.forecast_df = load_forecast_csv(p["freq"])
            ss.hist_df = pd.read_csv(ss.input_path)
            ss.freq = p["freq"]

            # SKU по умолчанию
            if ss.hist_df is not None:
                skus = sorted(ss.hist_df["sku"].astype(str).unique().tolist())
                if ss.sku_sel not in skus:
                    ss.sku_sel = skus[0] if skus else None

            # Завершили
            ss.is_running = False
            ss.busy_msg = ""
            ss.step = None
            st.rerun()
except Exception:
    ss.is_running = False
    ss.busy_msg = ""
    ss.step = None
    err_box.error("Ошибка во время выполнения шага:\n\n" + "".join(traceback.format_exc()))

# ====== Отрисовка результатов (блюрим, если идёт шаг) ======
st.markdown(f'<div class="{ "blurred" if ss.is_running else "" }">', unsafe_allow_html=True)

if ss.forecast_df is not None and ss.hist_df is not None:
    st.subheader("📄 Прогноз (первые 200 строк)")
    st.dataframe(ss.forecast_df.head(200))

    csv_bytes = ss.forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Скачать forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{ss.freq or 'monthly'}.csv",
        mime="text/csv",
        key="download_forecast"
    )

    skus = sorted(ss.hist_df["sku"].astype(str).unique().tolist())
    default_index = skus.index(ss.sku_sel) if (ss.sku_sel in skus) else 0
    ss.sku_sel = st.selectbox("SKU для графика", skus, index=default_index, key="sku_select")

    df_h = ss.hist_df.copy(); df_f = ss.forecast_df.copy()
    df_h["date"] = pd.to_datetime(df_h["date"], errors="coerce")
    df_f["date"] = pd.to_datetime(df_f["date"], errors="coerce")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_band = st.checkbox("Показать p10–p90", value=True, key="show_band")
    with c2:
        show_points = st.checkbox("Точки на истории", value=True, key="show_points")
    with c3:
        show_promo = st.checkbox("Подсветка промо", value=("promo_flag" in df_h.columns), key="show_promo")

    if ss.sku_sel is not None:
        plot_sku_interactive(df_h, df_f, ss.sku_sel,
                             show_band=show_band,
                             show_points=show_points,
                             show_promo=show_promo)

st.markdown('</div>', unsafe_allow_html=True)
