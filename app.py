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
import plotly.graph_objects as go
from base64 import b64encode
import streamlit.components.v1 as components

# ====== наши модули ======
from src.pro.optimize import main as optimize_main
from src.pro.smart_train import main as smart_train_main
from src.pro.forecast import main as forecast_main

# --- rerun совместимость
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

def rerun():
    _rerun()

err_box = st.empty()

# ====== логотипы ======
LOGO_BIG = "assets/logo_forecastpro_transparent.png"
LOGO_ICO = "assets/logo_forecastpro_128.png"

# ====== set_page_config ======
page_icon = "📈"
try:
    if Path(LOGO_ICO).exists():
        page_icon = Image.open(LOGO_ICO)
except Exception:
    pass
st.set_page_config(page_title="ForecastPro", page_icon=page_icon, layout="wide")

# ====== CSS ======
st.markdown("""
<style>
.stHorizontalBlock { justify-content: start !important; flex-wrap: nowrap; }
.stHorizontalBlock>div { min-width: fit-content; }

[data-testid="stSidebarUserContent"]{ margin-top: -3rem; }
[data-testid="stStatusWidget"], .stToolbarActions{ display: none; }
header.stAppHeader { background: none; }
.stMainBlockContainer.block-container { padding: 0.5rem 2rem; }
.stAppDeployButton { display: none !important; }

.blurred { filter: blur(5px) grayscale(12%); transition: filter .2s ease-in-out; pointer-events: none; }

.app-overlay{ position: fixed; inset: 0; background: rgba(0,0,0,.28); z-index: 10000;
              display:flex; align-items:center; justify-content:center; backdrop-filter: blur(2px); }
.app-overlay .card{ background: #111; color: #fff; padding: 14px 18px; border-radius: 12px;
                    box-shadow: 0 8px 30px rgba(0,0,0,.3); text-align: center; min-width: 280px; max-width: 80vw; font-size: 15px; }
.spinner { width: 46px; height: 46px; border: 4px solid rgba(255,255,255,0.35); border-top-color: #fff;
           border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 10px auto; }
@keyframes spin { to { transform: rotate(360deg); } }

.st-key-eta_ui { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ====== session_state ======
ss = st.session_state
def _init(k, v):
    if k not in ss: ss[k] = v

_init("is_running", False)
_init("busy_msg", "")
_init("step", None)               # 'optimize' -> 'train' -> 'forecast' -> None
_init("params", {})
_init("input_path", None)
_init("promo_path", None)
_init("hist_df", None)
_init("forecast_df", None)
_init("freq", None)
_init("sku_select", None)         # выбранный SKU
_init("cat_select", None)         # выбранная категория
_init("chart_mode", "SKU")        # "SKU" | "Категория"
_init("_scroll_to", None)         # флаг для авто-скролла к графику

# ====== нормализация SKU ======
def _norm_sku_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.where(~s.isin(["nan", "NaN", "None"]), other="NA")
    return s

def _norm_sku_col(df: pd.DataFrame, col: str = "sku") -> pd.DataFrame:
    if col not in df.columns:
        df[col] = "ALL"
        return df
    df[col] = _norm_sku_series(df[col])
    return df

# ====== Header ======
col1, col2 = st.columns([1, 12])
if Path(LOGO_BIG).exists():
    try:
        img_b64 = b64encode(Path(LOGO_BIG).read_bytes()).decode()
        with col1:
            st.markdown(
                f"<img src='data:image/png;base64,{img_b64}' "
                f"style='width:72px;height:auto;display:block;'/>",
                unsafe_allow_html=True
            )
    except Exception:
        with col1: st.markdown("<div style='font-size:48px'>📈</div>", unsafe_allow_html=True)
else:
    with col1: st.markdown("<div style='font-size:48px'>📈</div>", unsafe_allow_html=True)

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
    p = Path(OUTDIR) / f"forecast_{freq}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Не найден {p}")
    df = pd.read_csv(p, dtype={"sku": str})
    df = _norm_sku_col(df, "sku")
    return df

def _focus_chart():
    ss["_scroll_to"] = "chart"

# ====== Плоты ======
def plot_category_interactive(history: pd.DataFrame,
                              forecast: pd.DataFrame,
                              category: str,
                              freq: Optional[str] = None,
                              show_band: bool = True,
                              show_points: bool = True,
                              show_promo: bool = True):
    h = history.copy(); f = forecast.copy()
    h["category"] = h.get("category", "ALL").astype(str)
    f["category"] = f.get("category", "ALL").astype(str)
    cat = str(category)

    h = h[h["category"] == cat].copy()
    f = f[f["category"] == cat].copy()
    if h.empty and f.empty:
        st.info("Нет данных для выбранной категории.")
        return

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    h_agg = h.groupby("date", as_index=False)["qty"].sum()
    cols_to_sum = [c for c in ["p10", "p50", "p90"] if c in f.columns]
    f_agg = f.groupby("date", as_index=False)[cols_to_sum].sum() if cols_to_sum else f.copy()
    h_agg.sort_values("date", inplace=True); f_agg.sort_values("date", inplace=True)

    fig = go.Figure()
    fig.add_scatter(x=h_agg["date"], y=h_agg["qty"],
                    mode="lines+markers" if show_points else "lines",
                    name="История (qty, сумма по категории)",
                    hovertemplate="Дата: %{x|%Y-%m-%d}<br>Продажи: %{y:,}<extra></extra>")
    if "p50" in f_agg.columns and len(f_agg):
        fig.add_scatter(x=f_agg["date"], y=f_agg["p50"], mode="lines",
                        name="Прогноз p50 (сумма по категории)",
                        line=dict(dash="dash"),
                        hovertemplate="Дата: %{x|%Y-%m-%d}<br>p50: %{y:,}<extra></extra>")
    if show_band and {"p10","p90"}.issubset(f_agg.columns) and len(f_agg):
        fig.add_scatter(x=f_agg["date"], y=f_agg["p90"], mode="lines", line=dict(width=0),
                        name="p90", hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=f_agg["date"], y=f_agg["p10"], mode="lines", line=dict(width=0),
                        fill="tonexty", name="Диапазон p10–p90",
                        hovertemplate="Дата: %{x|%Y-%m-%d}<br>p10: %{y:,}<extra></extra>")

    if show_promo and "promo_flag" in h.columns and h["promo_flag"].sum() > 0:
        shapes = []
        for dt in h.loc[h["promo_flag"] == 1, "date"].unique():
            if (freq or ss.get("freq")) == "monthly":
                start = pd.to_datetime(dt).replace(day=1)
                end = (start + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1)
            else:
                start = pd.to_datetime(dt); end = start + pd.Timedelta(days=1)
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=start, x1=end, y0=0, y1=1,
                               fillcolor="rgba(255,165,0,0.15)", line=dict(width=0)))
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        title=dict(text=f"Категория: {cat}", x=0.01, xanchor="left", y=0.99, yanchor="top",
                   pad=dict(t=16, b=16)),
        margin=dict(l=10, r=10, t=70, b=20),
        xaxis=dict(title="Дата",
                   rangeslider=dict(visible=True),
                   rangeselector=dict(buttons=list([
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                   ]))),
        yaxis=dict(title="Продажи"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config=dict(displaylogo=False))

def plot_sku_interactive(history: pd.DataFrame,
                         forecast: pd.DataFrame,
                         sku: str,
                         show_band: bool = True,
                         show_points: bool = True,
                         show_promo: bool = True):
    h = _norm_sku_col(history.copy(), "sku")
    f = _norm_sku_col(forecast.copy(), "sku")
    sku = _norm_sku_series(pd.Series([sku])).iloc[0]

    h = h[h["sku"] == sku]; f = f[f["sku"] == sku]
    if h.empty and f.empty:
        st.info("Нет данных для выбранного SKU.")
        return

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    h.sort_values("date", inplace=True); f.sort_values("date", inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["qty"],
                             mode="lines+markers" if show_points else "lines",
                             name="История (qty)",
                             hovertemplate="Дата: %{x|%Y-%m-%d}<br>Продажи: %{y:,}<extra></extra>"))
    if "p50" in f.columns and len(f):
        fig.add_trace(go.Scatter(x=f["date"], y=f["p50"], mode="lines",
                                 name="Прогноз p50", line=dict(dash="dash"),
                                 hovertemplate="Дата: %{x|%Y-%m-%d}<br>p50: %{y:,}<extra></extra>"))
    if show_band and {"p10","p90"}.issubset(f.columns) and len(f):
        fig.add_trace(go.Scatter(x=f["date"], y=f["p90"], mode="lines", line=dict(width=0),
                                 name="p90", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=f["date"], y=f["p10"], mode="lines", line=dict(width=0),
                                 fill="tonexty", name="Диапазон p10–p90",
                                 hovertemplate="Дата: %{x|%Y-%m-%d}<br>p10: %{y:,}<extra></extra>"))
    if show_promo and "promo_flag" in h.columns and h["promo_flag"].sum() > 0:
        shapes = []
        for dt in h.loc[h["promo_flag"] == 1, "date"]:
            if ss.freq == "monthly":
                start = pd.to_datetime(dt).replace(day=1)
                end = (start + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1)
            else:
                start = pd.to_datetime(dt); end = start + pd.Timedelta(days=1)
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=start, x1=end, y0=0, y1=1,
                               fillcolor="rgba(255,165,0,0.15)", line=dict(width=0)))
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Дата",
                   rangeslider=dict(visible=True),
                   rangeselector=dict(buttons=list([
                       dict(count=3, label="3M", step="month", stepmode="backward"),
                       dict(count=6, label="6M", step="month", stepmode="backward"),
                       dict(count=1, label="1Y", step="year", stepmode="backward"),
                       dict(step="all", label="All")
                   ]))),
        yaxis=dict(title="Продажи"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True,
                    config=dict(displaylogo=False, toImageButtonOptions=dict(filename=f"forecast_{sku}")))

# ====== Sidebar ======
st.sidebar.title("⚙️ Параметры")

uploaded_csv = st.sidebar.file_uploader(
    "Загрузить исходный CSV (monthly/daily)",
    type=["csv"],
    help=("CSV с колонками: date (YYYY-MM-DD), sku, qty, promo_flag (0/1), category (опц.).")
)

freq = st.sidebar.selectbox(
    "Частота", ["monthly", "daily"], index=0,
    help=("Гранулярность данных и прогноза.")
)

model_algo = st.sidebar.selectbox(
    "Модель для Optimize", ["lgbm", "catboost"], index=0,
    help=("Какой алгоритм оптимизируем на этапе HPO.")
)

do_optimize = st.sidebar.checkbox(
    "Запустить Optimize (HPO)", value=True,
    help=("Перебор гиперпараметров (Optuna) с кросс-валидацией по времени.")
)

trials = st.sidebar.number_input(
    "HPO trials", min_value=10, max_value=500, value=60, step=10,
    help=("Сколько попыток выполнить при поиске гиперпараметров.")
)

st.sidebar.markdown("---")

horizon = st.sidebar.number_input(
    "Horizon", min_value=1, max_value=60, value=12,
    help=("Горизонт прогноза в периодах.")
)

cv = st.sidebar.number_input(
    "CV folds", min_value=2, max_value=10, value=3,
    help=("Сколько раз проверяем модель, двигая окно по времени.")
)

st.sidebar.markdown('<span id="eta_anchor"></span>', unsafe_allow_html=True)
eta = st.sidebar.number_input(
    "Ensemble η (жесткость штрафа)",
    min_value=0.1, max_value=20.0, value=5.0, step=0.1,
    help=("Пока не используется в расчёте весов; оставьте по умолчанию."),
    key="eta_ui",
)

lookback_months_val = st.sidebar.number_input(
    "Lookback (мес, ~дни для daily)", min_value=0, max_value=240, value=60,
    help=("Сколько последних периодов брать для обучения (0 = вся история).")
)

halflife_val = st.sidebar.number_input(
    "Halflife (мес/дни)", min_value=0, max_value=120, value=18,
    help=("Период полураспада для затухания весов обучающих точек (0 = выкл.).")
)

st.sidebar.markdown("---")

joint = st.sidebar.checkbox(
    "Совместный прогноз по всем SKU", value=True,
    help=("Обучаем одну модель на всех товарах (SKU/категория как признаки).")
)

calibrate_last_k = st.sidebar.number_input(
    "Калибровка: последние k точек", min_value=0, max_value=24, value=6,
    help=("Пост-калибровка уровня p50 по последним k фактам.")
)

cal_clip_low = st.sidebar.number_input(
    "Calibrate clip low", min_value=0.1, max_value=2.0, value=0.6, step=0.05,
    help=("Нижняя граница коэффициента масштабирования p50 при калибровке.")
)

cal_clip_high = st.sidebar.number_input(
    "Calibrate clip high", min_value=0.1, max_value=3.0, value=1.4, step=0.05,
    help=("Верхняя граница коэффициента масштабирования p50 при калибровке.")
)

promo_file = st.sidebar.file_uploader(
    "План промо (опц.) CSV", type=["csv"],
    help=("Опционально: date, sku (или ALL), promo_flag=1.")
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button(
    "🚀 Запустить Forecast",
    help="Последовательно: HPO (если включён) → обучение ансамбля → прогноз."
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

# ====== Обработка клика ======
if run_btn:
    try:
        # CSV
        if uploaded_csv is not None:
            input_path = save_uploaded_file(uploaded_csv, WORKDIR / "data.csv")
            ss.input_path = str(input_path)
        elif ss.input_path:
            input_path = Path(ss.input_path)
            if not input_path.exists():
                st.warning("Ранее загруженный CSV не найден, загрузите файл заново."); st.stop()
        else:
            st.warning("Загрузите исходный CSV."); st.stop()

        # нормализация колонок (+ SKU как строки)
        df_hist = pd.read_csv(ss.input_path, dtype={"sku": str})
        df_hist = _norm_sku_col(df_hist, "sku")
        if "qty" not in df_hist.columns: raise ValueError("В CSV должна быть колонка 'qty'.")
        if "date" not in df_hist.columns: raise ValueError("В CSV должна быть колонка 'date' (YYYY-MM-DD).")
        if "promo_flag" not in df_hist.columns: df_hist["promo_flag"] = 0
        if "category" not in df_hist.columns: df_hist["category"] = "ALL"
        df_hist.to_csv(ss.input_path, index=False)

        # промо-план
        if promo_file is not None:
            ss.promo_path = str(save_uploaded_file(promo_file, WORKDIR / "promo.csv"))

        # параметры запуска
        ss.params = dict(
            freq=freq, model_algo=model_algo, do_optimize=bool(do_optimize),
            trials=int(trials), horizon=int(horizon), cv=int(cv), eta=float(eta),
            lookback=int(lookback_months_val) if lookback_months_val > 0 else None,
            halflife=int(halflife_val) if halflife_val > 0 else None,
            joint=bool(joint), calibrate_last_k=int(calibrate_last_k),
            cal_clip_low=float(cal_clip_low), cal_clip_high=float(cal_clip_high),
        )

        ss.is_running = True
        if ss.params["do_optimize"]:
            ss.step = "optimize"; ss.busy_msg = f"Запуск HPO ({ss.params['model_algo']})…"
        else:
            ss.step = "train"; ss.busy_msg = "Обучение ансамбля…"
        rerun()

    except Exception:
        err_box.error("Ошибка на этапе подготовки:\n\n" + "".join(traceback.format_exc()))

# ====== Машина состояний ======
try:
    if ss.is_running and ss.step:
        p = ss.params
        if ss.step == "optimize":
            run_optimize(ss.input_path, p["model_algo"], p["freq"], p["horizon"], p["cv"], p["trials"])
            ss.step = "train"; ss.busy_msg = "Обучение ансамбля…"; rerun()

        elif ss.step == "train":
            run_train(ss.input_path, p["freq"], p["horizon"], p["cv"], p["eta"], p["lookback"], p["halflife"])
            ss.step = "forecast"; ss.busy_msg = "Расчёт прогноза…"; rerun()

        elif ss.step == "forecast":
            model_path = str(Path(OUTDIR) / f"model_ensemble_{p['freq']}.joblib")
            run_forecast(ss.input_path, model_path, p["freq"], p["horizon"],
                         p["joint"], p["calibrate_last_k"], p["cal_clip_low"], p["cal_clip_high"],
                         ss.promo_path)

            ss.forecast_df = load_forecast_csv(p["freq"])
            ss.hist_df = pd.read_csv(ss.input_path, dtype={"sku": str})
            ss.hist_df = _norm_sku_col(ss.hist_df, "sku")
            ss.freq = p["freq"]

            # выставим дефолты для селектов (без index)
            if ss.hist_df is not None:
                skus = sorted(ss.hist_df["sku"].dropna().astype(str).unique().tolist())
                if not skus:
                    ss.sku_select = None
                elif ss.sku_select not in skus:
                    ss.sku_select = skus[0]
                cats = sorted(ss.hist_df.get("category", pd.Series(["ALL"])).astype(str).unique().tolist())
                if not cats:
                    ss.cat_select = None
                elif ss.cat_select not in cats:
                    ss.cat_select = cats[0]

            ss.is_running = False; ss.busy_msg = ""; ss.step = None
            rerun()
except Exception:
    ss.is_running = False; ss.busy_msg = ""; ss.step = None
    err_box.error("Ошибка во время выполнения шага:\n\n" + "".join(traceback.format_exc()))

# ====== Отрисовка результатов ======
st.markdown(f'<div class="{ "blurred" if ss.is_running else "" }">', unsafe_allow_html=True)

if ss.forecast_df is not None and ss.hist_df is not None:
    df_h = ss.hist_df.copy()
    df_f = ss.forecast_df.copy()

    # гарантируем наличие category в forecast
    if "category" not in df_h.columns:
        df_h["category"] = "ALL"
    df_h["category"] = df_h["category"].astype(str)
    df_h["sku"] = _norm_sku_series(df_h["sku"].astype(str))

    sku_cat = df_h[["sku", "category"]].drop_duplicates()
    df_f["sku"] = _norm_sku_series(df_f["sku"].astype(str))
    df_f = df_f.merge(sku_cat, on="sku", how="left")
    df_f["category"] = df_f["category"].fillna("ALL").astype(str)

    df_h["date"] = pd.to_datetime(df_h["date"], errors="coerce")
    df_f["date"] = pd.to_datetime(df_f["date"], errors="coerce")

    # таблица
    st.subheader("📄 Прогноз (первые 200 строк)")
    cols_order = [c for c in ["date", "sku", "category", "promo_flag", "p10", "p50", "p90"] if c in df_f.columns]
    st.dataframe(df_f[cols_order].head(200))

    # скачать CSV
    st.download_button(
        "⬇️ Скачать forecast CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{ss.freq or 'monthly'}.csv",
        mime="text/csv",
        key="download_forecast_btn"
    )

    # якорь для авто-скролла к графику
    st.markdown('<span id="chart_anchor"></span>', unsafe_allow_html=True)

    # режим графика (храним в state)
    ss.chart_mode = st.radio("График по:", ["SKU", "Категория"],
                             horizontal=True, index=0 if ss.chart_mode=="SKU" else 1,
                             key="chart_mode_radio")

    if ss.chart_mode == "Категория":
        cats = sorted(df_h["category"].dropna().unique().tolist())
        if not cats:
            st.info("Нет категорий в данных.")
        else:
            # инициализация значения (один раз) и стабильный select без index
            if ss.cat_select not in cats:
                ss.cat_select = cats[0]
            st.selectbox("Категория для графика", options=cats, key="cat_select",
                         on_change=_focus_chart)
            c1, c2, c3 = st.columns(3)
            with c1: show_band = st.checkbox("Показать p10–p90", value=True, key="band_cat")
            with c2: show_points = st.checkbox("Точки на истории", value=True, key="pts_cat")
            with c3: show_promo  = st.checkbox("Подсветка промо", value=("promo_flag" in df_h.columns), key="promo_cat")

            if ss.cat_select is not None:
                plot_category_interactive(df_h, df_f, ss.cat_select,
                                          freq=ss.freq, show_band=show_band,
                                          show_points=show_points, show_promo=show_promo)
    else:
        skus = sorted(df_h["sku"].dropna().unique().tolist())
        if not skus:
            st.info("Нет SKU в данных.")
        else:
            if ss.sku_select not in skus:
                ss.sku_select = skus[0]
            st.selectbox("SKU для графика", options=skus, key="sku_select",
                         on_change=_focus_chart)
            c1, c2, c3 = st.columns(3)
            with c1: show_band = st.checkbox("Показать p10–p90", value=True, key="band_sku")
            with c2: show_points = st.checkbox("Точки на истории", value=True, key="pts_sku")
            with c3: show_promo  = st.checkbox("Подсветка промо", value=("promo_flag" in df_h.columns), key="promo_sku")

            if ss.sku_select is not None:
                plot_sku_interactive(df_h, df_f, ss.sku_select,
                                     show_band=show_band, show_points=show_points, show_promo=show_promo)

# авто-скролл к графику после смены селекта
if ss.get("_scroll_to") == "chart":
    components.html("<script>window.parent.location.hash='chart_anchor';</script>", height=0)
    ss["_scroll_to"] = None

st.markdown('</div>', unsafe_allow_html=True)
