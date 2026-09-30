# -*- coding: utf-8 -*-
"""
app_analista.py — Explorador de Frontera Eficiente por Perfil (Coril · analista)
================================================================================

Herramienta de asset allocation + security selection para analistas. El centro
es el gráfico "Composición del portafolio a lo largo de la frontera eficiente"
por perfil: permite descubrir portafolios a lo largo de la frontera manteniendo
la MISMA dinámica de perfiles y los mismos porcentajes de FICO / renta fija que
el modelo comercial de Coril.

Reutiliza:
    optimizer.py   — motor Black-Litterman (equilibrio, BL, MV).
    frontier.py    — frontera eficiente CON pesos por punto (núcleo del gráfico).
    profiles.py    — perfiles (30/70 … 70/30) y FICOs (Soles 7 % / Dólares 6 %).
    charts.py      — gráficos Plotly.
    data.py        — datos (demo sintético offline · real Coril+yfinance).
    projections.py — Monte Carlo y stress testing (opcional).
    bvl_catalog.py — universo BVL (sector, nombre, moneda).

Ejecutar:  streamlit run app_analista.py
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st

from optimizer import BLConfig, GKConfig, generate_gk_views, estimate_covariance
import frontier as F
import charts
import profiles as P
import data as D

try:
    from bvl_catalog import BVL_TICKERS
except Exception:
    BVL_TICKERS = []

# ───────────────────────────── Config general ─────────────────────────────
st.set_page_config(page_title="Coril · Frontera por Perfil",
                   page_icon="📈", layout="wide")

RF_RATE = 0.02        # tasa libre de riesgo anual
PPY     = 252         # días hábiles/año (datos diarios)

DEFAULT_EQUITY = ["ALICORC1", "BUENAVC1", "CREDITC1", "FERREYC1",
                  "INRETC1", "VOLCABC1"]
DEFAULT_RF     = ["AGG", "TLT", "LQD"]

st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
  h1, h2, h3 {color:#16324f;}
  .metric-row {font-size:0.9rem;}
  .stCaption {color:#64748b;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────── Utilidades ─────────────────────────────────
def display_name(ticker: str) -> str:
    if ticker in P.FICOS:
        return P.FICOS[ticker]["nombre"]
    return D.name_of(ticker)


@st.cache_data(show_spinner=False)
def cargar_datos(equity: tuple, rf: tuple, benchmark: str,
                 years: int, modo: str, seed: int):
    md = D.get_market_data(list(equity), list(rf), benchmark, years, modo, seed)
    return md.returns, md.benchmark, md.source, md.note


@st.cache_data(show_spinner=False)
def construir_fronteras(returns: pd.DataFrame, benchmark: pd.Series,
                        equity: tuple, rf: tuple, fico_choice: str,
                        perfiles: tuple, cap: float, tau: float,
                        n_points: int, path_smooth: float,
                        usar_gk: bool, ic: float):
    cfg = BLConfig(rf_annual=RF_RATE, periods_per_year=PPY,
                   tau=tau, max_weight_equity=cap, gamma_beta=5.0)
    forced = P.forced_from_choice(fico_choice)
    equity = [a for a in equity if a in returns.columns]

    # Vistas automáticas Grinold-Kahn (opcional), sobre RV
    views = []
    if usar_gk and equity:
        cov_eq = estimate_covariance(returns[equity], PPY, cfg.ridge)
        views = generate_gk_views(returns[equity], equity, cov_eq,
                                  GKConfig(ic=ic), forced_tickers=list(forced),
                                  periods_per_year=PPY, rf_annual=RF_RATE)

    resultados = {}
    for name in perfiles:
        prof = P.make_profile(name)
        mu, cov, eqa, fia = F.build_bl_inputs(
            returns, equity, forced, prof, rf_assets=list(rf), views=views,
            config=cfg, benchmark_returns=benchmark, views_as_alpha=True)
        fr = F.frontier_with_weights(mu, cov, eqa, fia, prof, cfg,
                                     n_points=n_points, path_smooth=path_smooth)
        resultados[name] = fr
    return resultados


def beta_portafolio(weights: pd.Series, returns: pd.DataFrame,
                    benchmark: pd.Series, forced: dict) -> float:
    """β del portafolio: cov(activo, bench)/var(bench); FICO usa su β forzado."""
    if benchmark is None or benchmark.empty:
        return float("nan")
    var_b = benchmark.var()
    if not np.isfinite(var_b) or var_b <= 0:
        return float("nan")
    b = 0.0
    for a, w in weights.items():
        if w <= 1e-6:
            continue
        if a in forced:
            b += w * forced[a].beta
        elif a in returns.columns:
            cov_ab = returns[a].cov(benchmark)
            b += w * (cov_ab / var_b)
        else:
            b += w * 1.0
    return float(b)


def excel_export(resultados: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, fr in resultados.items():
            tab = fr.table.copy()
            hoja = name[:28]
            tab.to_excel(xw, sheet_name=hoja, index_label="punto")
    return buf.getvalue()


# ═══════════════════════════════ SIDEBAR ══════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Frontera por Perfil")
    st.caption("Asset allocation + security selection · Grupo Coril")

    modo = st.radio("Fuente de datos", ["Demo (offline)", "Real (Coril + yfinance)"],
                    help="Demo genera datos sintéticos realistas para explorar sin "
                         "credenciales. Real usa la API de Coril (RV) y yfinance (RF).")
    modo_key = "demo" if modo.startswith("Demo") else "real"

    st.markdown("#### Universo de renta variable")
    universo = BVL_TICKERS or DEFAULT_EQUITY
    equity = st.multiselect("Acciones / ETFs (RV)", options=universo,
                            default=[t for t in DEFAULT_EQUITY if t in universo] or universo[:6],
                            help="Selección de valores (security selection).")

    st.markdown("#### Renta fija")
    rf = st.multiselect("ETFs de renta fija (mercado)", options=list(P.RF_ETFS),
                        default=DEFAULT_RF,
                        format_func=lambda t: P.RF_ETFS.get(t, t))
    fico_choice = st.radio("FICO Coril", ["No incluir", "FICO_PEN", "FICO_USD", "ambos"],
                           index=2,
                           format_func=lambda k: {"No incluir": "No incluir",
                                                  "FICO_PEN": "Soles (7%)",
                                                  "FICO_USD": "Dólares (6%)",
                                                  "ambos": "Ambos"}[k])

    st.markdown("#### Perfiles a comparar")
    perfiles = st.multiselect("Perfiles", options=P.PROFILE_ORDER,
                              default=["Moderado", "Agresivo"],
                              format_func=P.profile_label)

    benchmark = st.selectbox("Benchmark", options=list(P.BENCHMARKS),
                             format_func=lambda t: P.BENCHMARKS.get(t, t))

    with st.expander("⚙️ Parámetros del motor"):
        years   = st.slider("Años de histórico", 1, 10, 4)
        cap     = st.slider("Tope por activo RV", 0.05, 1.0, 0.25, 0.05,
                            help="max_weight_equity: límite de concentración por acción.")
        tau     = st.slider("τ (incertidumbre del equilibrio)", 0.01, 0.20, 0.05, 0.01)
        n_pts   = st.slider("Puntos de la frontera", 20, 160, 80, 10)
        smooth  = st.select_slider("Suavizado del camino (η)",
                                   options=[0.0, 1e-3, 5e-3, 1e-2, 2e-2], value=5e-3)
        usar_gk = st.checkbox("Vistas automáticas (Grinold-Kahn)", value=False,
                              help="Añade alpha de momentum + low-vol a la RV.")
        ic      = st.slider("IC (habilidad del forecast)", 0.02, 0.15, 0.05, 0.01,
                            disabled=not usar_gk)
        seed    = st.number_input("Semilla (demo)", 1, 9999, 123)

    construir = st.button("🔧 Construir frontera", type="primary",
                          use_container_width=True)


# ═══════════════════════════════ MAIN ═════════════════════════════════════
st.title("Composición del portafolio a lo largo de la frontera eficiente")
st.caption("Explora, por perfil, cómo cambia la asignación al recorrer la "
           "frontera — del portafolio de mínima varianza al de máximo retorno, "
           "manteniendo el split RV/RF y los FICOs del mandato.")

# Validaciones mínimas
if not perfiles:
    st.info("Elige al menos un perfil en la barra lateral."); st.stop()
if not equity:
    st.warning("Agrega al menos un activo de renta variable."); st.stop()
if not rf and fico_choice == "No incluir":
    st.warning("El bucket de renta fija está vacío: agrega ETFs de RF o un FICO.")
    st.stop()

# Estado: construir al pulsar el botón o en la primera carga
if construir or "resultados" not in st.session_state:
    with st.spinner("Cargando datos y construyendo fronteras…"):
        returns, bench, fuente, nota = cargar_datos(
            tuple(equity), tuple(rf), benchmark, years, modo_key, int(seed))
        resultados = construir_fronteras(
            returns, bench, tuple(equity), tuple(rf), fico_choice,
            tuple(perfiles), cap, tau, int(n_pts), float(smooth), usar_gk, ic)
    st.session_state.update(resultados=resultados, returns=returns, bench=bench,
                            fuente=fuente, nota=nota,
                            forced=P.forced_from_choice(fico_choice))

resultados = st.session_state.resultados
returns    = st.session_state.returns
bench      = st.session_state.bench
forced     = st.session_state.forced

# Barra de contexto
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fuente", st.session_state.fuente.upper())
c2.metric("Activos", f"{returns.shape[1]}")
c3.metric("Días", f"{returns.shape[0]}")
c4.metric("Perfiles", f"{len(resultados)}")
st.caption(st.session_state.nota)

# ── Gráfico central: composición a lo largo de la frontera ──
st.plotly_chart(
    charts.composicion_frontera(resultados, display=display_name,
                                mostrar_lineas=True),
    use_container_width=True)

# ── Dispersión retorno vs riesgo ──
st.plotly_chart(charts.frontera_dispersion(resultados),
                use_container_width=True)

st.divider()

# ── Selección e inspección de un portafolio por perfil ──
st.subheader("Inspeccionar un portafolio de la frontera")
tabs = st.tabs([P.profile_label(n) for n in resultados])
for tab, name in zip(tabs, resultados):
    fr = resultados[name]
    n = len(fr.table)
    with tab:
        colsel, _ = st.columns([2, 3])
        idx = colsel.slider(
            "Posición en la frontera (riesgo →)", 0, n - 1,
            value=int(fr.idx_maxsharpe), key=f"sel_{name}",
            help="0 = mínima varianza · derecha = máximo retorno.")
        w = fr.point(idx)
        ret = float(fr.table["ret"].iloc[idx])
        vol = float(fr.table["vol"].iloc[idx])
        shp = float(fr.table["sharpe"].iloc[idx])
        beta = beta_portafolio(w, returns, bench, forced)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Retorno esp.", f"{ret:.2%}")
        m2.metric("Volatilidad", f"{vol:.2%}")
        m3.metric("Sharpe", f"{shp:.2f}")
        m4.metric("Beta", f"{beta:.2f}" if np.isfinite(beta) else "—")
        eq_w = float(w[[a for a in fr.assets if a in set(fr.equity_assets)]].sum())
        m5.metric("RV / RF", f"{eq_w:.0%} / {1-eq_w:.0%}")

        cA, cB = st.columns([3, 2])
        with cA:
            st.plotly_chart(
                charts.asignacion_barras(w, fr, display=display_name,
                                         titulo="Pesos del portafolio"),
                use_container_width=True)
        with cB:
            sect = {a: D.sector_of(a) for a in fr.assets}
            st.plotly_chart(charts.contribucion_sector(w, sect),
                            use_container_width=True)

        # Tabla de pesos
        tabla = pd.DataFrame({
            "Activo": [display_name(a) for a in w.index],
            "Clase": ["Renta Fija" if a in set(fr.fico_assets) else "Renta Variable"
                      for a in w.index],
            "Sector": [D.sector_of(a) for a in w.index],
            "Peso": w.values,
        })
        tabla = tabla[tabla["Peso"] > 0.001].sort_values("Peso", ascending=False)
        tabla["Peso"] = (tabla["Peso"] * 100).round(2)
        st.dataframe(tabla, use_container_width=True, hide_index=True,
                     column_config={"Peso": st.column_config.NumberColumn(
                         "Peso %", format="%.2f")})

        # Proyección opcional (Monte Carlo)
        with st.expander("📉 Monte Carlo y stress (opcional)"):
            if st.button("Simular", key=f"mc_{name}"):
                from projections import monte_carlo, stress_test, CRISIS_PERIODS
                mu_bl = fr.mu
                cov_bl = fr.cov
                mc = monte_carlo(w, mu_bl, cov_bl, capital=100_000,
                                 horizon_years=3, periods_per_year=PPY)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("P(pérdida)", f"{mc.prob_loss:.1%}")
                cc2.metric("VaR 95%", f"S/ {mc.var_terminal:,.0f}")
                cc3.metric("CVaR 95%", f"S/ {mc.cvar_terminal:,.0f}")
                figmc = None
                try:
                    import plotly.graph_objects as go
                    figmc = go.Figure()
                    figmc.add_trace(go.Scatter(x=mc.dates, y=mc.percentiles[50],
                                               name="Mediana", line=dict(color="#2171b5")))
                    figmc.add_trace(go.Scatter(x=mc.dates, y=mc.percentiles[95],
                                               name="P95", line=dict(width=0)))
                    figmc.add_trace(go.Scatter(x=mc.dates, y=mc.percentiles[5],
                                               name="P5", fill="tonexty",
                                               fillcolor="rgba(33,113,181,0.12)",
                                               line=dict(width=0)))
                    figmc.update_layout(height=320, title="Trayectorias de capital (3 años)",
                                        plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(figmc, use_container_width=True)
                except Exception:
                    pass

st.divider()

# ── Comparativa mín-var vs máx-Sharpe ──
st.subheader("Puntos clave por perfil")
filas = []
for name, fr in resultados.items():
    for etiqueta, idx in [("Mínima varianza", fr.idx_minvar),
                          ("Máximo Sharpe", fr.idx_maxsharpe)]:
        filas.append({
            "Perfil": name, "Portafolio": etiqueta,
            "Retorno": f"{fr.table['ret'].iloc[idx]:.2%}",
            "Volatilidad": f"{fr.table['vol'].iloc[idx]:.2%}",
            "Sharpe": f"{fr.table['sharpe'].iloc[idx]:.2f}",
        })
st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

# ── Export ──
st.download_button("⬇️ Descargar fronteras (Excel)",
                   data=excel_export(resultados),
                   file_name="fronteras_coril.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Motor Black-Litterman + frontera con pesos por punto. Los FICOs y el "
           "split RV/RF por perfil replican el mandato comercial de Coril.")
