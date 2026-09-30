# -*- coding: utf-8 -*-
"""
data.py — Capa de datos del explorador de frontera (Coril · analista)
======================================================================

Dos modos:
  1. DEMO (sintético, offline): genera retornos realistas con estructura de
     factores por sector (usa bvl_catalog). Permite explorar la frontera SIN
     credenciales ni red — ideal para desarrollo y demo en Streamlit Cloud.
  2. REAL: renta variable vía la API de Coril (coril_api) y renta fija /
     benchmark vía yfinance, igual que el app comercial. Si falta una
     dependencia o credencial, cae a DEMO de forma transparente.

Devuelve siempre log-retornos DIARIOS alineados (DataFrame fechas × activos) y
la serie de log-retornos del benchmark.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from bvl_catalog import BVL_SECTORS, BVL_NAMES, BVL_CURRENCY
except Exception:                       # catálogo mínimo si no está
    BVL_SECTORS, BVL_NAMES, BVL_CURRENCY = {}, {}, {}


@dataclass
class MarketData:
    returns:   pd.DataFrame        # log-ret diarios (fechas × activos RV+RF de mercado)
    benchmark: pd.Series           # log-ret diarios del benchmark
    source:    str                 # "demo" | "real" | "mixto"
    note:      str = ""            # detalle para la interfaz


# ─────────────────────────── Metadatos de catálogo ────────────────────────
def sector_of(ticker: str) -> str:
    if ticker in BVL_SECTORS:
        return BVL_SECTORS[ticker]
    return "Renta Fija" if ticker.startswith("FICO") else "Otros"


def name_of(ticker: str) -> str:
    return BVL_NAMES.get(ticker, ticker)


def currency_of(ticker: str) -> str:
    return BVL_CURRENCY.get(ticker, "USD")


# =============================================================================
# MODO DEMO — retornos sintéticos con factores por sector
# =============================================================================

# Perfil de retorno/vol anual aproximado por "familia" de sector (solo demo).
_SECTOR_PROFILE = {
    "Tecnología":            (0.18, 0.34),
    "Consumo discrecional":  (0.12, 0.28),
    "Consumo Básico":        (0.08, 0.18),
    "Salud":                 (0.10, 0.22),
    "Financiero":            (0.11, 0.26),
    "Financieras":           (0.11, 0.26),
    "Industrial":            (0.09, 0.24),
    "Industriales":          (0.09, 0.24),
    "Materiales":            (0.10, 0.30),
    "Mineras":               (0.13, 0.38),
    "Inmobiliario":          (0.07, 0.22),
    "Energía":               (0.09, 0.32),
    "ETF internacional":     (0.10, 0.20),
    "Otros (internacional)": (0.08, 0.24),
    "Otros":                 (0.08, 0.24),
    "Renta Fija":            (0.04, 0.06),
}
_RF_PROFILE = {   # ETFs de renta fija de mercado (demo)
    "AGG": (0.035, 0.05), "BND": (0.035, 0.05), "TLT": (0.03, 0.13),
    "IEF": (0.03, 0.07),  "SHY": (0.025, 0.015), "LQD": (0.045, 0.08),
    "TIP": (0.035, 0.06), "EMB": (0.055, 0.10),
}


def synthetic_returns(equity: Sequence[str], rf: Sequence[str],
                      benchmark: str = "^GSPC", years: float = 4.0,
                      seed: int = 123) -> MarketData:
    """
    Genera log-retornos diarios sintéticos con estructura de factores:
        r = β_mkt·F_mkt + β_sec·F_sector + ε
    de modo que las correlaciones intra-sector sean realistas y la frontera
    se comporte de forma no trivial. Los niveles salen de _SECTOR_PROFILE.
    """
    rng = np.random.default_rng(seed)
    n = int(years * 252)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    PPY = 252

    # Factor de mercado común
    F_mkt = rng.normal(0.06 / PPY, 0.14 / np.sqrt(PPY), n)
    # Factores por sector
    sectores = sorted({sector_of(t) for t in equity})
    F_sec = {s: rng.normal(0.0, 0.10 / np.sqrt(PPY), n) for s in sectores}

    cols = {}
    for t in equity:
        s = sector_of(t)
        mu_a, vol_a = _SECTOR_PROFILE.get(s, _SECTOR_PROFILE["Otros"])
        # dispersión idiosincrática del activo dentro del sector
        mu_a = mu_a * rng.uniform(0.6, 1.3)
        beta_mkt = rng.uniform(0.7, 1.3)
        beta_sec = rng.uniform(0.6, 1.1)
        resid_vol = max(vol_a**2 - (beta_mkt*0.14)**2 - (beta_sec*0.10)**2, 0.02**2)
        eps = rng.normal(0.0, np.sqrt(resid_vol) / np.sqrt(PPY), n)
        r = mu_a/PPY + beta_mkt*(F_mkt - 0.06/PPY) + beta_sec*F_sec[s] + eps
        cols[t] = r

    for t in rf:
        mu_a, vol_a = _RF_PROFILE.get(t, (0.035, 0.06))
        beta_mkt = rng.uniform(-0.05, 0.10)
        eps = rng.normal(0.0, vol_a/np.sqrt(PPY), n)
        cols[t] = mu_a/PPY + beta_mkt*(F_mkt - 0.06/PPY) + eps

    returns = pd.DataFrame(cols, index=idx)
    # Benchmark ≈ factor de mercado
    bench = pd.Series(F_mkt, index=idx, name=benchmark)
    return MarketData(returns=returns, benchmark=bench, source="demo",
                      note=f"Datos sintéticos ({years:.0f} años, {len(equity)} RV + "
                           f"{len(rf)} RF). Sin conexión a mercado.")


# =============================================================================
# MODO REAL — Coril API (RV) + yfinance (RF y benchmark)
# =============================================================================

def _yf_logret(tickers: Sequence[str], start, end) -> pd.DataFrame:
    """Log-retornos diarios vía yfinance (RF y benchmark)."""
    import yfinance as yf
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(list(tickers), start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px.dropna(how="all")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    cal = pd.date_range(px.index.min(), px.index.max(), freq="B")
    px = px.reindex(cal).ffill()
    return np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def load_real(equity: Sequence[str], rf: Sequence[str], benchmark: str,
              start, end) -> Optional[MarketData]:
    """
    Intenta cargar datos reales. Devuelve None si no hay forma (sin creds /
    sin dependencias / sin datos), para que el llamador caiga a demo.
    """
    try:
        import coril_api
    except Exception:
        coril_api = None

    eq_ret = pd.DataFrame()
    # Renta variable: Coril API (si está configurada)
    if coril_api is not None and getattr(coril_api, "api_configurada", lambda: False)():
        try:
            eq_ret = coril_api.log_retornos(list(equity))
        except Exception:
            eq_ret = pd.DataFrame()

    # Renta fija de mercado + benchmark: yfinance
    try:
        rf_ret = _yf_logret(list(rf), start, end)
    except Exception:
        rf_ret = pd.DataFrame()
    try:
        bench_ret = _yf_logret([benchmark], start, end)
        bench = bench_ret[benchmark] if benchmark in bench_ret.columns else pd.Series(dtype=float)
    except Exception:
        bench = pd.Series(dtype=float)

    if eq_ret.empty and rf_ret.empty:
        return None

    # Alinear al rango común real de la RV (evita estirar al histórico del bono)
    frames = [df for df in (eq_ret, rf_ret) if not df.empty]
    returns = pd.concat(frames, axis=1)
    if not eq_ret.empty:
        lo, hi = eq_ret.dropna(how="all").index.min(), eq_ret.dropna(how="all").index.max()
        returns = returns.loc[lo:hi]
    returns = returns.dropna(how="all").ffill()
    if not bench.empty:
        bench = bench.reindex(returns.index).ffill()

    src = "real" if not eq_ret.empty else "mixto"
    note = (f"RV Coril: {eq_ret.shape[1]} activos · RF yfinance: {rf_ret.shape[1]} · "
            f"{len(returns)} días.") if not eq_ret.empty else \
           "Sin RV de Coril (revisa credenciales); solo RF de mercado."
    return MarketData(returns=returns, benchmark=bench, source=src, note=note)


# =============================================================================
# ENTRADA ÚNICA
# =============================================================================

def get_market_data(equity: Sequence[str], rf: Sequence[str],
                    benchmark: str = "^GSPC", years: float = 4.0,
                    modo: str = "auto", seed: int = 123) -> MarketData:
    """
    modo:
      · "demo"  → siempre sintético.
      · "real"  → intenta real; si falla, cae a demo con aviso.
      · "auto"  → real si hay credenciales/datos, si no demo.
    """
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=int(max(years, 1)))

    if modo == "demo":
        return synthetic_returns(equity, rf, benchmark, years, seed)

    md = load_real(equity, rf, benchmark, start, end)
    if md is not None and not md.returns.empty:
        return md

    # Fallback
    md = synthetic_returns(equity, rf, benchmark, years, seed)
    if modo == "real":
        md.note = "⚠️ No se pudo cargar data real (credenciales o red). " + md.note
    return md
