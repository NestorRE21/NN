# -*- coding: utf-8 -*-
"""
profiles.py — Perfiles de riesgo y FICOs de Coril (versión analista)
=====================================================================

Misma dinámica que el app comercial:
  · 5 perfiles definidos por el split Renta Variable / Renta Fija.
  · Dos FICOs de Coril como activos de renta fija con retorno forzado
    (Soles 7 %, Dólares 6 %), inyectados al bucket de RF.

Aquí solo se declaran los parámetros; el motor (optimizer.py) y la frontera
(frontier.py) los consumen. Cambiar un % de perfil o el retorno de un FICO se
hace en un único lugar: este archivo.
"""
from __future__ import annotations
from dataclasses import dataclass

from optimizer import RiskProfile, ForcedAsset


# ─────────────────────────── Perfiles (RV / RF) ───────────────────────────
# (equity_target, fico_target)  — deben sumar 1.0
PROFILE_SPLITS: dict[str, tuple[float, float]] = {
    "Conservador":   (0.30, 0.70),
    "Moderado-bajo": (0.40, 0.60),
    "Moderado":      (0.50, 0.50),
    "Crecimiento":   (0.60, 0.40),
    "Agresivo":      (0.70, 0.30),
}

PROFILE_DESC: dict[str, str] = {
    "Conservador":   "Preservar capital. Máxima renta fija.",
    "Moderado-bajo": "Leve crecimiento con colchón de RF.",
    "Moderado":      "Balance entre riesgo y retorno.",
    "Crecimiento":   "Mayor exposición a renta variable.",
    "Agresivo":      "Máxima renta variable dentro del mandato.",
}

PROFILE_ORDER = list(PROFILE_SPLITS.keys())


def make_profile(name: str) -> RiskProfile:
    """Construye el RiskProfile del motor a partir del nombre de perfil."""
    eq, fi = PROFILE_SPLITS[name]
    return RiskProfile.for_split(eq, fi, label=f"{name} ({eq:.0%}/{fi:.0%})")


def profile_label(name: str) -> str:
    eq, fi = PROFILE_SPLITS[name]
    return f"{name} ({eq:.0%}/{fi:.0%})"


# ─────────────────────────────── FICOs Coril ──────────────────────────────
# Retorno/vol forzados: no se estiman de datos, se fijan (fondos de Coril).
FICO_PEN = ForcedAsset(ret_annual=0.07, vol_annual=0.010, beta=0.30,
                       sector="Factoring", region="Perú", moneda="PEN",
                       instrumento="Fondo", asset_class="Renta Fija")
FICO_USD = ForcedAsset(ret_annual=0.06, vol_annual=0.010, beta=0.30,
                       sector="Factoring", region="Perú", moneda="USD",
                       instrumento="Fondo", asset_class="Renta Fija")

FICOS: dict[str, dict] = {
    "FICO_PEN": {"ticker": "FICO_PEN", "obj": FICO_PEN,
                 "nombre": "FICO Coril Soles (7%)",   "moneda": "PEN"},
    "FICO_USD": {"ticker": "FICO_USD", "obj": FICO_USD,
                 "nombre": "FICO Coril Dólares (6%)", "moneda": "USD"},
}


def forced_from_choice(choice: str | None) -> dict[str, ForcedAsset]:
    """
    Devuelve el dict {ticker: ForcedAsset} para inyectar al motor.
    choice ∈ {None/"", "FICO_PEN", "FICO_USD", "ambos"}.
    """
    if not choice or choice == "No incluir":
        return {}
    if choice == "ambos":
        return {k: v["obj"] for k, v in FICOS.items()}
    if choice in FICOS:
        return {choice: FICOS[choice]["obj"]}
    return {}


def fico_display(ticker: str) -> str:
    """Nombre visible de un FICO."""
    return FICOS.get(ticker, {}).get("nombre", ticker)


def es_fico(ticker: str) -> bool:
    return ticker in FICOS or ticker == "FICCMP13"


# ───────────────────── ETFs de renta fija de mercado (yfinance) ───────────
# Universo sugerido para el bucket de RF (además del FICO). El analista puede
# agregar o quitar. Se descargan vía yfinance como el app comercial.
RF_ETFS: dict[str, str] = {
    "AGG": "Bonos EE.UU. amplio (AGG)",
    "BND": "Bonos totales (BND)",
    "TLT": "Bonos Tesoro largo plazo (TLT)",
    "IEF": "Bonos Tesoro 7-10 años (IEF)",
    "SHY": "Bonos Tesoro corto plazo (SHY)",
    "LQD": "Bonos corporativos IG (LQD)",
    "TIP": "Bonos indexados a inflación (TIP)",
    "EMB": "Bonos emergentes USD (EMB)",
}

# Benchmarks típicos (índices) — solo referencia, no entran a la optimización.
BENCHMARKS: dict[str, str] = {
    "^GSPC":    "S&P 500 (EE.UU.)",
    "^SPBLPGPT":"S&P/BVL Perú General",
    "^IXIC":    "Nasdaq Composite",
    "ACWI":     "MSCI ACWI (global)",
    "EPU":      "MSCI Perú (EPU)",
}
