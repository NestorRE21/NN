# -*- coding: utf-8 -*-
"""
frontier.py — Frontera eficiente con PESOS por punto (Coril · versión analista)
================================================================================

El motor `optimizer.efficient_frontier` devuelve solo (ret, vol) para dibujar
la curva. Para el gráfico "Composición del portafolio a lo largo de la frontera
eficiente" necesitamos, además, los PESOS de cada activo en cada punto de la
frontera. Este módulo hace exactamente eso, reutilizando el mismo sistema de
restricciones por buckets (RV / RF) del perfil.

Idea:
    Se barre el retorno objetivo de r_min a r_max (ascendente). En cada nivel se
    minimiza la varianza sujeto a:
        Σw = 1
        Σw_RV  = equity_target   (bucket duro del perfil)
        Σw_RF  = fico_target
        0 ≤ w_RV ≤ cap           (tope por activo de renta variable)
        wᵀμ = r_target           (retorno objetivo de ese punto)
    El total RV/RF se mantiene fijo (lo define el perfil); lo que CAMBIA a lo
    largo de la frontera es la composición INTERNA de cada bucket: a menor
    retorno objetivo domina lo defensivo/low-vol, a mayor retorno objetivo el
    bucket de RV se inclina hacia los activos de mayor μ (hasta el cap).

Salida principal:
    FrontierResult con:
        .table   DataFrame (n_points × [ret, vol, sharpe, <un activo por columna>])
                 ordenado por retorno objetivo ascendente (índice 0..n-1).
        .assets  orden de activos (columnas de pesos).
        .idx_minvar   índice del punto de mínima varianza (línea punteada negra).
        .idx_maxsharpe índice del punto de máximo Sharpe / tangencia (línea de color).

Sin dependencias de Streamlit. Solo numpy / pandas / scipy (+ el optimizer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimizer import (
    EPS, BLConfig, RiskProfile,
    nearest_psd, estimate_covariance, equilibrium_returns,
    black_litterman, build_views, calibrate_lambda, market_weights,
    inject_forced_assets, ForcedAsset, View,
)


# =============================================================================
# RESULTADO
# =============================================================================

@dataclass
class FrontierResult:
    """Frontera eficiente con pesos por punto para un perfil."""
    table:          pd.DataFrame      # n_points × [ret, vol, sharpe, *assets]
    assets:         list              # orden de columnas de pesos
    equity_assets:  list
    fico_assets:    list
    idx_minvar:     int               # punto de mínima varianza
    idx_maxsharpe:  int               # punto de máximo Sharpe (tangencia)
    profile:        RiskProfile
    mu:             pd.Series          # retornos usados (BL o μ crudo)
    cov:            pd.DataFrame       # covarianza usada
    rf_annual:      float

    @property
    def weights(self) -> pd.DataFrame:
        """Solo la matriz de pesos (n_points × activos)."""
        return self.table[self.assets]

    def point(self, idx: int) -> pd.Series:
        """Pesos del portafolio en el punto `idx` de la frontera (Series)."""
        return self.table.loc[idx, self.assets]

    @property
    def minvar_weights(self) -> pd.Series:
        return self.point(self.idx_minvar)

    @property
    def maxsharpe_weights(self) -> pd.Series:
        return self.point(self.idx_maxsharpe)


# =============================================================================
# SOLVER DE UN PUNTO DE LA FRONTERA (barrido de aversión al riesgo γ)
# =============================================================================
#
# En vez de fijar un retorno objetivo con una igualdad (frágil cuando la
# frontera es angosta), resolvemos el problema penalizado:
#
#     max  wᵀμ − γ · wᵀΣw      s.a.  Σw=1, buckets RV/RF, 0≤w≤cap
#
# Al barrer γ de grande (→ mínima varianza) a pequeño (→ máximo retorno) se
# recorre TODA la frontera, y cada solve es factible (solo hay igualdades de
# bucket + cotas). La composición se desplaza de forma suave y continua.

def _solve_gamma(mu_np, S, bounds, base_cons, seeds, gamma,
                 w_ref=None, eta=0.0):
    """
    max wᵀμ − γ·wᵀΣw − η·‖w − w_ref‖²   (η = término de continuidad / homotopía)

    Σ es definida positiva (Ledoit-Wolf + ridge), así que el óptimo es único.
    El término η ancla suavemente cada punto al anterior del barrido: en zonas
    donde varios arreglos de RV son casi equivalentes (p.ej. perfiles muy
    dominados por el FICO), rompe el empate hacia el camino CONTINUO en vez de
    saltar entre esquinas. η pequeño ⇒ sesgo despreciable, solo desempata.
    """
    ref = None if (w_ref is None or eta <= 0) else np.asarray(w_ref, dtype=float)
    def neg_obj(w):
        base = -(w @ mu_np) + gamma * (w @ S @ w)
        if ref is not None:
            base += eta * float((w - ref) @ (w - ref))
        return base
    best = None
    for x0 in seeds:
        if x0 is None:
            continue
        res = minimize(neg_obj, x0, method="SLSQP",
                       bounds=bounds, constraints=base_cons,
                       options={"maxiter": 4000, "ftol": 1e-13})
        w = np.clip(np.asarray(res.x, dtype=float), 0, None)
        w[w < 1e-9] = 0.0
        if w.sum() > EPS:
            w = w / w.sum()
            val = neg_obj(w)
            # El óptimo es único: solo se cambia de semilla si la mejora es
            # MATERIAL. Así el warm-start gana los empates y el camino no salta
            # entre esquinas numéricamente equivalentes.
            tol = 1e-6 * (abs(best[1]) + 1.0) if best is not None else 0.0
            if best is None or val < best[1] - tol:
                best = (w, val)
    return best[0] if best is not None else None


# =============================================================================
# FRONTERA CON PESOS
# =============================================================================

def frontier_with_weights(mu: pd.Series, cov: pd.DataFrame,
                          equity_assets: Sequence[str],
                          fico_assets: Sequence[str],
                          profile: RiskProfile, config: BLConfig,
                          n_points: int = 60,
                          gamma_hi: float = 300.0,
                          gamma_lo: float = 0.20,
                          path_smooth: float = 5e-3) -> FrontierResult:
    """
    Barre la frontera eficiente del perfil y guarda los pesos en cada punto.

    mu, cov   : retornos esperados (anual) y covarianza (anual) — típicamente BL.
    equity_assets / fico_assets : universo de cada bucket.
    profile   : define equity_target / fico_target y el cap por activo RV.
    n_points  : nº de portafolios a lo largo de la frontera (resolución del gráfico).
    gamma_hi  : aversión al riesgo máxima (extremo de mínima varianza).
    gamma_lo  : aversión al riesgo mínima (extremo de máximo retorno).
    path_smooth : η del término de continuidad. Suaviza el camino de composición
                  en zonas planas sin sesgar de forma apreciable. 0 lo desactiva.
    """
    assets = list(mu.index)
    n      = len(assets)
    eq_set = set(equity_assets)
    fi_set = set(fico_assets)

    mu_np = mu.reindex(assets).to_numpy(dtype=float)
    S     = nearest_psd(cov.reindex(index=assets, columns=assets).to_numpy(dtype=float),
                        config.ridge)

    eq_idx = np.array([i for i, a in enumerate(assets) if a in eq_set], dtype=int)
    fi_idx = np.array([i for i, a in enumerate(assets) if a in fi_set], dtype=int)

    bounds = [(0.0, config.max_weight_equity) if a in eq_set else (0.0, 1.0)
              for a in assets]
    base_cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if len(eq_idx):
        base_cons.append({"type": "eq",
                          "fun": lambda w, ix=eq_idx: w[ix].sum() - profile.equity_target})
    if len(fi_idx):
        base_cons.append({"type": "eq",
                          "fun": lambda w, ix=fi_idx: w[ix].sum() - profile.fico_target})

    # Semillas: equiponderado por bucket + sesgo por retorno (evita mínimos locales)
    x0_eq = np.zeros(n)
    if len(eq_idx): x0_eq[eq_idx] = profile.equity_target / len(eq_idx)
    if len(fi_idx): x0_eq[fi_idx] = profile.fico_target   / len(fi_idx)

    x0_tilt = np.zeros(n)
    if len(fi_idx): x0_tilt[fi_idx] = profile.fico_target / len(fi_idx)
    if len(eq_idx):
        order = np.argsort(mu_np[eq_idx])[::-1]
        restante, cap = profile.equity_target, config.max_weight_equity
        for k in order:
            take = min(cap, restante)
            x0_tilt[eq_idx[k]] = take
            restante -= take
            if restante <= EPS:
                break
    # Barrido de γ (aversión al riesgo): grande → mínima varianza,
    # pequeño → máximo retorno. Escala geométrica para muestrear bien ambos
    # extremos. WARM-START: cada γ se siembra con la solución del γ anterior
    # para que el camino de composición sea continuo.
    gammas = np.geomspace(gamma_hi, gamma_lo, n_points)

    rows, prev = [], None
    for g in gammas:
        seeds = [prev, x0_tilt, x0_eq]   # warm-start primero, estructurales de respaldo
        w = _solve_gamma(mu_np, S, bounds, base_cons, seeds, g,
                         w_ref=prev, eta=path_smooth)
        if w is None:
            continue
        prev = w
        ret = float(w @ mu_np)
        vol = float(np.sqrt(max(w @ S @ w, EPS)))
        shp = (ret - config.rf_annual) / vol if vol > EPS else np.nan
        row = {"gamma": float(g), "ret": ret, "vol": vol, "sharpe": shp}
        row.update({a: float(w[i]) for i, a in enumerate(assets)})
        rows.append(row)

    if not rows:
        raise RuntimeError("No se pudo construir ningún punto de la frontera.")

    # Orden final por VOLATILIDAD ascendente: el eje X es el espectro de riesgo,
    # de mínima varianza (izquierda) a máximo riesgo/retorno (derecha). Con el
    # warm-start el camino ya es continuo, así que ordenar solo limpia micro-
    # oscilaciones numéricas. El índice 0..n-1 es la posición en el eje X.
    tbl = pd.DataFrame(rows).sort_values("vol").reset_index(drop=True)
    idx_minvar    = int(tbl["vol"].idxmin())
    idx_maxsharpe = int(tbl["sharpe"].idxmax()) if tbl["sharpe"].notna().any() else idx_minvar

    return FrontierResult(
        table=tbl, assets=assets,
        equity_assets=list(equity_assets), fico_assets=list(fico_assets),
        idx_minvar=idx_minvar, idx_maxsharpe=idx_maxsharpe,
        profile=profile, mu=mu.reindex(assets), cov=cov.reindex(index=assets, columns=assets),
        rf_annual=config.rf_annual,
    )


# =============================================================================
# CONVENIENCIA: μ y Σ (Black-Litterman) listos para la frontera
# =============================================================================

def build_bl_inputs(returns: pd.DataFrame,
                    equity_assets: Sequence[str],
                    forced_assets: dict,
                    profile: RiskProfile,
                    rf_assets: Optional[Sequence[str]] = None,
                    views: Optional[Sequence[View]] = None,
                    config: Optional[BLConfig] = None,
                    benchmark_returns: Optional[pd.Series] = None,
                    views_as_alpha: bool = True):
    """
    Prepara (mu_bl, cov_bl, equity_assets, fico_assets) para frontier_with_weights,
    con la MISMA lógica que optimizer.run_profile (equilibrio + views BL + FICO
    forzado). Devuelve además el universo de cada bucket ya filtrado.
    """
    config    = config or BLConfig()
    views     = list(views or [])
    rf_assets = list(rf_assets or [])

    equity_assets = [a for a in equity_assets if a in returns.columns]
    rf_market     = [a for a in rf_assets if a in returns.columns]
    forced_tickers = list(forced_assets.keys())
    all_rf        = rf_market + forced_tickers

    data_cols = equity_assets + rf_market
    full = inject_forced_assets(returns[data_cols], forced_assets, config.periods_per_year)
    assets = list(full.columns)

    cov = estimate_covariance(full, config.periods_per_year, config.ridge)

    lam   = calibrate_lambda(benchmark_returns, config)
    w_mkt = market_weights(assets, equity_assets, all_rf,
                           profile.equity_target, profile.fico_target)
    pi    = equilibrium_returns(cov, w_mkt, lam, config.rf_annual)
    for ticker, fa in forced_assets.items():
        if ticker in pi.index:
            pi.loc[ticker] = fa.ret_annual

    if views_as_alpha and views:
        anchored = []
        for v in views:
            if v.kind == "absolute" and v.asset in pi.index:
                anchored.append(View(kind="absolute", asset=v.asset,
                                     q=float(pi.loc[v.asset] + v.q),
                                     confidence=v.confidence, name=v.name))
            else:
                anchored.append(v)
        views = anchored

    P, Q, conf, _ = build_views(assets, views)
    ret_bl, sigma_bl = black_litterman(cov, pi, P, Q, conf, config)
    for ticker, fa in forced_assets.items():
        if ticker in ret_bl.index:
            ret_bl.loc[ticker] = fa.ret_annual

    return ret_bl, sigma_bl, equity_assets, all_rf
