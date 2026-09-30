# -*- coding: utf-8 -*-
"""
charts.py — Gráficos Plotly para el explorador de frontera (Coril · analista)
==============================================================================

Gráfico central:
    composicion_frontera(...)  → "Composición del portafolio a lo largo de la
    frontera eficiente", un panel por perfil (small multiples). RF en azules
    (arriba), RV en cálidos (abajo), con líneas verticales de mínima varianza
    (punteada) y máximo Sharpe (rayada). Reproduce el gráfico del notebook.

Otros:
    frontera_dispersion(...)   → retorno vs riesgo de cada frontera.
    asignacion_barras(...)     → pesos de un portafolio (allocation + selección).

Todo recibe FrontierResult de frontier.py. Sin estado global.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontier import FrontierResult

# ── Paletas ────────────────────────────────────────────────────────────────
# RV: cálidos (rojos/naranjas) · RF: fríos (azules/teal), como la referencia.
_WARM = ["#7f0000", "#b30000", "#d7301f", "#ef6548", "#fc8d59", "#fdbb84",
         "#fdd49e", "#fee8c8", "#fff7ec", "#990000", "#e34a33", "#f16913"]
_COOL = ["#a6c7e8", "#74a9cf", "#4292c6", "#2171b5", "#08519c", "#6baed6",
         "#3690c0", "#0570b0"]
_MINVAR_LINE   = "#000000"   # punteada negra
_SHARPE_LINE   = "#d6604d"   # rayada rojo/naranja
_SELECT_LINE   = "#2e7d32"   # portafolio seleccionado (verde)


def _color_map(fr: FrontierResult, display=None, sectors=None,
               color_by: str = "asset") -> dict:
    """
    Asigna un color a cada activo. RV en cálidos, RF en fríos.
    color_by: 'asset' (uno por activo) o 'sector' (agrupa RV por sector).
    """
    display = display or (lambda t: t)
    eq = [a for a in fr.assets if a in set(fr.equity_assets)]
    fi = [a for a in fr.assets if a in set(fr.fico_assets)]
    cmap = {}
    # RV ordenada por peso promedio (los grandes con el rojo más intenso)
    w = fr.weights.mean(axis=0)
    eq_sorted = sorted(eq, key=lambda a: -w.get(a, 0))
    for i, a in enumerate(eq_sorted):
        cmap[a] = _WARM[i % len(_WARM)]
    fi_sorted = sorted(fi, key=lambda a: -w.get(a, 0))
    for i, a in enumerate(fi_sorted):
        cmap[a] = _COOL[i % len(_COOL)]
    return cmap


def _ordered_assets(fr: FrontierResult) -> list:
    """RV primero (abajo en el stack), RF al final (arriba), por peso medio."""
    w = fr.weights.mean(axis=0)
    eq = sorted([a for a in fr.assets if a in set(fr.equity_assets)],
                key=lambda a: -w.get(a, 0))
    fi = sorted([a for a in fr.assets if a in set(fr.fico_assets)],
                key=lambda a: -w.get(a, 0))
    return eq + fi   # equity abajo, RF arriba


# =============================================================================
# GRÁFICO CENTRAL: COMPOSICIÓN A LO LARGO DE LA FRONTERA
# =============================================================================

def composicion_frontera(results: dict[str, FrontierResult],
                         display=None,
                         mostrar_lineas: bool = True,
                         seleccion: Optional[dict[str, int]] = None,
                         alto: int = 460) -> go.Figure:
    """
    Un panel por perfil. Área apilada de los pesos (0-100%) a lo largo de la
    frontera; el eje X es la posición en la frontera (riesgo creciente →).

    results    : {nombre_perfil: FrontierResult}
    display    : función ticker → nombre visible (opcional).
    seleccion  : {nombre_perfil: idx} para marcar un portafolio elegido.
    """
    display = display or (lambda t: t)
    names = list(results.keys())
    ncol = len(names)
    fig = make_subplots(rows=1, cols=ncol, shared_yaxes=True,
                        horizontal_spacing=0.03,
                        subplot_titles=[_panel_titulo(n, results[n]) for n in names])

    leyenda_vistos = set()
    for c, name in enumerate(names, start=1):
        fr = results[name]
        tbl = fr.table
        x = list(range(len(tbl)))
        cmap = _color_map(fr, display)
        for a in _ordered_assets(fr):
            y = (tbl[a] * 100).tolist()
            show = a not in leyenda_vistos
            leyenda_vistos.add(a)
            fig.add_trace(go.Scatter(
                x=x, y=y, name=display(a),
                legendgroup=a, showlegend=show,
                mode="lines", line=dict(width=0.5, color=cmap[a]),
                stackgroup=f"g{c}", fillcolor=cmap[a],
                hovertemplate=f"<b>{display(a)}</b><br>%{{y:.1f}}%<extra></extra>",
            ), row=1, col=c)

        if mostrar_lineas:
            _vline(fig, c, fr.idx_minvar, _MINVAR_LINE, "dot", "Mín. var")
            _vline(fig, c, fr.idx_maxsharpe, _SHARPE_LINE, "dash", "Máx. Sharpe")
        if seleccion and name in seleccion:
            _vline(fig, c, seleccion[name], _SELECT_LINE, "solid", "Elegido")

        fig.update_xaxes(title_text="Frontera (riesgo →)", row=1, col=c,
                         showgrid=False, range=[0, max(len(tbl) - 1, 1)])

    fig.update_yaxes(title_text="Peso", row=1, col=1, range=[0, 100],
                     ticksuffix="%", showgrid=True, gridcolor="#eef1f4")
    fig.update_layout(
        title="Composición del portafolio a lo largo de la frontera eficiente",
        height=alto, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.32,
                    xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=55, r=20, t=70, b=90),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def _panel_titulo(name: str, fr: FrontierResult) -> str:
    eq = fr.profile.equity_target
    fi = fr.profile.fico_target
    return f"{name} · {eq:.0%} RV / {fi:.0%} RF"


def _vline(fig, col, xpos, color, dash, texto):
    """Línea vertical en un subplot concreto con anotación arriba."""
    fig.add_vline(x=xpos, line=dict(color=color, width=1.6, dash=dash),
                  row=1, col=col)
    fig.add_annotation(x=xpos, y=1.02, yref="y domain", xref=f"x{col if col>1 else ''}",
                       text=texto, showarrow=False, font=dict(size=9, color=color),
                       yanchor="bottom")


# =============================================================================
# DISPERSIÓN RETORNO vs RIESGO
# =============================================================================

def frontera_dispersion(results: dict[str, FrontierResult],
                        alto: int = 430) -> go.Figure:
    """Retorno esperado vs volatilidad de cada frontera, con puntos clave."""
    fig = go.Figure()
    pal = ["#2171b5", "#d7301f", "#238b45", "#6a51a3", "#cc4c02"]
    for i, (name, fr) in enumerate(results.items()):
        tbl = fr.table
        col = pal[i % len(pal)]
        fig.add_trace(go.Scatter(
            x=tbl["vol"] * 100, y=tbl["ret"] * 100, mode="lines",
            name=name, line=dict(color=col, width=2),
            hovertemplate="Vol %{x:.2f}% · Ret %{y:.2f}%<extra>"+name+"</extra>"))
        mv, ms = fr.idx_minvar, fr.idx_maxsharpe
        fig.add_trace(go.Scatter(
            x=[tbl["vol"].iloc[mv] * 100], y=[tbl["ret"].iloc[mv] * 100],
            mode="markers", marker=dict(color=col, size=9, symbol="circle"),
            name=f"{name} · mín-var", showlegend=False,
            hovertemplate="Mín. var<br>Vol %{x:.2f}% · Ret %{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[tbl["vol"].iloc[ms] * 100], y=[tbl["ret"].iloc[ms] * 100],
            mode="markers", marker=dict(color=col, size=12, symbol="star"),
            name=f"{name} · máx-Sharpe", showlegend=False,
            hovertemplate="Máx. Sharpe<br>Vol %{x:.2f}% · Ret %{y:.2f}%<extra></extra>"))
    fig.update_layout(
        title="Frontera eficiente por perfil (retorno vs riesgo)",
        xaxis_title="Volatilidad anual", yaxis_title="Retorno esperado anual",
        height=alto, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(ticksuffix="%", gridcolor="#eef1f4"),
        yaxis=dict(ticksuffix="%", gridcolor="#eef1f4"),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=55, r=20, t=60, b=60))
    return fig


# =============================================================================
# ASIGNACIÓN DE UN PORTAFOLIO (allocation + security selection)
# =============================================================================

def asignacion_barras(weights: pd.Series, fr: FrontierResult,
                      display=None, titulo: str = "Asignación del portafolio",
                      alto: int = 420) -> go.Figure:
    """Barras horizontales de los pesos (>0.1%), RV cálido / RF frío."""
    display = display or (lambda t: t)
    cmap = _color_map(fr, display)
    w = weights[weights > 0.001].sort_values()
    colors = [cmap.get(a, "#888") for a in w.index]
    fig = go.Figure(go.Bar(
        x=(w.values * 100), y=[display(a) for a in w.index],
        orientation="h", marker=dict(color=colors),
        text=[f"{v:.1%}" for v in w.values], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(
        title=titulo, height=alto, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Peso", ticksuffix="%", gridcolor="#eef1f4"),
        yaxis=dict(title=""), margin=dict(l=10, r=40, t=55, b=40))
    return fig


def contribucion_sector(weights: pd.Series, sectors: dict,
                        alto: int = 380) -> go.Figure:
    """Dona de exposición por sector/clase del portafolio elegido."""
    agg = {}
    for a, wv in weights.items():
        if wv <= 0.001:
            continue
        s = sectors.get(a, "Otros")
        agg[s] = agg.get(s, 0.0) + float(wv)
    s = pd.Series(agg).sort_values(ascending=False)
    fig = go.Figure(go.Pie(
        labels=s.index.tolist(), values=(s.values * 100).tolist(), hole=0.55,
        textinfo="label+percent", hovertemplate="%{label}: %{value:.1f}%<extra></extra>"))
    fig.update_layout(title="Exposición por sector", height=alto,
                      showlegend=False, margin=dict(l=10, r=10, t=55, b=10),
                      paper_bgcolor="white")
    return fig
