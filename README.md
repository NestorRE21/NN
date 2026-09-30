# Coril · Explorador de Frontera Eficiente por Perfil

Herramienta **para analista** (asset allocation + security selection). El centro
es el gráfico **"Composición del portafolio a lo largo de la frontera eficiente"**
por perfil: permite descubrir portafolios recorriendo la frontera —del de mínima
varianza al de máximo retorno— manteniendo la **misma dinámica de perfiles** y los
**mismos porcentajes de FICO / renta fija** del mandato comercial de Coril.

## Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run app_analista.py
```

Arranca en **modo Demo** (datos sintéticos realistas, sin credenciales): sirve
para explorar la frontera de inmediato. Para datos reales, cambia a **modo Real**
en la barra lateral y configura los secrets (abajo).

## Despliegue en Streamlit Cloud (desde GitHub)

1. Sube esta carpeta a un repo de GitHub.
2. En share.streamlit.io → *New app* → apunta a `app_analista.py`.
3. *Settings → Secrets*: pega el contenido de `.streamlit/secrets.toml.example`
   con tus credenciales reales de la API de Coril.
4. Deploy. (El modo Demo funciona aunque no configures secrets.)

## Estructura

| Archivo | Rol | Origen |
|---|---|---|
| `app_analista.py` | App Streamlit (UI de analista) | **nuevo** |
| `frontier.py` | Frontera eficiente **con pesos por punto** (núcleo del gráfico) | **nuevo** |
| `charts.py` | Gráficos Plotly (composición, dispersión, asignación) | **nuevo** |
| `profiles.py` | Perfiles (30/70…70/30) y FICOs (Soles 7 % / Dólares 6 %) | **nuevo** |
| `data.py` | Datos: demo sintético + real (Coril API + yfinance) | **nuevo** |
| `optimizer.py` | Motor Black-Litterman (equilibrio, BL, mean-variance) | reutilizado |
| `projections.py` | Monte Carlo y stress testing | reutilizado |
| `coril_api.py` | Cliente API de market data de Coril (RV) | reutilizado |
| `bvl_data.py` | Cliente API BVL (alternativo) | reutilizado |
| `bvl_catalog.py` | Universo BVL: sector, nombre, moneda (~219 tickers) | reutilizado |

## Perfiles y FICOs (idénticos al comercial)

| Perfil | RV | RF |
|---|---|---|
| Conservador | 30 % | 70 % |
| Moderado-bajo | 40 % | 60 % |
| Moderado | 50 % | 50 % |
| Crecimiento | 60 % | 40 % |
| Agresivo | 70 % | 30 % |

FICOs (renta fija forzada): **FICO Coril Soles 7 %**, **FICO Coril Dólares 6 %**
(beta 0.30, vol 1 %). Se inyectan al bucket de RF junto con los ETFs de bonos.

## Cómo se construye la frontera con pesos

Para cada perfil se resuelve, barriendo la aversión al riesgo γ:

```
max  wᵀμ − γ·wᵀΣw      s.a.  Σw = 1,  Σw_RV = %RV,  Σw_RF = %RF,  0 ≤ w_RV ≤ cap
```

con μ y Σ de **Black-Litterman** (equilibrio + vistas opcionales Grinold-Kahn) y
el FICO con su retorno forzado. Al variar γ de grande (mínima varianza) a pequeño
(máximo retorno) se recorre la frontera; en cada punto se guardan los pesos. El
**eje X del gráfico es el espectro de riesgo** (volatilidad ascendente): a la
izquierda el portafolio de mínima varianza, a la derecha el de máximo retorno.
Las líneas verticales marcan **mínima varianza** (punteada) y **máximo Sharpe**
(rayada). Se usa warm-start + un término de continuidad para que la composición
morfe de forma suave.

> Nota: el notebook original ordenaba el eje por retorno objetivo (dejando el
> min-var en el interior). Aquí se ordena por volatilidad, que es el eje de
> riesgo canónico y más legible para el analista. Es un cambio de presentación,
> no de la matemática.

## Parámetros (barra lateral → *Parámetros del motor*)

- **Años de histórico**, **Tope por activo RV** (`max_weight_equity`), **τ**.
- **Puntos de la frontera** (resolución del gráfico).
- **Suavizado del camino (η)**.
- **Vistas automáticas (Grinold-Kahn)** con su **IC**.
