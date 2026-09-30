# -*- coding: utf-8 -*-
"""
bvl_data.py — Conexión a la API de la BVL (Bolsa de Valores de Lima)
=====================================================================
Reemplaza yfinance como fuente de datos.

Flujo:
  1. get_token()  → OAuth2 Client Credentials → access_token
  2. get_history(ticker, start, end, token) → precios históricos

Las credenciales se leen de st.secrets (nunca hardcodeadas).
"""
from __future__ import annotations
import requests
import pandas as pd
import numpy as np

# Endpoints
TOKEN_URL = "https://auth-bvl-prod.bvl.com.pe/oauth2/token"
DATA_URL  = "https://api.bvl.com.pe/core/v1/historical-stock-quotes"


def get_token(client_id: str, client_secret: str) -> str | None:
    """
    Obtiene un access_token vía OAuth2 Client Credentials.
    Intenta dos métodos: credenciales en el body (según doc BVL) y,
    si falla, Basic Auth en el header (fallback estándar OAuth2).
    """
    # Método 1: credenciales en el body (lo que dice la doc de la BVL)
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception as e:
        print(f"Método body falló: {e}")

    # Método 2: Basic Auth en header (fallback)
    try:
        resp = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        print(f"Método basic-auth devolvió {resp.status_code}")
    except Exception as e:
        print(f"Método basic-auth falló: {e}")

    return None


def get_history(ticker: str, start: str, end: str,
                token: str, api_key: str) -> pd.Series | None:
    """
    Descarga precios de cierre históricos de un ticker.
    start/end en formato 'YYYYMMDD' (ej: '20210707').
    Devuelve una Serie de precios de cierre indexada por fecha, o None.
    """
    try:
        resp = requests.get(
            DATA_URL,
            params={"start-date": start, "end-date": end, "ticker": ticker},
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {token}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        body = data.get("body", [])
        if not body:
            return None
        df = pd.DataFrame(body)
        if "date" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df[df["close"] > 0].dropna(subset=["close"])
        s = df.set_index("date")["close"].sort_index()
        s.name = ticker
        # Eliminar duplicados de fecha (quedarse con el último)
        s = s[~s.index.duplicated(keep="last")]

        # ── Filtro de outliers por mediana móvil ──
        # Un precio que se desvía >25% de la mediana de su entorno (ventana 11
        # días) es casi seguro un dato erróneo (operación mínima a precio raro,
        # error de captura). Se reemplaza por la mediana local, dejando la
        # tendencia real intacta. Esto imita el precio "ajustado" de TradingView.
        if len(s) >= 11:
            med = s.rolling(11, center=True, min_periods=3).median()
            desv = (s - med).abs() / med.replace(0, np.nan)
            outliers = desv > 0.25
            if outliers.any():
                s = s.mask(outliers, med)
                s = s.ffill().bfill()
        return s
    except Exception as e:
        print(f"Error descargando {ticker}: {e}")
        return None


def debug_history(ticker: str, start: str, end: str, token: str, api_key: str) -> dict:
    """
    Versión de diagnóstico: devuelve el detalle crudo de la respuesta HTTP
    para entender por qué no llegan datos.
    """
    info = {"ticker": ticker, "start": start, "end": end}
    try:
        resp = requests.get(
            DATA_URL,
            params={"start-date": start, "end-date": end, "ticker": ticker},
            headers={"x-api-key": api_key, "Authorization": f"Bearer {token}"},
            timeout=30,
        )
        info["status_code"] = resp.status_code
        info["url"] = resp.url
        try:
            j = resp.json()
            info["json_keys"] = list(j.keys()) if isinstance(j, dict) else "no es dict"
            body = j.get("body") if isinstance(j, dict) else None
            info["body_type"] = type(body).__name__
            info["body_len"] = len(body) if body else 0
            if body and len(body) > 0:
                info["primer_registro"] = body[0]
            else:
                # Mostrar el JSON completo si no hay body
                info["respuesta_completa"] = str(j)[:500]
        except Exception as je:
            info["json_error"] = str(je)
            info["texto_crudo"] = resp.text[:500]
        return info
    except Exception as e:
        info["excepcion"] = str(e)
        return info


def get_history_chunked(ticker, start, end, token, api_key, chunk_years=4):
    """
    Descarga histórico en tramos de chunk_years años y los une.
    Evita el límite 'Period not allowed' de la API (máx ~5 años por llamada).
    start/end en formato 'YYYYMMDD'.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    trozos = []
    cursor = start_dt
    while cursor < end_dt:
        tramo_fin = min(cursor + pd.DateOffset(years=chunk_years), end_dt)
        s = get_history(ticker,
                        cursor.strftime("%Y%m%d"),
                        tramo_fin.strftime("%Y%m%d"),
                        token, api_key)
        if s is not None and len(s) > 0:
            trozos.append(s)
        cursor = tramo_fin + pd.Timedelta(days=1)
    if not trozos:
        return None
    # Unir todos los tramos, quitar duplicados de fecha
    full = pd.concat(trozos)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    full.name = ticker
    return full


def ajustar_saltos(serie, umbral=0.30):
    """
    Corrige saltos anómalos de nivel (splits, cambios de nominal, empalmes de
    tramos) en una serie de PRECIOS. Cuando entre dos días consecutivos el
    precio salta más de `umbral` (ej. 30%), reescala hacia atrás todo el
    histórico anterior por el ratio del salto, dejando la serie continua.
    Es el ajuste estándar retroactivo de splits.
    """
    s = serie.dropna().sort_index().copy()
    if len(s) < 3:
        return s
    vals = s.values.astype(float)
    n = len(vals)
    # Recorrer de atrás hacia adelante detectando saltos
    factor = 1.0
    ajustada = vals.copy()
    for i in range(n - 1, 0, -1):
        prev = vals[i-1]; cur = vals[i]
        if prev <= 0 or cur <= 0:
            continue
        ratio = cur / prev
        # salto brusco (caída o subida) → probable split/nominal/empalme
        if ratio < (1 - umbral) or ratio > 1/(1 - umbral):
            # reescalar todo lo ANTERIOR al salto para empalmar
            ajustada[:i] = ajustada[:i] * ratio
    return pd.Series(ajustada, index=s.index, name=s.name)


def download_prices(tickers, start, end, client_id, client_secret, api_key):
    """
    Descarga precios de varios tickers y devuelve un DataFrame de log-retornos.
    Corrige splits individuales y elimina huecos de datos que afectan a todo
    el mercado a la vez (fechas donde la API no tenía datos y el ffill inventó
    un movimiento artificial simultáneo en todas las acciones).
    """
    token = get_token(client_id, client_secret)
    if token is None:
        return None

    series = {}
    for tk in tickers:
        s = get_history_chunked(tk, start, end, token, api_key)
        if s is not None and len(s) > 5:
            series[tk] = ajustar_saltos(s, umbral=0.30)

    if not series:
        return None

    prices = pd.DataFrame(series).sort_index()

    # Índice común solo con días donde AL MENOS la mitad de las acciones
    # tienen dato REAL (no forward-fill). Esto elimina feriados peruanos y
    # huecos de la API donde el ffill crearía movimientos artificiales.
    tiene_dato = prices.notna()
    min_activos = max(1, int(np.ceil(prices.shape[1] * 0.5)))
    dias_validos = tiene_dato.sum(axis=1) >= min_activos
    prices = prices.loc[dias_validos]

    # Ahora sí, forward-fill los huecos individuales restantes
    prices = prices.ffill()

    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)

    # Detectar movimientos anómalos SIMULTÁNEOS (mismo día, muchas acciones):
    # si en un día >60% de las acciones se mueven más de ±20%, es un artefacto
    # de datos, no un evento de mercado. Se neutraliza esa fila.
    if log_ret.shape[1] >= 3:
        anom = (log_ret.abs() > 0.20)
        frac_anom = anom.sum(axis=1) / log_ret.shape[1]
        dias_malos = frac_anom > 0.60
        if dias_malos.any():
            log_ret.loc[dias_malos] = 0.0

    # Red de seguridad: saltos individuales residuales muy grandes
    log_ret = log_ret.mask(log_ret.abs() > 0.40, 0.0)
    log_ret = log_ret.dropna(how="all")
    return log_ret
