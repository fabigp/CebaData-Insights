"""Construcción de las 3 variables del modelo final."""
import numpy as np
import pandas as pd

import config


def verdor_acumulado(serie_ndvi, anio):
    """
    Verdor acumulado de una temporada (área bajo la curva de NDVI por encima del suelo desnudo).

    Pasos:
      1. Mediana diaria de NDVI entre 1 de abril y 30 de noviembre.
      2. Interpolación a una curva diaria y suavizado con mediana móvil de 15 días.
      3. Base = percentil 10 de abril-junio (suelo sin cultivo); pico = máximo desde junio.
      4. Inicio y fin del ciclo = días en que la curva cruza base + 50 % de la amplitud.
      5. Integral = suma diaria de (NDVI - base) entre inicio y fin, dividida entre 100.
    Devuelve NaN si la temporada tiene menos de 8 fechas limpias.
    """
    ini, fin = f"{anio}-{config.TEMPORADA_INICIO}", f"{anio}-{config.TEMPORADA_FIN}"
    g = serie_ndvi[(serie_ndvi.index >= ini) & (serie_ndvi.index <= fin)]
    g = g.groupby(level=0).median()
    if len(g) < 8:
        return np.nan
    dias = pd.date_range(ini, fin)
    curva = (g.reindex(dias).interpolate(limit_direction="both")
             .rolling(config.VENTANA_SUAVIZADO_DIAS, center=True, min_periods=1).median())
    doy, v = curva.index.dayofyear.values, curva.values
    base = np.percentile(v[doy <= 181], 10)
    i_pico = np.argmax(np.where(doy >= 152, v, -9))
    umbral = base + config.UMBRAL_FENOLOGICO * (v[i_pico] - base)
    antes = np.where(v[:i_pico] < umbral)[0]
    despues = np.where(v[i_pico:] < umbral)[0]
    if not len(antes) or not len(despues):
        return 0.0
    d_ini, d_fin = doy[antes[-1] + 1], doy[i_pico + despues[0]]
    dentro = (doy >= d_ini) & (doy <= d_fin)
    return float((v[dentro] - base).clip(0).sum() / 100)


def historial_verdor(s2):
    """Promedio del verdor acumulado de 2022, 2023 y 2024 para cada parcela."""
    limpias = s2[(s2.porcentaje_nubosidad <= config.NUBOSIDAD_MAXIMA) & s2.ndvi_promedio.notna()]
    filas = []
    for pid, g in limpias.groupby("ID_POLIGONO"):
        serie = g.set_index("fecha").ndvi_promedio.sort_index()
        valores = {a: verdor_acumulado(serie, a) for a in config.ANIOS_HISTORIAL}
        filas.append({"ID_POLIGONO": pid, **{f"verdor_{a}": v for a, v in valores.items()},
                      "hist_integral_prom": np.nanmean(list(valores.values()))})
    return pd.DataFrame(filas)


def construir_tabla(rend, parcelas, s2):
    """Una fila por parcela con las variables del modelo y el rendimiento."""
    t = rend.merge(parcelas, on="ID_POLIGONO").merge(historial_verdor(s2), on="ID_POLIGONO")
    t["zona_norte"] = t.Municipio.isin(config.MUNICIPIOS_NORTE).astype(int)
    t["zona"] = np.where(t.zona_norte == 1, "Norte", "Sur")
    t["log_area"] = np.log(t.AREA_HA)
    return t.sort_values("ID_POLIGONO").reset_index(drop=True)


def ndvi_mensual(s2, anio=None):
    """NDVI promedio de cada mes del año del ciclo (observaciones con nubosidad ≤ 10 %). 12 valores por parcela."""
    anio = anio or config.ANIO_CICLO
    s = s2[(s2.porcentaje_nubosidad <= config.NUBOSIDAD_MAXIMA) & s2.ndvi_promedio.notna() & (s2.fecha.dt.year == anio)]
    t = s.groupby(["ID_POLIGONO", s.fecha.dt.month]).ndvi_promedio.mean().unstack()
    return t.reindex(columns=range(1, 13))
