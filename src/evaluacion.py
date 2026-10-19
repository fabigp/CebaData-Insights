"""Validación cruzada repetida, líneas base y prueba t corregida (Nadeau y Bengio, 2003)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

import config


def _rmse(real, pred):
    return float(np.sqrt(np.mean((real - pred) ** 2)))


def particiones(n):
    rkf = RepeatedKFold(n_splits=config.N_PARTICIONES, n_repeats=config.N_REPETICIONES, random_state=config.SEMILLA)
    return list(rkf.split(np.arange(n)))


def prueba_t_corregida(errores_a, errores_b):
    """Compara dos modelos en las mismas particiones. Devuelve diferencia media, % de victorias de A y p."""
    d = np.asarray(errores_a) - np.asarray(errores_b)
    k = len(d)
    if np.allclose(d, 0):                                     # mismo modelo: no hay nada que comparar
        return 0.0, float("nan"), float("nan")
    proporcion_prueba = 1 / (config.N_PARTICIONES - 1)       # tamaño de prueba / tamaño de entrenamiento
    t = d.mean() / np.sqrt((1 / k + proporcion_prueba) * d.var(ddof=1))
    return float(d.mean()), float((d < 0).mean() * 100), float(2 * stats.t.sf(abs(t), k - 1))


def validar(tabla):
    """
    Evalúa el modelo final y las líneas base con exactamente las mismas particiones.
    Devuelve: métricas resumen, errores por partición y predicciones fuera de muestra por parcela.
    """
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"].reset_index(drop=True)
    X, y = tr[config.VARIABLES_MODELO].values, tr.RENDIMIENTO_T_HA.values
    zona, muni = tr.zona_norte.values, tr.Municipio.values
    parts = particiones(len(tr))
    pred = np.full((config.N_REPETICIONES, len(tr)), np.nan)
    por_particion = []
    for i, (a, b) in enumerate(parts):
        p = LinearRegression().fit(X[a], y[a]).predict(X[b])
        pred[i // config.N_PARTICIONES, b] = p
        p_global = np.full(len(b), y[a].mean())
        p_zona = pd.Series(zona[b]).map(pd.Series(y[a]).groupby(zona[a]).mean()).values
        p_muni = pd.Series(muni[b]).map(pd.Series(y[a]).groupby(muni[a]).mean()).fillna(y[a].mean()).values
        sct = ((y[b] - y[b].mean()) ** 2).sum()
        por_particion.append({
            "particion": i + 1,
            "RMSE_modelo": _rmse(y[b], p), "MAE_modelo": float(np.mean(np.abs(y[b] - p))),
            "R2_modelo": float(1 - ((y[b] - p) ** 2).sum() / sct),
            "RMSE_promedio_global": _rmse(y[b], p_global), "RMSE_promedio_zona": _rmse(y[b], p_zona),
            "RMSE_promedio_municipio": _rmse(y[b], p_muni)})
    P = pd.DataFrame(por_particion)
    oof = pred.mean(axis=0)                       # predicción fuera de muestra (promedio de 10 repeticiones)
    e = y - oof
    cortes = np.quantile(y, [1 / 3, 2 / 3])
    m = {
        "RMSE (t/ha)": P.RMSE_modelo.mean(),
        "RMSE: desviación entre particiones": P.RMSE_modelo.std(),
        "MAE (t/ha)": P.MAE_modelo.mean(),
        "R² promedio por partición": P.R2_modelo.mean(),
        "R² fuera de muestra (todas las parcelas)": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "RMSE relativo (% del promedio)": 100 * P.RMSE_modelo.mean() / y.mean(),
        "MAPE (%)": 100 * np.mean(np.abs(e) / y),
        "Sesgo (real − predicho, t/ha)": e.mean(),
        "% de parcelas a ±0.5 t/ha": 100 * np.mean(np.abs(e) <= 0.5),
        "% de parcelas a ±1.0 t/ha": 100 * np.mean(np.abs(e) <= 1.0),
        "Correlación real vs. predicho": np.corrcoef(y, oof)[0, 1],
        "Acierto de categoría bajo/medio/alto (%)": 100 * np.mean(np.digitize(y, cortes) == np.digitize(oof, cortes)),
        "RMSE zona norte": _rmse(y[zona == 1], oof[zona == 1]),
        "RMSE zona sur": _rmse(y[zona == 0], oof[zona == 0]),
    }
    comparaciones = []
    for base in ("promedio_global", "promedio_municipio", "promedio_zona"):
        d, gana, p = prueba_t_corregida(P.RMSE_modelo, P[f"RMSE_{base}"])
        comparaciones.append({"línea base": base.replace("_", " "), "RMSE línea base": P[f"RMSE_{base}"].mean(),
                              "RMSE modelo": P.RMSE_modelo.mean(), "diferencia": d,
                              "% particiones que gana el modelo": gana, "p (t corregida)": p})
    detalle = tr[["ID_POLIGONO", "Municipio", "zona", "RENDIMIENTO_T_HA"]].assign(
        PREDICCION_FUERA_DE_MUESTRA=oof, ERROR=e)
    return pd.Series(m), pd.DataFrame(comparaciones), P, detalle
