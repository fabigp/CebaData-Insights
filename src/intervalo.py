"""
Intervalo del 80 % y "segunda opinión" con bosque de regresión por cuantiles (QRF).

Se comparan tres formas de construir el intervalo con las mismas 50 particiones del modelo final:
  A. QRF directo: percentiles 10 y 90 del bosque.
  B. Regresión + semianchos del QRF: el rango del QRF, recentrado en la predicción de la regresión.
  C. Conformal por zona: predicción de la regresión + percentiles 10 y 90 de sus propios errores
     fuera de muestra, calculados por separado en el norte y en el sur.
Un buen intervalo del 80 % debe contener el valor real en ~80 % de los casos en AMBAS zonas.
"""
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

import config

Q_INF, Q_SUP = (1 - config.NIVEL_INTERVALO) / 2, 1 - (1 - config.NIVEL_INTERVALO) / 2


def _qrf():
    return RandomForestQuantileRegressor(n_estimators=config.QRF_ARBOLES, min_samples_leaf=config.QRF_HOJA_MINIMA,
                                         random_state=0, n_jobs=-1)


def validar(tabla, parts):
    """Evalúa los tres intervalos y la coincidencia regresión–QRF. Devuelve tablas y errores por partición."""
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"].reset_index(drop=True)
    X, y, z = tr[config.VARIABLES_MODELO].values, tr.RENDIMIENTO_T_HA.values, tr.zona_norte.values
    filas, errores_reg, por_particion = [], [], []
    for i, (a, b) in enumerate(parts):
        p = LinearRegression().fit(X[a], y[a]).predict(X[b])
        lo, me, hi = _qrf().fit(X[a], y[a]).predict(X[b], quantiles=[Q_INF, 0.5, Q_SUP]).T
        # errores fuera de muestra DENTRO del entrenamiento, para calibrar el intervalo conformal
        e_in = y[a] - cross_val_predict(LinearRegression(), X[a], y[a], cv=KFold(5, shuffle=True, random_state=1))
        c_lo, c_hi = np.empty(len(b)), np.empty(len(b))
        for zz in (0, 1):
            q1, q2 = np.quantile(e_in[z[a] == zz], [Q_INF, Q_SUP])
            s = z[b] == zz
            c_lo[s], c_hi[s] = p[s] + q1, p[s] + q2
        filas.append(pd.DataFrame({
            "particion": i + 1, "real": y[b], "zona_norte": z[b], "regresion": p, "qrf_mediana": me,
            "A_inf": lo, "A_sup": hi, "B_inf": p - (me - lo), "B_sup": p + (hi - me), "C_inf": c_lo, "C_sup": c_hi}))
        errores_reg.append(pd.DataFrame({"zona_norte": z[b], "error": y[b] - p}))
        rm = lambda v: float(np.sqrt(np.mean((y[b] - v) ** 2)))
        por_particion.append({"particion": i + 1, "RMSE_regresion": rm(p), "RMSE_qrf": rm(me),
                              "RMSE_promedio_reg_qrf": rm((p + me) / 2),
                              "MAE_qrf": float(np.mean(np.abs(y[b] - me))),
                              "R2_qrf": float(1 - ((y[b] - me) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum())})
    d = pd.concat(filas, ignore_index=True)
    nombres = {"A": "QRF directo (P10–P90)", "B": "Regresión + semianchos del QRF", "C": "Conformal por zona (elegido)"}
    metodos = []
    for k, nombre in nombres.items():
        dentro = (d.real >= d[f"{k}_inf"]) & (d.real <= d[f"{k}_sup"])
        ancho = d[f"{k}_sup"] - d[f"{k}_inf"]
        metodos.append({"método": nombre, "cobertura_%": 100 * dentro.mean(),
                        "cobertura_norte_%": 100 * dentro[d.zona_norte == 1].mean(),
                        "cobertura_sur_%": 100 * dentro[d.zona_norte == 0].mean(), "ancho_promedio_t_ha": ancho.mean()})
    # ¿La diferencia entre modelos o el ancho del QRF anticipan el error? (correlación de Spearman)
    err = (d.real - d.regresion).abs()
    dif = (d.regresion - d.qrf_mediana).abs()
    coinciden = dif <= config.UMBRAL_COINCIDENCIA
    acuerdo = {
        "% de predicciones en que regresión y QRF coinciden": 100 * coinciden.mean(),
        "error medio cuando coinciden (t/ha)": err[coinciden].mean(),
        "error medio cuando no coinciden (t/ha)": err[~coinciden].mean(),
        "correlación error vs. diferencia entre modelos": stats.spearmanr(err, dif)[0],
        "correlación error vs. ancho del QRF": stats.spearmanr(err, d.A_sup - d.A_inf)[0],
        "p de la correlación con el ancho del QRF": stats.spearmanr(err, d.A_sup - d.A_inf)[1],
    }
    return pd.DataFrame(metodos), pd.Series(acuerdo), pd.DataFrame(por_particion), pd.concat(errores_reg)


def finales(tabla, errores_reg):
    """
    Intervalo conformal por zona para las 59 parcelas (con los errores fuera de muestra de las 50 particiones)
    y predicción del QRF entrenado con las 138 parcelas.
    """
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"]
    te = tabla[tabla.CONJUNTO == "PREDICCION"]
    desplazamientos = {zz: np.quantile(errores_reg.error[errores_reg.zona_norte == zz], [Q_INF, Q_SUP]) for zz in (0, 1)}
    q = _qrf().fit(tr[config.VARIABLES_MODELO].values, tr.RENDIMIENTO_T_HA.values)
    p10, p50, p90 = q.predict(te[config.VARIABLES_MODELO].values, quantiles=[Q_INF, 0.5, Q_SUP]).T
    qrf = pd.DataFrame({"ID_POLIGONO": te.ID_POLIGONO.values, "QRF_P10": p10, "QRF_MEDIANA": p50, "QRF_P90": p90})
    return desplazamientos, qrf
