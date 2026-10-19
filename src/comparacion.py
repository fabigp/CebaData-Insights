"""
Comparación de familias de modelos con las 3 variables finales y en las mismas 50 particiones,
e importancia de cada variable del modelo final (cuánto empeora el error al quitarla).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.evaluacion import prueba_t_corregida

MODELOS_EXTRA = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1),
    "KNN (vecinos más parecidos)": lambda: make_pipeline(StandardScaler(), GridSearchCV(
        KNeighborsRegressor(), {"n_neighbors": [5, 10, 15, 20, 30], "weights": ["uniform", "distance"]},
        cv=5, scoring="neg_root_mean_squared_error")),
    "GPR (proceso gaussiano)": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
        ConstantKernel() * RBF(length_scale=np.ones(len(config.VARIABLES_MODELO))) + WhiteKernel(),
        normalize_y=True, random_state=0)),
}


def _rmse_por_particion(fabrica, X, y, parts):
    out = []
    for a, b in parts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = fabrica().fit(X[a], y[a]).predict(X[b])
        out.append(float(np.sqrt(np.mean((y[b] - p) ** 2))))
    return np.array(out)


def comparar(tabla, parts, por_particion_f1, por_particion_qrf):
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"].reset_index(drop=True)
    X, y = tr[config.VARIABLES_MODELO].values, tr.RENDIMIENTO_T_HA.values
    rmse = {
        "Promedio general": por_particion_f1.RMSE_promedio_global.values,
        "Promedio por municipio": por_particion_f1.RMSE_promedio_municipio.values,
        "Promedio por zona": por_particion_f1.RMSE_promedio_zona.values,
        "Regresión lineal (modelo final)": por_particion_f1.RMSE_modelo.values,
        "QRF (mediana)": por_particion_qrf.RMSE_qrf.values,
        "Promedio regresión + QRF": por_particion_qrf.RMSE_promedio_reg_qrf.values,
    }
    for nombre, fabrica in MODELOS_EXTRA.items():
        print(f"    {nombre}...")
        rmse[nombre] = _rmse_por_particion(fabrica, X, y, parts)
    filas = []
    for nombre, r in rmse.items():
        d_z, g_z, p_z = prueba_t_corregida(r, rmse["Promedio por zona"])
        d_r, g_r, p_r = prueba_t_corregida(r, rmse["Regresión lineal (modelo final)"])
        filas.append({"modelo": nombre, "RMSE": r.mean(), "RMSE_desv": r.std(ddof=1),
                      "dif_vs_zona": d_z, "gana_a_zona_%": g_z, "p_vs_zona": p_z,
                      "dif_vs_final": d_r, "p_vs_final": p_r})
    t = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    t.loc[t.modelo == "Promedio por zona", ["dif_vs_zona", "gana_a_zona_%", "p_vs_zona"]] = np.nan
    t.loc[t.modelo == "Regresión lineal (modelo final)", ["dif_vs_final", "p_vs_final"]] = np.nan
    return t


def importancia_final(tabla, parts, rmse_final):
    """Aumento del RMSE al quitar cada variable del modelo final (mismas particiones)."""
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"].reset_index(drop=True)
    y = tr.RENDIMIENTO_T_HA.values
    filas = []
    for v in config.VARIABLES_MODELO:
        resto = [c for c in config.VARIABLES_MODELO if c != v]
        r = _rmse_por_particion(LinearRegression, tr[resto].values, y, parts)
        d, _, p = prueba_t_corregida(r, rmse_final)
        filas.append({"variable": v, "RMSE_sin_la_variable": r.mean(), "aumento_RMSE": d, "p": p})
    return pd.DataFrame(filas).sort_values("aumento_RMSE", ascending=False).reset_index(drop=True)
