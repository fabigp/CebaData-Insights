"""Entrenamiento del modelo final con las 138 parcelas y predicción de las 59."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def entrenar(tabla):
    """Regresión lineal por mínimos cuadrados; devuelve el modelo y su tabla de coeficientes."""
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"]
    X = sm.add_constant(tr[config.VARIABLES_MODELO])
    modelo = sm.OLS(tr.RENDIMIENTO_T_HA, X).fit()
    ic = modelo.conf_int()
    coef = pd.DataFrame({
        "coeficiente": modelo.params, "error_estandar": modelo.bse,
        "ic95_inferior": ic[0], "ic95_superior": ic[1], "p": modelo.pvalues})
    desv = tr[config.VARIABLES_MODELO].std()
    coef.loc[config.VARIABLES_MODELO, "efecto_por_1_desv_estandar"] = coef.loc[config.VARIABLES_MODELO, "coeficiente"] * desv
    ajuste = {"R² entrenamiento": modelo.rsquared, "R² ajustado": modelo.rsquared_adj,
              "parcelas de entrenamiento": int(modelo.nobs)}
    return modelo, coef, ajuste


def predecir(modelo, tabla):
    """Predicción para las parcelas del conjunto PREDICCION."""
    te = tabla[tabla.CONJUNTO == "PREDICCION"].copy()
    te["RENDIMIENTO_PREDICHO_T_HA"] = modelo.predict(sm.add_constant(te[config.VARIABLES_MODELO], has_constant="add"))
    return te
