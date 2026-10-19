"""
Preguntas bonus del reto, respondidas con las mismas 50 particiones del modelo final.

  1. ¿Qué índice es el más importante?
  2. ¿Qué variable climática influye más?
  3. ¿Qué combinación de variables funciona mejor?

Para cada variable candidata se mide:
  - "por sí sola": qué tanto explica el rendimiento ella sola (R² fuera de muestra);
  - "aporte": cuánto baja el error al agregarla a la base zona + área (regresión lineal), con su valor p;
  - "selección Lasso": % de particiones en que Lasso la elige cuando compite contra TODAS las candidatas.
"""
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.evaluacion import prueba_t_corregida

NO_CANDIDATAS = {"ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO", "Municipio", "Estado"}

# ---------------------------------------------------------------------------------------------
# Nombres legibles para la página
# ---------------------------------------------------------------------------------------------
MESES = {"m05": "mayo", "m06": "junio", "m07": "julio", "m08": "agosto", "m09": "septiembre", "m10": "octubre", "m11": "noviembre"}
SENSOR = {"s2": "Sentinel-2", "pl": "Planet", "ls": "Landsat"}
FENO = {"inicio_doy": "fecha de inicio", "pico_doy": "fecha del pico", "fin_doy": "fecha de fin", "duracion_dias": "duración",
        "ndvi_pico": "NDVI máximo", "amplitud": "amplitud", "vel_subida": "velocidad de crecimiento",
        "vel_bajada": "velocidad de secado", "integral": "verdor acumulado", "pico": "NDVI máximo",
        "duracion": "duración", "inicio": "fecha de inicio"}
CLIMA_VAR = {"prec": "Lluvia", "tmax": "Temp. máxima", "tmin": "Temp. mínima", "gdd": "Grados-día"}
CLIMA_ETAPA = {"mes_inicio": "del mes de siembra/emergencia", "mes_pico": "del mes de máximo verdor", "ciclo": "del ciclo",
               "pico_a_fin": "del pico a la cosecha", "vegetativa": "en etapa vegetativa", "reproductiva": "en etapa reproductiva"}
FIJOS = {
    "zona_norte": "Zona norte", "log_area": "Tamaño de la parcela (log del área)", "lon": "Longitud", "lat": "Latitud",
    "elev_media": "Altitud media", "elev_std": "Variación de altitud", "pend_media": "Pendiente media",
    "tpi_500m": "Posición en el relieve (500 m)", "tpi_2km": "Posición en el relieve (2 km)", "orient_norte": "Orientación al norte",
    "hist_integral_prom": "Verdor acumulado histórico 2022–2024 (NDVI)", "hist_pico_prom": "NDVI máximo histórico 2022–2024",
    "hist_integral_2024": "Verdor acumulado 2024 (NDVI)", "hist_duracion_prom": "Duración histórica del ciclo",
    "s2_ndvi_pico": "NDVI máximo 2025", "s2_doy_pico": "Fecha del NDVI máximo 2025", "s2_ndvi_auc": "Área bajo la curva NDVI 2025",
    "s2_ndvi_caida_pico_oct": "Caída del NDVI del pico a octubre", "het_s2_ndvi_std": "Heterogeneidad del NDVI (Sentinel-2)",
    "het_s2_ndvi_cv": "Heterogeneidad relativa del NDVI (Sentinel-2)", "het_pl_ndvi_std": "Heterogeneidad del NDVI (Planet)",
    "het_pl_ndvi_cv": "Heterogeneidad relativa del NDVI (Planet)", "vi6t_jul_oct": "VI6T Landsat jul–oct",
    "vi6t_n_obs": "Número de imágenes Landsat", "prec_jun_oct": "Lluvia jun–oct", "tmax_jul_oct": "Temp. máxima jul–oct",
    "tmin_jul_oct": "Temp. mínima jul–oct", "amplitud_jul_oct": "Amplitud térmica jul–oct",
    "prec_anomalia_pct": "Anomalía de lluvia 2025 (%)", "clima_gdd_ciclo": "Grados-día del ciclo",
    "rot_anios_ciclo_largo": "Rotación: años con ciclo largo", "rot_anios_sin_cultivo": "Rotación: años sin cultivo",
    "rot_2024_ciclo_largo": "Rotación: ciclo largo en 2024", "rot_dias_verde_prom": "Rotación: días verdes por año",
    "re_lai_pico": "LAI borde rojo (Planet), pico", "re_lai_integral": "LAI borde rojo (Planet), acumulado",
    "re_mtci_pico": "MTCI borde rojo, pico", "re_ireci_pico": "IRECI borde rojo, pico",
    "re_frac_bajo_vigor_pico": "Fracción de bajo vigor en el pico", "evi_integral_llenado": "EVI acumulado en llenado de grano",
    "ndwi_llenado": "NDWI en llenado de grano", "fao_biomasa": "Biomasa estimada (método FAO)",
    "fao_frac_post_pico": "Fracción de verdor después del pico (FAO)", "suelo_arcilla_pct": "Suelo: arcilla (%)",
    "suelo_arena_pct": "Suelo: arena (%)", "suelo_carbono_org_g_kg": "Suelo: carbono orgánico", "suelo_ph": "Suelo: pH",
    "s1_vh_pico": "Radar Sentinel-1 VH, pico", "s1_cr_pico": "Radar Sentinel-1 razón cruzada, pico",
    "s1_cr_reproductiva": "Radar Sentinel-1 razón cruzada, etapa reproductiva",
    "cd_racha_seca_reproductiva": "Racha seca en etapa reproductiva", "cd_dias_helada_ciclo": "Días con helada",
    "cd_dias_calor_reproductiva": "Días de calor en etapa reproductiva", "cd_prec_floracion": "Lluvia en floración",
}


def etiqueta(v):
    if v in FIJOS:
        return FIJOS[v]
    m = re.fullmatch(r"(s2|pl|ls)_(\w+?)_(m\d\d)", v)
    if m:
        return f"{m.group(2).upper()} {SENSOR[m.group(1)]}, {MESES[m.group(3)]}"
    m = re.fullmatch(r"idx_(\w+?)_(max_|min_)?(agsep|oct)", v)
    if m:
        extra = {"max_": " máximo", "min_": " mínimo"}.get(m.group(2) or "", "")
        return f"{m.group(1).upper()}{extra} 2025, {'ago–sep' if m.group(3) == 'agsep' else 'octubre'}"
    m = re.fullmatch(r"pl_(\w+?)_(agsep|oct)", v)
    if m:
        return f"{m.group(1).upper()} Planet 2025, {'ago–sep' if m.group(2) == 'agsep' else 'octubre'}"
    m = re.fullmatch(r"fen_(\w+)", v)
    if m:
        return f"Fenología 2025: {FENO.get(m.group(1), m.group(1))}"
    m = re.fullmatch(r"s2fen25_(\w+)", v)
    if m:
        return f"Fenología 2025 (curva diaria): {FENO.get(m.group(1), m.group(1))}"
    m = re.fullmatch(r"anom25_(\w+)", v)
    if m:
        return f"Anomalía 2025 vs. historial: {FENO.get(m.group(1), m.group(1))}"
    m = re.fullmatch(r"prec_(m\d\d)", v)
    if m:
        return f"Lluvia de {MESES[m.group(1)]}"
    m = re.fullmatch(r"clima_(prec|tmax|tmin)_(\w+)", v)
    if m:
        return f"{CLIMA_VAR[m.group(1)]} {CLIMA_ETAPA.get(m.group(2), m.group(2))}"
    return v


def tipo(v):
    if v in ("zona_norte", "lon", "lat"):
        return "ubicación"
    if v in ("log_area",):
        return "tamaño"
    if v.startswith(("prec_", "tmax_", "tmin_", "amplitud_", "clima_", "cd_")):
        return "clima"
    if v.startswith(("elev_", "pend_", "tpi_", "orient_", "suelo_")):
        return "relieve y suelo"
    if v.startswith("rot_") or v == "vi6t_n_obs":
        return "otros"
    return "índice satelital"


# ---------------------------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------------------------
def _ols():
    return make_pipeline(SimpleImputer(strategy="median"), LinearRegression())


def _evaluar(fabrica, X, y, parts):
    """RMSE por partición y R² con la predicción fuera de muestra promedio."""
    rmse, pred = [], np.full((config.N_REPETICIONES, len(y)), np.nan)
    for i, (a, b) in enumerate(parts):
        p = fabrica().fit(X[a], y[a]).predict(X[b])
        pred[i // config.N_PARTICIONES, b] = p
        rmse.append(float(np.sqrt(np.mean((y[b] - p) ** 2))))
    o = pred.mean(0)
    return np.array(rmse), float(1 - ((y - o) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def preparar(tabla, candidatas):
    """Une las variables del modelo final con las candidatas y descarta columnas constantes."""
    cols = [c for c in candidatas.columns if c not in NO_CANDIDATAS and c not in tabla.columns]
    t = tabla.merge(candidatas[["ID_POLIGONO"] + cols], on="ID_POLIGONO", how="left")
    tr = t[t.CONJUNTO == "ENTRENAMIENTO"].reset_index(drop=True)
    todas = list(config.VARIABLES_MODELO) + cols
    constantes = [c for c in todas if tr[c].nunique(dropna=True) <= 1]
    return tr, [c for c in todas if c not in constantes], constantes


def analizar(tabla, candidatas, parts):
    tr, variables, constantes = preparar(tabla, candidatas)
    y = tr.RENDIMIENTO_T_HA.values
    base = config.BASE_BONUS
    rmse_base, r2_base = _evaluar(_ols, tr[base].values, y, parts)
    rmse_final, r2_final = _evaluar(_ols, tr[config.VARIABLES_MODELO].values, y, parts)

    # 1) Cada variable candidata: por sí sola y sumada a la base zona + área
    print(f"    evaluando {len(variables)} variables candidatas...")
    filas = []
    for v in variables:
        if v in base:
            continue
        _, r2_sola = _evaluar(_ols, tr[[v]].values, y, parts)
        r, r2 = _evaluar(_ols, tr[base + [v]].values, y, parts)
        d, gana, p = prueba_t_corregida(r, rmse_base)
        filas.append({"variable": v, "nombre": etiqueta(v), "tipo": tipo(v), "R2_sola": r2_sola,
                      "RMSE_con_base": r.mean(), "R2_con_base": r2, "aporte_RMSE": d, "gana_%": gana, "p": p})
    cand = pd.DataFrame(filas)

    # 2) Lasso compitiendo con TODAS las candidatas (selección dentro de cada partición)
    print("    selección con Lasso en las 50 particiones...")
    X = tr[variables].values
    elegidas, rmse_lasso = np.zeros(len(variables)), []
    for a, b in parts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LassoCV(cv=5, random_state=0, max_iter=20000)).fit(X[a], y[a])
        elegidas += m[-1].coef_ != 0
        rmse_lasso.append(float(np.sqrt(np.mean((y[b] - m.predict(X[b])) ** 2))))
    frecuencia = pd.Series(100 * elegidas / len(parts), index=variables)
    cand["seleccion_lasso_%"] = cand.variable.map(frecuencia)

    # Orden: primero la frecuencia con que Lasso la elige compitiendo contra todas (más estable),
    # y como desempate el error al sumarla a la base.
    orden = lambda d: d.sort_values(["seleccion_lasso_%", "RMSE_con_base"], ascending=[False, True]).reset_index(drop=True)
    indices = orden(cand[cand.tipo == "índice satelital"])
    clima = orden(cand[cand.tipo == "clima"])
    relieve = orden(cand[cand.tipo == "relieve y suelo"])

    # 3) Combinaciones
    mejor_indice_2025 = indices[~indices.variable.str.startswith("hist_")].iloc[0].variable
    mejor_clima, mejor_relieve = clima.iloc[0].variable, relieve.iloc[0].variable
    final = list(config.VARIABLES_MODELO)
    combos = [
        ("Solo zona", ["zona_norte"]),
        ("Zona + área", ["zona_norte", "log_area"]),
        ("Zona + verdor histórico", ["zona_norte", "hist_integral_prom"]),
        ("Zona + verdor histórico + área (modelo final)", final),
        (f"Modelo final + {etiqueta(mejor_clima)}", final + [mejor_clima]),
        (f"Modelo final + {etiqueta(mejor_indice_2025)}", final + [mejor_indice_2025]),
        (f"Modelo final + {etiqueta(mejor_relieve)}", final + [mejor_relieve]),
        ("Modelo final + mejor clima + mejor índice 2025", final + [mejor_clima, mejor_indice_2025]),
    ]
    # también la variable que más baja el error por sí misma, si no es la misma que elige Lasso
    por_aporte = lambda d: d.sort_values("RMSE_con_base").iloc[0].variable
    for v in (por_aporte(indices[~indices.variable.str.startswith("hist_")]), por_aporte(clima)):
        if v not in (mejor_indice_2025, mejor_clima):
            combos.insert(-1, (f"Modelo final + {etiqueta(v)}", final + [v]))
    filas = []
    for nombre, cols in combos:
        r, r2 = _evaluar(_ols, tr[cols].values, y, parts)
        filas.append((nombre, len(cols), r, r2))
    filas.append((f"Todas las variables ({len(variables)}, Lasso elige)", len(variables), np.array(rmse_lasso), np.nan))
    comb = []
    for nombre, n, r, r2 in filas:
        d, gana, p = prueba_t_corregida(r, rmse_final)
        comb.append({"combinación": nombre, "n_variables": n, "RMSE": r.mean(), "R2": r2,
                     "dif_vs_final": d, "gana_a_final_%": gana, "p_vs_final": p})
    comb = pd.DataFrame(comb)
    comb.loc[comb.combinación.str.contains("(modelo final)", regex=False), ["dif_vs_final", "gana_a_final_%", "p_vs_final"]] = np.nan

    # Respuestas (regla: solo se acepta una mejora si es comprobada, p < ALFA)
    mejor_i, mejor_c = indices.iloc[0], clima.iloc[0]
    mejoras = comb[(comb.p_vs_final < config.ALFA) & (comb.dif_vs_final < 0)]
    elegida = mejoras.sort_values("RMSE").iloc[0] if len(mejoras) else comb[comb.combinación.str.contains("(modelo final)", regex=False)].iloc[0]
    respuestas = {
        "indice": {"variable": mejor_i.variable, "nombre": mejor_i.nombre, "aporte_RMSE": mejor_i.aporte_RMSE,
                   "p": mejor_i.p, "seleccion_lasso_%": mejor_i["seleccion_lasso_%"], "R2_sola": mejor_i.R2_sola,
                   "comprobado": bool(mejor_i.p < config.ALFA)},
        "clima": {"variable": mejor_c.variable, "nombre": mejor_c.nombre, "aporte_RMSE": mejor_c.aporte_RMSE,
                  "p": mejor_c.p, "seleccion_lasso_%": mejor_c["seleccion_lasso_%"], "R2_sola": mejor_c.R2_sola,
                  "comprobado": bool(mejor_c.p < config.ALFA),
                  "engañosa": {"nombre": clima.loc[clima.R2_sola.idxmax(), "nombre"], "R2_sola": clima.R2_sola.max(),
                               "aporte_RMSE": clima.loc[clima.R2_sola.idxmax(), "aporte_RMSE"],
                               "p": clima.loc[clima.R2_sola.idxmax(), "p"]},
                  "n_clima_comprobadas": int(((clima.p < config.ALFA) & (clima.aporte_RMSE < 0)).sum()),
                  "n_clima": len(clima)},
        "combinacion": {"nombre": elegida.combinación, "RMSE": elegida.RMSE, "R2": elegida.R2,
                        "mejor_RMSE_sin_comprobar": comb.RMSE.min(),
                        "p_menor_RMSE": comb.sort_values("RMSE").iloc[0].p_vs_final,
                        "combinacion_menor_RMSE": comb.sort_values("RMSE").iloc[0].combinación},
        "base": {"variables": base, "RMSE": rmse_base.mean(), "R2": r2_base},
        "final": {"RMSE": rmse_final.mean(), "R2": r2_final},
        "n_candidatas": len(variables), "constantes_descartadas": constantes,
        "lasso_todas": {"RMSE": float(np.mean(rmse_lasso)), "p_vs_final": comb.iloc[-1].p_vs_final},
    }
    return {"indices": indices, "clima": clima, "relieve": relieve, "combinaciones": comb,
            "frecuencia_lasso": frecuencia.sort_values(ascending=False), "respuestas": respuestas}
