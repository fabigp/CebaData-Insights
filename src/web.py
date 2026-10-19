"""Arma datos_parcelas.js: el único archivo de datos que lee la página (window.CEBA_DATA)."""
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.bonus import etiqueta


def _r(v, d=3):
    """Redondea y convierte NaN en null para JSON."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), d)
    return v


def _limpiar(o):
    """Recorre dicts y listas: redondea números, NaN → null, booleanos de numpy → bool."""
    if isinstance(o, dict):
        return {k: _limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return _r(o)


def _redondear_coords(c, d=6):
    return [_redondear_coords(x, d) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], d), round(c[1], d)]


def _tabla(df, d=3):
    return [{k: _r(v, d) if not isinstance(v, str) else v for k, v in fila.items()} for fila in df.to_dict("records")]


def construir(tabla, geom, ndvi, modelo, coef, metricas, comparaciones_f1, detalle, desplazamientos, qrf,
              metodos_intervalo, acuerdo, modelos, importancia, bonus):
    tr = tabla[tabla.CONJUNTO == "ENTRENAMIENTO"]
    medias = tr[config.VARIABLES_MODELO].mean()
    promedio_entrenamiento = float(tr.RENDIMIENTO_T_HA.mean())
    cortes = np.quantile(tr.RENDIMIENTO_T_HA, [1 / 3, 2 / 3])
    ref_zona = tr.groupby("zona").RENDIMIENTO_T_HA.mean()
    ref_mun = tr.groupby("Municipio").RENDIMIENTO_T_HA.agg(["mean", "count"])
    oof = detalle.set_index("ID_POLIGONO").PREDICCION_FUERA_DE_MUESTRA
    qrf = qrf.set_index("ID_POLIGONO")
    b = modelo.params
    nombres_contrib = {"zona_norte": "zona", "hist_integral_prom": "historial", "log_area": "area"}

    feats = []
    gj = json.loads(geom.merge(tabla[["ID_POLIGONO"]], on="ID_POLIGONO").to_json(drop_id=True))
    geo_por_id = {f["properties"]["ID_POLIGONO"]: f["geometry"] for f in gj["features"]}
    for _, p in tabla.iterrows():
        pid = p.ID_POLIGONO
        es_pred = p.CONJUNTO == "PREDICCION"
        g = geo_por_id[pid]
        g = {"type": g["type"], "coordinates": _redondear_coords(g["coordinates"])}
        props = {
            "id": pid, "municipio": p.Municipio, "estado": p.Estado, "zona": p.zona, "area": _r(p.AREA_HA, 2),
            "conjunto": p.CONJUNTO, "rendimiento": None if es_pred else _r(p.RENDIMIENTO_T_HA, 2),
            "verdor": {str(a): _r(p[f"verdor_{a}"]) for a in config.ANIOS_HISTORIAL} | {"prom": _r(p.hist_integral_prom)},
            "ref_zona": _r(ref_zona[p.zona], 2),
            "ref_municipio": _r(ref_mun["mean"].get(p.Municipio, np.nan), 2),
            "n_municipio": int(ref_mun["count"].get(p.Municipio, 0)),
            "ndvi": [_r(v) for v in (ndvi.loc[pid].values if pid in ndvi.index else [np.nan] * 12)],
        }
        if es_pred:
            x = p[config.VARIABLES_MODELO]
            pred = float(b["const"] + (b[config.VARIABLES_MODELO] * x).sum())
            lo, hi = desplazamientos[int(p.zona_norte)]
            q = qrf.loc[pid]
            props |= {
                "prediccion": _r(pred), "int_inf": _r(pred + lo), "int_sup": _r(pred + hi),
                "categoria": ["Bajo", "Medio", "Alto"][int(np.digitize(pred, cortes))],
                "qrf_p10": _r(q.QRF_P10), "qrf_mediana": _r(q.QRF_MEDIANA), "qrf_p90": _r(q.QRF_P90),
                "coinciden": bool(abs(pred - q.QRF_MEDIANA) <= config.UMBRAL_COINCIDENCIA),
                "contrib": {nombres_contrib[v]: _r(b[v] * (x[v] - medias[v])) for v in config.VARIABLES_MODELO},
            }
        else:
            props |= {"pred_validacion": _r(oof[pid]), "error_validacion": _r(p.RENDIMIENTO_T_HA - oof[pid])}
        feats.append({"type": "Feature", "geometry": g, "properties": props})

    m = metricas
    comp = comparaciones_f1.set_index("línea base")
    metodo_elegido = metodos_intervalo[metodos_intervalo["método"].str.contains("elegido")].iloc[0]
    metodo_qrf = metodos_intervalo[metodos_intervalo["método"].str.startswith("QRF")].iloc[0]
    resp = bonus["respuestas"]
    datos = {
        "type": "FeatureCollection",
        "features": feats,
        "meta": {"generado": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "parcelas": len(tabla), "entrenamiento": len(tr), "prediccion": int((tabla.CONJUNTO == "PREDICCION").sum()),
                 "particiones": f"{config.N_PARTICIONES} × {config.N_REPETICIONES}"},
        "metricas": {
            "rmse": _r(m["RMSE (t/ha)"]), "rmse_desv": _r(m["RMSE: desviación entre particiones"]),
            "mae": _r(m["MAE (t/ha)"]), "r2": _r(m["R² fuera de muestra (todas las parcelas)"]),
            "rmse_rel": _r(m["RMSE relativo (% del promedio)"], 1), "mape": _r(m["MAPE (%)"], 1),
            "sesgo": _r(m["Sesgo (real − predicho, t/ha)"]), "pct_05": _r(m["% de parcelas a ±0.5 t/ha"], 1),
            "pct_10": _r(m["% de parcelas a ±1.0 t/ha"], 1), "acierto_categoria": _r(m["Acierto de categoría bajo/medio/alto (%)"], 1),
            "rmse_norte": _r(m["RMSE zona norte"]), "rmse_sur": _r(m["RMSE zona sur"]),
            "r2_entrenamiento": _r(modelo.rsquared), "promedio_entrenamiento": _r(promedio_entrenamiento),
            "rmse_zona": _r(comp.loc["promedio zona", "RMSE línea base"]),
            "p_vs_zona": _r(comp.loc["promedio zona", "p (t corregida)"]),
            "gana_a_zona": _r(comp.loc["promedio zona", "% particiones que gana el modelo"], 1),
            "rmse_municipio": _r(comp.loc["promedio municipio", "RMSE línea base"]),
            "rmse_global": _r(comp.loc["promedio global", "RMSE línea base"]),
        },
        "modelo": {
            "coeficientes": {k: _r(v, 4) for k, v in b.items()},
            "promedio_base": _r(promedio_entrenamiento),
            "variables": [{"variable": v, "nombre": etiqueta(v), "coef": _r(b[v], 4), "p": _r(coef.loc[v, "p"], 4)}
                          for v in config.VARIABLES_MODELO],
            "cortes_categoria": [_r(c, 2) for c in cortes],
        },
        "intervalo": {
            "nivel": int(config.NIVEL_INTERVALO * 100),
            "metodo": "Conformal por zona: predicción + percentiles 10 y 90 de los errores fuera de muestra de su zona",
            "desplazamientos": {"Sur": [_r(v) for v in desplazamientos[0]], "Norte": [_r(v) for v in desplazamientos[1]]},
            "cobertura": _r(metodo_elegido["cobertura_%"], 1), "cobertura_norte": _r(metodo_elegido["cobertura_norte_%"], 1),
            "cobertura_sur": _r(metodo_elegido["cobertura_sur_%"], 1), "ancho": _r(metodo_elegido.ancho_promedio_t_ha),
            "metodos": _tabla(metodos_intervalo, 3),
            "qrf": {"cobertura": _r(metodo_qrf["cobertura_%"], 1), "cobertura_norte": _r(metodo_qrf["cobertura_norte_%"], 1),
                    "cobertura_sur": _r(metodo_qrf["cobertura_sur_%"], 1), "ancho": _r(metodo_qrf.ancho_promedio_t_ha),
                    "rmse": _r(modelos.set_index("modelo").loc["QRF (mediana)", "RMSE"]),
                    "pct_coinciden": _r(acuerdo["% de predicciones en que regresión y QRF coinciden"], 1),
                    "error_coinciden": _r(acuerdo["error medio cuando coinciden (t/ha)"]),
                    "error_no_coinciden": _r(acuerdo["error medio cuando no coinciden (t/ha)"]),
                    "corr_ancho_error": _r(acuerdo["correlación error vs. ancho del QRF"]),
                    "umbral": config.UMBRAL_COINCIDENCIA},
        },
        "modelos": _tabla(modelos, 3),
        "importancia": [{"variable": r.variable, "nombre": etiqueta(r.variable), "aumento": _r(r.aumento_RMSE), "p": _r(r.p, 4)}
                        for r in importancia.itertuples()],
        "bonus": {
            "respuestas": _limpiar(resp),
            "indices": _tabla(bonus["indices"].head(12)[["variable", "nombre", "R2_sola", "R2_con_base", "aporte_RMSE", "p", "seleccion_lasso_%"]]),
            "clima": _tabla(bonus["clima"][["variable", "nombre", "R2_sola", "R2_con_base", "aporte_RMSE", "p", "seleccion_lasso_%"]]),
            "combinaciones": _tabla(bonus["combinaciones"]),
        },
    }
    return datos


def escribir(datos, carpeta):
    carpeta.mkdir(exist_ok=True)
    ruta = carpeta / "datos_parcelas.js"
    texto = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    ruta.write_text(f"/* Generado por run_fase3.py el {datos['meta']['generado']}. No editar a mano. */\n"
                    f"window.CEBA_DATA = {texto};\n", encoding="utf-8")
    return ruta
