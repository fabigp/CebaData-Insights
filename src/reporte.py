"""Reporte en Markdown de la Fase 3 (para documentar la ejecución local)."""
import platform
from datetime import datetime

import numpy as np
import pandas as pd

import config


def _tabla_md(df, decimales=3):
    df = df.copy()
    for c in df.select_dtypes("number").columns:
        if c.startswith("p"):
            df[c] = [("< 0.001" if v < 0.001 else round(v, 3)) if pd.notna(v) else "—" for v in df[c]]
        else:
            df[c] = [round(v, decimales) if pd.notna(v) else "—" for v in df[c]]
    cols = list(df.columns)
    filas = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    filas += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(filas)


def escribir(carpeta, datos, metricas, metodos_intervalo, acuerdo, modelos, importancia, bonus, predicciones, segundos):
    esperado = config.REFERENCIA_ESPERADA
    obtenido = {"RMSE": metricas["RMSE (t/ha)"], "MAE": metricas["MAE (t/ha)"],
                "R2_fuera_de_muestra": metricas["R² fuera de muestra (todas las parcelas)"]}
    chequeo = pd.DataFrame({"métrica": list(esperado), "esperado": list(esperado.values()),
                            "obtenido": [obtenido[k] for k in esperado]})
    chequeo["coincide"] = np.where((chequeo.obtenido - chequeo.esperado).abs() <= 0.002, "sí", "REVISAR")
    r = bonus["respuestas"]
    i, c, cb = r["indice"], r["clima"], r["combinacion"]
    col_b = ["nombre", "R2_sola", "aporte_RMSE", "p", "seleccion_lasso_%"]
    texto = f"""# Reporte Fase 3 — Datos para la página web

Generado: {datetime.now():%Y-%m-%d %H:%M} · Duración: {segundos:.0f} s · Python {platform.python_version()}

Salida principal: `web/datos_parcelas.js` (lo lee `web/index.html`).

## 1. Comprobación del modelo final (igual que la Fase 1)

{_tabla_md(chequeo)}

## 2. Intervalo del 80 %

Un intervalo del 80 % es correcto si contiene el rendimiento real en ~80 % de las parcelas, en ambas zonas.

{_tabla_md(metodos_intervalo)}

Se usa el **conformal por zona**: es el único con cobertura cercana al 80 % en el norte y en el sur a la vez.
El QRF directo cubre de más en el sur y de menos en el norte; recentrar su rango en la regresión no alcanza el 80 %.

Desplazamientos aplicados a la predicción (percentiles 10 y 90 de los errores fuera de muestra):
norte {datos['intervalo']['desplazamientos']['Norte']} · sur {datos['intervalo']['desplazamientos']['Sur']} t/ha.

## 3. QRF como segunda opinión

{_tabla_md(acuerdo.rename("valor").reset_index().rename(columns={"index": "indicador"}))}

Lectura: cuando las dos predicciones no coinciden, el error no es claramente mayor y el ancho del QRF casi no se relaciona
con el error. Por eso el QRF se muestra como confirmación independiente y no como medida de riesgo de cada parcela.

## 4. Comparación de modelos (3 variables, mismas particiones)

{_tabla_md(modelos)}

## 5. Importancia de cada variable del modelo final (aumento del RMSE al quitarla)

{_tabla_md(importancia)}

## 6. Preguntas bonus

Base de comparación: {", ".join(r["base"]["variables"])} (RMSE {r["base"]["RMSE"]:.3f}). Candidatas evaluadas: {r["n_candidatas"]}.
Columnas constantes descartadas: {", ".join(r["constantes_descartadas"]) or "ninguna"}.
Orden de las tablas: % de particiones en que Lasso elige la variable compitiendo contra todas; desempate por aporte.
`R2_sola` negativo significa que, ella sola, predice peor que el promedio general.

### 6.1 ¿Qué índice es el más importante?

**{i["nombre"]}**: baja el error {abs(i["aporte_RMSE"]):.3f} t/ha sobre la base (p {i["p"]:.3g}); Lasso lo elige en {i["seleccion_lasso_%"]:.0f} % de las particiones.

{_tabla_md(bonus["indices"].head(12)[col_b])}

### 6.2 ¿Qué variable climática influye más?

**{c["nombre"]}**: aporte {c["aporte_RMSE"]:+.3f} t/ha sobre la base (p {c["p"]:.3g}); Lasso la elige en {c["seleccion_lasso_%"]:.0f} % de las particiones.
Variables de clima con mejora comprobada (p < {config.ALFA}): {c["n_clima_comprobadas"]} de {c["n_clima"]}.
Ojo: **{c["engañosa"]["nombre"]}** explica por sí sola R² {c["engañosa"]["R2_sola"]:.2f}, pero sumada a la zona aporta {c["engañosa"]["aporte_RMSE"]:+.3f} t/ha:
su relación con el rendimiento es la diferencia norte–sur, no un efecto propio del clima.

{_tabla_md(bonus["clima"].head(12)[col_b])}

### 6.3 ¿Qué combinación de variables funciona mejor?

Regla: solo se acepta una combinación más grande si mejora al modelo final de forma comprobada (p < {config.ALFA}).
Respuesta: **{cb["nombre"]}** (RMSE {cb["RMSE"]:.3f}).

{_tabla_md(bonus["combinaciones"])}

## 7. Predicciones con los tres criterios

{_tabla_md(predicciones.groupby(["zona", "Municipio"]).agg(parcelas=("ID_POLIGONO", "count"),
           prediccion=("PREDICCION_T_HA", "mean"), inferior_80=("INTERVALO_80_INF", "mean"),
           superior_80=("INTERVALO_80_SUP", "mean"), qrf=("QRF_MEDIANA", "mean"),
           coinciden=("REGRESION_Y_QRF_COINCIDEN", "sum")).reset_index(), 2)}

## Notas para la documentación

- Escribe aquí observaciones de la ejecución local.
"""
    ruta = carpeta / "REPORTE_FASE3.md"
    ruta.write_text(texto, encoding="utf-8")
    return ruta
