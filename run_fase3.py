"""
Reto AgroCebada 2026 — Fase 3: datos para la página web.

Integra en un solo archivo de datos (web/datos_parcelas.js):
  1. el modelo final (el mismo número que se entrega a la convocatoria);
  2. el intervalo del 80 % validado y la segunda opinión del QRF;
  3. las respuestas a las preguntas bonus (índice, clima y combinación de variables).

Ejecución:   python run_fase3.py          (tarda unos minutos)
Salidas:     web/datos_parcelas.js  y  resultados/ (tablas CSV y REPORTE_FASE3.md)
"""
import time

import pandas as pd

import config
from src import bonus, carga, comparacion, evaluacion, intervalo, modelo, reporte, variables, web


def main():
    t0 = time.time()
    R = config.CARPETA_RESULTADOS
    R.mkdir(exist_ok=True)

    print("1/7 Cargando archivos...")
    rend, parcelas, s2 = carga.cargar_rendimientos(), carga.cargar_parcelas(), carga.cargar_sentinel2()
    carga.verificar_consistencia(rend, parcelas, s2)
    geom, candidatas = carga.cargar_geometrias(), carga.cargar_candidatas()

    print("2/7 Variables del modelo y NDVI mensual 2025...")
    tabla = variables.construir_tabla(rend, parcelas, s2)
    ndvi = variables.ndvi_mensual(s2)
    parts = evaluacion.particiones(int((tabla.CONJUNTO == "ENTRENAMIENTO").sum()))

    print("3/7 Modelo final: validación y entrenamiento...")
    metricas, comparaciones_f1, por_particion_f1, detalle = evaluacion.validar(tabla)
    mod, coef, _ = modelo.entrenar(tabla)
    pred = modelo.predecir(mod, tabla)
    print(f"    RMSE {metricas['RMSE (t/ha)']:.3f} | MAE {metricas['MAE (t/ha)']:.3f} | "
          f"R² {metricas['R² fuera de muestra (todas las parcelas)']:.3f}")

    print("4/7 Intervalo del 80 % y QRF (unos 2 minutos)...")
    metodos, acuerdo, por_particion_qrf, errores_reg = intervalo.validar(tabla, parts)
    desplazamientos, qrf = intervalo.finales(tabla, errores_reg)
    print(metodos.round(3).to_string(index=False))

    print("5/7 Comparación de modelos e importancia de variables...")
    modelos = comparacion.comparar(tabla, parts, por_particion_f1, por_particion_qrf)
    importancia = comparacion.importancia_final(tabla, parts, por_particion_f1.RMSE_modelo.values)

    print("6/7 Preguntas bonus...")
    b = bonus.analizar(tabla, candidatas, parts)
    for k, v in (("Índice", b["respuestas"]["indice"]["nombre"]), ("Clima", b["respuestas"]["clima"]["nombre"]),
                 ("Combinación", b["respuestas"]["combinacion"]["nombre"])):
        print(f"    {k}: {v}")

    print("7/7 Escribiendo datos de la página y reporte...")
    datos = web.construir(tabla, geom, ndvi, mod, coef, metricas, comparaciones_f1, detalle, desplazamientos, qrf,
                          metodos, acuerdo, modelos, importancia, b)
    ruta_js = web.escribir(datos, config.CARPETA_WEB)

    p = pd.DataFrame([f["properties"] for f in datos["features"] if f["properties"]["conjunto"] == "PREDICCION"])
    predicciones = pd.DataFrame({
        "ID_POLIGONO": p.id, "Municipio": p.municipio, "zona": p.zona, "AREA_HA": p.area,
        "PREDICCION_T_HA": p.prediccion, "INTERVALO_80_INF": p.int_inf, "INTERVALO_80_SUP": p.int_sup,
        "CATEGORIA": p.categoria, "QRF_P10": p.qrf_p10, "QRF_MEDIANA": p.qrf_mediana, "QRF_P90": p.qrf_p90,
        "REGRESION_Y_QRF_COINCIDEN": p.coinciden, "PROMEDIO_REAL_ZONA": p.ref_zona})
    predicciones.to_csv(R / "predicciones_tres_criterios.csv", index=False)
    (pred[["ID_POLIGONO", "RENDIMIENTO_PREDICHO_T_HA"]].round(2)
     .rename(columns={"RENDIMIENTO_PREDICHO_T_HA": "RENDIMIENTO_T_HA"}).to_csv(R / "entrega_predicciones.csv", index=False))
    metodos.to_csv(R / "intervalo_metodos.csv", index=False)
    acuerdo.rename("valor").to_csv(R / "qrf_segunda_opinion.csv")
    modelos.to_csv(R / "comparacion_modelos.csv", index=False)
    importancia.to_csv(R / "importancia_modelo_final.csv", index=False)
    for k in ("indices", "clima", "relieve", "combinaciones"):
        b[k].to_csv(R / f"bonus_{k}.csv", index=False)
    b["frecuencia_lasso"].rename("seleccion_lasso_%").to_csv(R / "bonus_frecuencia_lasso.csv")

    ruta_md = reporte.escribir(R, datos, metricas, metodos, acuerdo, modelos, importancia, b, predicciones, time.time() - t0)
    print(f"\nListo en {time.time() - t0:.0f} s.\n  Página: {ruta_js}\n  Reporte: {ruta_md}")


if __name__ == "__main__":
    main()
