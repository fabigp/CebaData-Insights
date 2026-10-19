# Reto AgroCebada 2026 — Fase 3: modelo aplicado en la página web

Genera `web/datos_parcelas.js`, el archivo de datos que lee la página. Integra tres criterios por parcela y las respuestas bonus:

1. **Predicción oficial**: regresión lineal con zona, verdor histórico y área. Es el mismo número de la Fase 1, el que se entrega.
2. **Rango probable del 80%**: intervalo conformal por zona, validado con las mismas 50 particiones.
3. **Segunda opinión (QRF)**: bosque de regresión por cuantiles con las mismas variables.
4. **Preguntas bonus**: índice más importante, variable climática más influyente y mejor combinación.

## Estructura

```
fase3/
├── run_fase3.py          ← programa principal
├── config.py             ← todas las decisiones (rutas, intervalo, QRF, reglas de las preguntas bonus)
├── requirements.txt
├── src/
│   ├── carga.py, variables.py, evaluacion.py, modelo.py   ← los mismos de la Fase 1, más geometrías y NDVI mensual
│   ├── intervalo.py      ← compara 3 formas de intervalo y entrena el QRF
│   ├── comparacion.py    ← compara familias de modelos e importancia de variables
│   ├── bonus.py          ← preguntas bonus (175 variables candidatas)
│   ├── web.py            ← arma datos_parcelas.js
│   └── reporte.py        ← REPORTE_FASE3.md
├── datos_entrada/        ← archivos del reto + variables_candidatas.csv (ya incluido)
├── web/
│   ├── index.html        ← página actualizada
│   └── datos_parcelas.js ← datos generados (ya incluido, listo para usar)
└── resultados/           ← se crea al ejecutar
```

## Uso rápido (sin ejecutar nada)

`web/datos_parcelas.js` ya viene generado. Para actualizar tu sitio:

1. Reemplaza el `index.html` y el `datos_parcelas.js` de tu sitio por los de `web/`.
2. Conserva tu carpeta `img/` (cebadito.png, cebada2.png) junto al `index.html`.
3. Abre `index.html` en el navegador.

Si tu `index.html` tiene cambios que no estaban en la versión que compartiste, copia solo las partes nuevas. Los cambios están en la sección `<style>`, en las secciones `#predicciones`, `#analisis` y `#bonus`, y en el `<script>` final.

## Regenerar los datos (Windows, PowerShell)

1. Copia a `datos_entrada\` los mismos archivos de la Fase 1: el CSV de rendimientos, el CSV BÁSICO y el shapefile con sus 5 archivos. `variables_candidatas.csv` ya está incluido.
2. Instala lo necesario. Puedes reusar el entorno de la Fase 1 y agregar `quantile-forest`:
   ```
   cd D:\Pablo\CebaData_2026\fase3
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Ejecuta `python run_fase3.py`. Tarda unos 5 minutos.
4. Revisa `resultados\REPORTE_FASE3.md` (Ctrl+Shift+V en VS Code) y copia `web\datos_parcelas.js` a tu sitio.

## Cómo comprobar que todo salió bien

- La sección 1 del reporte debe decir **"sí"** en RMSE 0.467, MAE 0.338 y R² 0.686.
- En la sección 2, el conformal por zona debe cubrir alrededor del 79% en ambas zonas.
- `resultados\entrega_predicciones.csv` debe ser idéntico al de la Fase 1.

## Salidas en `resultados\`

| Archivo | Contenido |
| --- | --- |
| `REPORTE_FASE3.md` | Reporte completo: comprobación, intervalo, QRF, modelos, importancia y preguntas bonus |
| `predicciones_tres_criterios.csv` | Las 59 parcelas: predicción, rango 80%, categoría, QRF (P10, mediana, P90), coincidencia y promedio de su zona |
| `entrega_predicciones.csv` | Archivo para la convocatoria (igual a la Fase 1) |
| `intervalo_metodos.csv` | Cobertura y ancho de las 3 formas de intervalo |
| `qrf_segunda_opinion.csv` | Qué tanto coinciden la regresión y el QRF, y si eso anticipa el error |
| `comparacion_modelos.csv` | Error de cada familia de modelos con las 3 variables, con valores p |
| `importancia_modelo_final.csv` | Cuánto empeora el error al quitar cada variable |
| `bonus_indices.csv`, `bonus_clima.csv`, `bonus_relieve.csv` | Cada variable candidata: R² sola, aporte sobre zona + área, p y % de selección Lasso |
| `bonus_combinaciones.csv` | Error de cada combinación de variables |
| `bonus_frecuencia_lasso.csv` | % de particiones en que Lasso elige cada una de las 175 variables |

## Sobre `variables_candidatas.csv`

Contiene las variables candidatas (índices 2025, fenología, clima, relieve, suelo, radar, borde rojo, etc.) que se calcularon durante la investigación con los rásteres del reto y Google Earth Engine. Solo se usa para las preguntas bonus: el modelo final y las predicciones no dependen de este archivo.
