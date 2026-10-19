"""
Configuración de la Fase 3: datos para la página web (modelo final + intervalo + QRF + preguntas bonus).

Aquí se concentran todas las decisiones, para que cualquier cambio quede documentado en un solo lugar.
Las secciones "Rutas", "Zonas", "Historial", "Modelo" y "Validación" son idénticas a la Fase 1.
"""
from pathlib import Path

# --- Rutas -----------------------------------------------------------------------------------
CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_ENTRADA = CARPETA_PROYECTO / "datos_entrada"
CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"     # tablas y reporte para la documentación
CARPETA_WEB = CARPETA_PROYECTO / "web"                   # index.html + datos_parcelas.js (se copia al sitio)

ARCHIVO_RENDIMIENTOS = CARPETA_ENTRADA / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
ARCHIVO_SENTINEL2 = CARPETA_ENTRADA / "Conjunto_datos_BASICO_AgroCebada2026.csv"
ARCHIVO_PARCELAS = CARPETA_ENTRADA / "Parcelas_Reto_AGC_CONJUNTO.shp"
# Tabla de variables candidatas (índices, clima, relieve, suelo...) generada durante la investigación.
# Solo se usa para responder las preguntas bonus; el modelo final no depende de ella.
ARCHIVO_CANDIDATAS = CARPETA_ENTRADA / "variables_candidatas.csv"

# --- Zonas ------------------------------------------------------------------------------------
MUNICIPIOS_NORTE = ["Chignahuapan", "Almoloya", "Cuautepec de Hinojosa", "Tepeapulco"]

# --- Historial (verdor acumulado 2022-2024, Sentinel-2) ---------------------------------------
NUBOSIDAD_MAXIMA = 10
ANIOS_HISTORIAL = (2022, 2023, 2024)
TEMPORADA_INICIO = "04-01"
TEMPORADA_FIN = "11-30"
VENTANA_SUAVIZADO_DIAS = 15
UMBRAL_FENOLOGICO = 0.5
ANIO_CICLO = 2025                # año de la curva de NDVI mensual que muestra la página

# --- Modelo -----------------------------------------------------------------------------------
VARIABLES_MODELO = ["zona_norte", "hist_integral_prom", "log_area"]

# --- Validación cruzada ----------------------------------------------------------------------
N_PARTICIONES = 5
N_REPETICIONES = 10
SEMILLA = 42

# --- Intervalo y segunda opinión ---------------------------------------------------------------
NIVEL_INTERVALO = 0.80           # intervalo del 80 %: percentiles 10 y 90
QRF_ARBOLES = 500
QRF_HOJA_MINIMA = 5
UMBRAL_COINCIDENCIA = 0.25       # t/ha: si |regresión − QRF| es menor, se dice que ambos modelos coinciden

# --- Preguntas bonus -------------------------------------------------------------------------
BASE_BONUS = ["zona_norte", "log_area"]   # el aporte de cada índice o variable de clima se mide sobre esta base
ALFA = 0.05                                # nivel para decir que una mejora es "comprobada"

REFERENCIA_ESPERADA = {"RMSE": 0.467, "MAE": 0.338, "R2_fuera_de_muestra": 0.686}
