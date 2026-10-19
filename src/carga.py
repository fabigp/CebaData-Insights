"""Carga y verificación de los archivos del reto."""
import geopandas as gpd
import pandas as pd

import config


def _verificar_archivo(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta.name} en {ruta.parent}. "
            "Copia los archivos del reto a la carpeta datos_entrada (ver README)."
        )


def cargar_rendimientos():
    """ID, área, rendimiento (vacío en las parcelas a predecir) y conjunto."""
    _verificar_archivo(config.ARCHIVO_RENDIMIENTOS)
    df = pd.read_csv(config.ARCHIVO_RENDIMIENTOS, encoding="utf-8-sig",
                     usecols=["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"])
    return df


def cargar_parcelas():
    """Municipio y estado de cada parcela, leídos de la tabla del shapefile."""
    _verificar_archivo(config.ARCHIVO_PARCELAS)
    g = gpd.read_file(config.ARCHIVO_PARCELAS)
    g.columns = [c.strip() for c in g.columns]
    df = pd.DataFrame(g.drop(columns="geometry"))
    df = df.rename(columns={"ID_POLIGON": "ID_POLIGONO"})
    return df[["ID_POLIGONO", "Municipio", "Estado"]]


def cargar_sentinel2():
    """Observaciones de Sentinel-2 (solo NDVI, fecha y nubosidad)."""
    _verificar_archivo(config.ARCHIVO_SENTINEL2)
    df = pd.read_csv(config.ARCHIVO_SENTINEL2, encoding="utf-8-sig",
                     usecols=["ID_POLIGONO", "fecha_captura", "sensor", "ndvi_promedio", "porcentaje_nubosidad"])
    df = df[df.sensor == "Sentinel-2"].copy()
    df["fecha"] = pd.to_datetime(df.fecha_captura, format="%d/%m/%Y")
    return df


def verificar_consistencia(rend, parcelas, s2):
    """Comprobaciones básicas; devuelve un resumen para el reporte."""
    ids_rend, ids_shp, ids_s2 = set(rend.ID_POLIGONO), set(parcelas.ID_POLIGONO), set(s2.ID_POLIGONO)
    if not (ids_rend == ids_shp == ids_s2):
        raise ValueError("Los identificadores de parcela no coinciden entre archivos.")
    resumen = {
        "parcelas totales": len(rend),
        "parcelas de entrenamiento": int((rend.CONJUNTO == "ENTRENAMIENTO").sum()),
        "parcelas a predecir": int((rend.CONJUNTO == "PREDICCION").sum()),
        "observaciones Sentinel-2": len(s2),
        "municipios": parcelas.Municipio.nunique(),
    }
    falt = parcelas[~parcelas.Municipio.isin(config.MUNICIPIOS_NORTE)].Municipio.unique()
    resumen["municipios del sur"] = ", ".join(sorted(falt))
    return resumen


def cargar_geometrias():
    """Polígonos de las parcelas en 2D (el shapefile trae coordenada Z) y en WGS84, para el mapa."""
    import shapely
    g = gpd.read_file(config.ARCHIVO_PARCELAS)
    g.columns = [c.strip() for c in g.columns]
    g = g.rename(columns={"ID_POLIGON": "ID_POLIGONO"}).to_crs(4326)
    g["geometry"] = shapely.force_2d(g.geometry)
    return g[["ID_POLIGONO", "geometry"]]


def cargar_candidatas():
    """Variables candidatas de la investigación (solo para las preguntas bonus)."""
    _verificar_archivo(config.ARCHIVO_CANDIDATAS)
    return pd.read_csv(config.ARCHIVO_CANDIDATAS, encoding="utf-8-sig")
