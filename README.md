# Segmentación no supervisada de una nube de puntos 3D de una fábrica

Actividad 3 · Machine Learning · UTEC · **Grupo 67**

- José Luis Alva Espinoza
- Juan Carlos Ticlia Maqui
- Elmer José Manuel Villegas Suárez
- Joseph Anderson Cose Rojas

Segmentamos la nube `FARO_Scan_Full_Sub1_down3_with_normals.ply` (1 647 021 puntos con xyz, RGB y normal) en piso, techo, columnas, vigas, tuberías, equipos, personas y vehículos usando solo clustering y reglas geométricas, sin etiquetas.

## Entregables

| Archivo | Contenido |
|---|---|
| `doc/informe.pdf` | Informe con el procedimiento, resultados y capturas (fuente: `doc/informe.tex`) |
| `entregable/final_segmentos.ply` | **Nube final** coloreada por clase (etapas 1-5), con normales |
| `entregable/final_leyenda.csv` | Color RGB de cada clase |
| `entregable/final_clases.csv` | Puntos, porcentaje y segmentos por clase |
| `entregable/final_personas_vehiculos.csv` | Detecciones de persona y vehículo (centro, tamaño, regla que las aceptó) |
| `entregable/final_segmentos.csv` | Instancias de la etapa 4 con bbox y descriptores |
| `entregable/clase_segmentos.ply` | Línea base con lo visto en clase (k-means, K=11) |
| `entregable/etapa4_clases.npz` | Clase por punto al final de la etapa 4 (entrada de la etapa 5) |
| `entregable/*.csv` restantes | Métricas de clase, ablaciones y comparación con la línea base |

Los PLY usan las propiedades estándar `red green blue`, así que se abren con color en Open3D, MeshLab y CloudCompare:

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("entregable/final_segmentos.ply")
o3d.visualization.draw_geometries([pcd])
```

## Leyenda de la nube final

| Clase | RGB |
|---|---|
| piso | 70, 155, 70 |
| techo | 90, 140, 230 |
| columna | 230, 60, 55 |
| viga_horizontal | 245, 170, 35 |
| viga_o_tuberia | 225, 125, 25 |
| equipo_bajo | 155, 75, 190 |
| compacto_plausible | 30, 200, 190 |
| mezcla_no_identificada | 210, 110, 160 |
| persona | 255, 0, 255 |
| vehiculo | 20, 20, 140 |
| ruido | 45, 45, 45 |

## Método

1. **Línea base (clase):** k-means, DBSCAN, jerárquico, GMM y PCA sobre xyz, color Lab, normales y altura (`notebooks/clase/`).
2. **Etapa 1 · Estructura:** RANSAC con filtro de normales separa piso y techo.
3. **Etapa 2 · Orientación local:** PCA local (k=96) divide el resto en lineal vertical, lineal horizontal en x o y, y planar.
4. **Etapa 3 · Instancias:** HDBSCAN dentro de cada orientación y fusión de fragmentos colineales.
5. **Etapa 4 · Clases heurísticas:** reglas de forma y tamaño (columna, viga, equipo, etc.).
6. **Etapa 5 · Personas y vehículos:** DBSCAN sobre objetos libres respecto del piso local, reglas de tamaño humano y vehicular, y separación de personas pegadas por cabezas.

El detalle, las ablaciones y las limitaciones están en el informe.

## Estructura

```
data/                       nube original (PLY)
src/pcseg/                  io, features, métricas, propagación y etapas (stages.py)
notebooks/clase/            01-06: línea base con lo visto en clase
notebooks/experimentacion/  00-05: etapas 1-4 y ablaciones · 06: etapa 5 (personas y vehículos)
scripts/                    generadores de notebooks y verify.sh
tests/                      pytest
doc/                        informe.tex, informe.pdf y figuras
entregable/                 resultados finales
```

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests

# Rápido: solo la etapa 5 (usa entregable/etapa4_clases.npz)
jupyter nbconvert --to notebook --execute --inplace notebooks/experimentacion/06_etapa5_personas_vehiculos.ipynb

# Completo: ejecutar en orden los notebooks de clase (01-06) y luego los de experimentación (00-06).
# Las salidas intermedias van a outputs/ (no versionado).
```

HDBSCAN puede dar pequeñas diferencias entre máquinas (por ejemplo 239 contra 243 segmentos). Por eso la etapa 5 parte del resultado guardado en `etapa4_clases.npz`, y el PLY final es reproducible desde ahí.

Para compilar el informe: `cd doc && latexmk -pdf informe.tex` (o `tectonic informe.tex`).
