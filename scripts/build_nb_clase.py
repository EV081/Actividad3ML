"""Genera seis notebooks autocontenidos; no modifica pcseg ni instala paquetes.

Uso: .venv/bin/python scripts/build_nb_clase.py
Después de ejecutarlos: .venv/bin/python scripts/build_nb_clase.py --conclusions
"""
from pathlib import Path
import json
import re
import sys
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebooks/clase'
DEST.mkdir(parents=True, exist_ok=True)
NAMES = ['01_carga_y_eda', '02_kmeans', '03_dbscan', '04_jerarquico_gmm',
         '05_pca_visualizacion', '06_resultado_clase']

def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())

def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())

SETUP = '''
import os, sys, json, time, importlib.util
from pathlib import Path
sys.path.insert(0, "../../src")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from threadpoolctl import threadpool_limits
from pcseg.io import read_ply, write_ply, labels_to_rgb
from pcseg.features import voxel_downsample, voxel_ids, build_features
from pcseg.propagate import propagate_labels
from pcseg.metrics import cluster_summary, silhouette_sample, davies_bouldin, stability_ari
# Mostramos los PNG generados sin abrir ventanas gráficas.
plt.switch_backend("module://matplotlib_inline.backend_inline")
# Evitamos que muchos threads hagan más lento un problema pequeño.
thread_limit = threadpool_limits(limits=2)
OUT = Path('../../outputs'); FIG = OUT / 'figs'; ENT = Path('../../entregable'); ENT.mkdir(exist_ok=True)
CACHE = OUT / 'cache'; LOGS = OUT / 'logs'
for folder in (OUT, FIG, CACHE, LOGS): folder.mkdir(exist_ok=True)
(LOGS / 'clase_labels').mkdir(exist_ok=True)
START = time.perf_counter()
SEED = 0
print('Python:', sys.executable, sys.version.split()[0])
assert Path(sys.prefix).resolve() == Path('../../.venv').resolve(), 'Ejecutar con .venv'
'''
VIZ = '''
# Reordenamos tab20: primero los tonos fuertes; misma paleta en PNG y PLY.
SEGMENT_PALETTE = (np.array(plt.get_cmap('tab20').colors)[
    np.r_[np.arange(0,20,2), np.arange(1,20,2)]] * 255).astype(np.uint8)
def segment_rgb(labels):
    return labels_to_rgb(labels, palette=SEGMENT_PALETTE)

def scatter_views(points, values, path, title, max_points=40000, seed=0):
    # Variante local: puntos visibles y alpha=0.6, sin modificar pcseg.viz.
    if len(points) > max_points:
        idx = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points, values = points[idx], values[idx]
    colors = (segment_rgb(values) if values.ndim == 1 else values) / 255.
    fig = plt.figure(figsize=(18,6))
    for j, (axis, label) in enumerate(((1,'y'),(2,'z')), start=1):
        ax = fig.add_subplot(1,3,j)
        ax.scatter(points[:,0],points[:,axis],c=colors,s=2.2,alpha=.6,linewidths=0)
        ax.set(xlabel='x',ylabel=label,title='Superior' if j==1 else 'Lateral',aspect='equal')
    ax = fig.add_subplot(1,3,3,projection='3d')
    ax.scatter(*points.T,c=colors,s=2,alpha=.6,linewidths=0)
    ax.set(xlabel='x',ylabel='y',zlabel='z',title='3D')
    ax.set_box_aspect(np.maximum(np.ptp(points,axis=0),1e-9))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
'''

LOAD = '''
cloud = read_ply('../../data/FARO_Scan_Full_Sub1_down3_with_normals.ply')
sample_idx = np.load(OUT / 'sample_idx.npy')
assert len(np.unique(sample_idx)) == len(sample_idx)
xyz, rgb, normals = (cloud[key][sample_idx] for key in ('xyz', 'rgb', 'normals'))
print('Muestra común:', len(xyz), 'de', len(cloud['xyz']))
'''
FEATURES = '''
FEATURES = {
    'xyz': {},
    'xyz+rgb': {'use_rgb': True},
    'xyz+normales': {'use_normals': True},
    'xyz+rgb+normales+height': {'use_rgb': True, 'use_normals': True, 'use_height': True},
}
def features(name, scaled=True):
    return build_features(xyz, rgb, normals, standardize=scaled, **FEATURES[name])
# Evaluación común: igual distancia y mismas filas para comparar métodos/features.
EVAL = features('xyz+rgb+normales+height')
'''
HELPERS = '''
rows = []
def record(method, feature, scaled, params, X, labels, seconds, ari, ari_std=0., **extra):
    summary = cluster_summary(labels)
    own_sil = silhouette_sample(X, labels, n=1200, seed=0)
    own_db = davies_bouldin(X, labels)
    common_sil = silhouette_sample(EVAL, labels, n=1200, seed=0)
    common_db = davies_bouldin(EVAL, labels)
    counts = np.array(list(summary['sizes'].values()), dtype=int)
    row_id = f'{NB}_{len(rows):03d}'
    row = dict(id=row_id, metodo=method, features=feature, estandarizado=bool(scaled),
        parametros=json.dumps(params, sort_keys=True), n_fit=extra.get("n_ajuste", len(X)), n_evaluacion=len(X),
        n_clusters=summary['n_clusters'], ruido=summary['noise_frac'],
        silhouette=own_sil, DB=own_db, ARI=ari, ARI_std=ari_std,
        silhouette_comun=common_sil, DB_comun=common_db, segundos=seconds,
        min_cluster_frac=float(counts.min()/len(labels)) if len(counts) else 0.,
        max_cluster_frac=float(counts.max()/len(labels)) if len(counts) else 0., **extra)
    rows.append(row)
    np.save(LOGS / 'clase_labels' / f'{row_id}.npy', labels.astype(np.int32))
    return row

def table():
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'metricas_{NB}.csv', index=False)
    return df

def view(labels, suffix, title):
    path = FIG / f'{NB}_{suffix}.png'
    scatter_views(xyz, labels, path, title, max_points=30000, seed=0)
    display(Image(filename=str(path)))

def finish(payload):
    payload['segundos_notebook'] = time.perf_counter() - START
    (CACHE / f'clase_resumen_{NB}.json').write_text(json.dumps(payload, indent=2))
    print(payload)
'''
PROTOCOL = '''
### Antes de comparar: fijemos la regla
Trabajamos siempre con los índices voxel de 01. `rgb` pasa a CIE Lab dentro de
`build_features`: aquí el nombre del conjunto indica el dato de origen.
`height = z - percentil_1(z)` repite la información de z y le da más peso a la altura.

Calculamos silhouette en 1200 puntos (semilla 0) y Davies-Bouldin sin ruido.
Silhouette mayor y DB menor son deseables, pero dependen de la distancia elegida.
Por eso guardamos además `silhouette_comun` y `DB_comun`, siempre sobre las mismas
features completas estandarizadas. No hay ground truth: ninguna métrica demuestra
que los grupos sean objetos físicos. ARI compara corridas, no exactitud.
Los segundos del CSV incluyen ajuste, predicción y estabilidad, y excluyen evaluación y figuras.
'''

def base(title, load=True, helpers=False, nb=None):
    cells = [md(title), code(SETUP), md('### Cómo vamos a dibujar\nUsamos una función local de matplotlib: puntos de tamaño 2–2.2 y alpha=0.6. La paleta prioriza colores contrastantes y es la misma en figuras y PLY. Esto cambia la presentación, no el clustering.'), code(VIZ)]
    if load:
        cells += [md('### Recuperamos la misma muestra\nNo volvemos a sortear puntos: cargamos los índices guardados en 01.'), code(LOAD)]
    if helpers:
        cells += [md(PROTOCOL), code(FEATURES), code(f"NB = '{nb}'\n" + HELPERS)]
    return cells

def save(name, cells):
    # Mostrar explícitamente los PNG permite ver figuras incluso con backend Agg.
    for cell in cells:
        if cell.cell_type == 'code':
            figure = re.search(r"fig.savefig\(FIG\s*/\s*'([^']+)'", cell.source)
            if figure:
                show = f"display(Image(filename=str(FIG / '{figure.group(1)}')))"
                cell.source = cell.source.replace('plt.show()', show)
    cells.append(md('## Qué aprendimos / qué falló\n\nPendiente de redactar después de inspeccionar la ejecución real.'))
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (.venv)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'clase': {'referencias': [
            'session-10-simple-a-kmeans-and-what-goes-wrong.ipynb',
            'session-10-simple-b-how-many-groups-and-clouds.ipynb',
            'session-10-simple-c-crowds-trees-and-choosing.ipynb',
            'session-11a-pca-lda-and-factorizations.ipynb',
            'session-11b-tsne-and-umap.ipynb'], 'semilla': 0}})
    nbf.write(nb, DEST / f'{name}.ipynb')


def build():
    c = base('''# 01 · Carga y EDA
    **Objetivo.** Conocer la nube de la fábrica antes de elegir grupos. Vamos a revisar
    coordenadas, color y normales, y guardar una muestra voxel reproducible.
    Antes de mirar: ¿dónde habrá más puntos, en el piso o en las partes altas?''', load=False)
    c += [md('### Primero, revisamos qué llegó\nLas coordenadas ubican cada punto; el color describe su apariencia y las normales su orientación. No tenemos etiquetas reales.'), code('''
cloud = read_ply('../../data/FARO_Scan_Full_Sub1_down3_with_normals.ply')
assert len(cloud['xyz']) == 1647021
assert all(np.isfinite(v).all() for v in cloud.values())
columns = ['x','y','z','r','g','b','nx','ny','nz']
values = np.column_stack([cloud['xyz'], cloud['rgb'], cloud['normals']])
stats = pd.DataFrame(values, columns=columns).describe().T
stats.to_csv(CACHE / 'clase_eda_estadisticas.csv')
display(stats)
print('Normales con norma cero:', int((np.linalg.norm(cloud['normals'], axis=1) == 0).sum()))
'''), md('### Miremos las distribuciones\nUn histograma cuenta puntos por intervalo. Para el color superponemos los tres canales originales, todavía sin convertir a Lab.'), code('''
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].hist(cloud['xyz'][:,2], bins=80); ax[0].set(xlabel='z', ylabel='Puntos', title='Altura')
ax[1].hist(cloud['normals'][:,2], bins=60); ax[1].set(xlabel='nz', title='Orientación vertical')
for j, color in enumerate(['r','g','b']):
    ax[2].hist(cloud['rgb'][:,j], bins=64, alpha=.4, color=color, label=color)
ax[2].set(xlabel='Canal (0–255)', title='Color original'); ax[2].legend()
fig.tight_layout(); fig.savefig(FIG / '01_histogramas.png', dpi=150); plt.show(); plt.close(fig)
'''), md('### Elegimos el tamaño del voxel con los datos\nCada voxel conserva su primer punto, sin promediar colores ni normales. Empezamos en 0.12 unidades y ajustamos hasta tener entre 60 mil y 150 mil representantes; apuntamos a 90 mil. El algoritmo no es aleatorio. Guardamos índices, tamaño de voxel e IDs para poder auditar y repetir.'), code('''
voxel = .12
trials = []
for attempt in range(12):
    sample_idx = voxel_downsample(cloud['xyz'], voxel)
    trials.append({'voxel': voxel, 'n': len(sample_idx)})
    if 60000 <= len(sample_idx) <= 150000:
        break
    voxel *= np.clip((len(sample_idx) / 90000)**.5, .65, 1.7)
assert 60000 <= len(sample_idx) <= 150000
ids = voxel_ids(cloud['xyz'], voxel)
assert np.array_equal(ids[sample_idx], np.arange(len(sample_idx)))
if (OUT/'sample_idx.npy').exists():
    assert np.array_equal(sample_idx,np.load(OUT/'sample_idx.npy')), 'La muestra común cambió'
np.save(OUT / 'sample_idx.npy', sample_idx)
np.save(CACHE / 'clase_voxel_ids.npy', ids.astype(np.int32))
(CACHE / 'clase_muestra.json').write_text(json.dumps({'voxel': voxel, 'n': len(sample_idx),
    'n_original': len(cloud['xyz']), 'seed': 0, 'trials': trials}, indent=2))
display(pd.DataFrame(trials))
'''), md('### La nube con sus colores originales\nDibujamos hasta 40 mil puntos de la nube original para que la figura sea legible. Esto solo reduce el dibujo; la muestra de trabajo sigue siendo la guardada arriba.'), code('''
path = FIG / '01_original.png'
scatter_views(cloud['xyz'], cloud['rgb'], path, 'Fábrica · color original', max_points=40000, seed=0)
display(Image(filename=str(path)))
payload = dict(n_original=len(cloud['xyz']), n_sample=len(sample_idx), voxel=voxel,
    z_min=float(cloud['xyz'][:,2].min()), z_max=float(cloud['xyz'][:,2].max()),
    segundos_notebook=time.perf_counter()-START)
(CACHE / 'clase_resumen_01.json').write_text(json.dumps(payload, indent=2))
print(payload)
''')]
    save(NAMES[0], c)

    c = base('''# 02 · k-means: distancia, K y puntos de partida
    **Objetivo.** Comparar cuatro conjuntos de features con y sin estandarizar.
    k-means acerca puntos a centroides y minimiza WCSS. ¿Cambiar las unidades
    cambia los grupos aunque los puntos sean los mismos?''', helpers=True, nb='02')
    c += [md('### Probamos K, sin decidir por una sola curva\nPara cada representación ajustamos K=2…12 con tres starts (`n_init=3`). Repetimos el ajuste con semillas fijas 0, 1 y 2; `stability_ari` compara las tres parejas. Son excepciones deliberadas a la semilla principal 0 para medir sensibilidad. Guardamos cada configuración y las etiquetas de la corrida 0. WCSS solo se compara dentro de una misma representación.'), code('''
from sklearn.cluster import KMeans
for name in FEATURES:
    for scaled in (False, True):
        X = features(name, scaled)
        for k in range(2, 13):
            tick = time.perf_counter()
            runs = {s: KMeans(n_clusters=k, n_init=3, max_iter=200, random_state=s).fit(X)
                    for s in (0, 1, 2)}
            ari, sd = stability_ari(lambda data, seed: runs[seed].labels_, X, [0,1,2])
            elapsed = time.perf_counter() - tick
            record('kmeans', name, scaled, {'n_clusters': k, 'n_init': 3, 'max_iter': 200},
                   X, runs[0].labels_, elapsed, ari, sd, WCSS=runs[0].inertia_)
        print(name, 'estandarizado =', scaled, 'listo', flush=True)
df = table()
display(df.sort_values('silhouette_comun', ascending=False).head(8))
'''), md('### Elbow, silhouette y estabilidad\nElbow busca dónde la mejora de WCSS empieza a ser menor. Silhouette pregunta si cada punto está más cerca de su grupo que del vecino. ARI muestra si otros starts recuperan una partición parecida; una curva alta de silhouette puede venir con inestabilidad.'), code('''
fig, axes = plt.subplots(4, 3, figsize=(13, 12))
for i, name in enumerate(FEATURES):
    for scaled in (False, True):
        part = df[(df.features == name) & (df.estandarizado == scaled)]
        ks = [json.loads(p)['n_clusters'] for p in part.parametros]
        for j, metric in enumerate(['WCSS','silhouette','ARI']):
            axes[i,j].plot(ks, part[metric], 'o-', label='Con escala' if scaled else 'Sin escala')
            axes[i,j].set(xlabel='K', ylabel=metric, title=name)
    axes[i,0].legend()
fig.tight_layout(); fig.savefig(FIG / '02_curvas.png', dpi=150); plt.show(); plt.close(fig)
best = df.sort_values(['silhouette_comun','DB_comun'], ascending=[False,True]).iloc[0]
view(np.load(LOGS / 'clase_labels' / f'{best.id}.npy'), 'mejor', f'k-means · {best.id}')
finish({'mejor': best.to_dict(), 'ari_min': float(df.ARI.min()),
        'ari_max': float(df.ARI.max()), 'configuraciones': len(df)})
''')]
    save(NAMES[1], c)

    c = base('''# 03 · DBSCAN: vecinos, densidad y ruido
    **Objetivo.** Elegir eps con una curva k-distance y comprobar qué ocurre al mover
    eps y minPts. DBSCAN no recibe K: conecta regiones densas y deja ruido (-1).''', helpers=True, nb='03')
    c += [md('### ¿A qué distancia está el vecino número 10?\nUsamos xyz estandarizado para estudiar continuidad espacial. La consulta incluye el propio punto, igual que `min_samples` en DBSCAN. Ordenamos la distancia número 10 y marcamos cuantiles; son candidatos razonables para explorar, no un eps demostrado.'), code('''
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
X = features('xyz')
neighbors = NearestNeighbors(n_neighbors=10, n_jobs=2).fit(X)
distances, _ = neighbors.kneighbors(X)
kdist = np.sort(distances[:,-1])
q = np.quantile(kdist, [.50,.85,.97])
eps_values = np.unique(np.r_[q[0]*.25, q, q[-1]*1.5])
fig, ax = plt.subplots(figsize=(8,3.5))
ax.plot(np.linspace(0,1,len(kdist)), kdist)
for e in eps_values: ax.axhline(e, alpha=.4, label=f'eps={e:.4f}')
ax.set(xlabel='Fracción ordenada', ylabel='Distancia al vecino 10', title='k-distance'); ax.legend()
fig.tight_layout(); fig.savefig(FIG / '03_kdistance.png', dpi=150); plt.show(); plt.close(fig)
print('eps candidatos:', eps_values)
'''), md('### Una grilla pequeña, con un control de memoria\nProbamos dos minPts por eps. Antes contamos vecinos en 1500 puntos para evitar un vecindario demasiado denso. Si supera 1200 vecinos medios reducimos eps y lo anotamos en el CSV. DBSCAN es determinista con este orden fijo: su ARI entre semillas es 1 por construcción; eso no prueba estabilidad ante cambiar eps ni ante remuestrear.'), code('''
probe = np.random.default_rng(0).choice(len(X), min(1500,len(X)), replace=False)
for proposed in eps_values:
    eps = float(proposed)
    while True:
        neigh = neighbors.radius_neighbors(X[probe], radius=eps, return_distance=False)
        mean_neighbors = float(np.mean([len(v) for v in neigh]))
        if mean_neighbors <= 1200: break
        eps *= .8
    for m in (5, 15):
        tick = time.perf_counter()
        labels = DBSCAN(eps=eps, min_samples=m, n_jobs=2).fit_predict(X)
        elapsed = time.perf_counter()-tick
        ari, sd = stability_ari(lambda data, seed: labels, X, [0,1,2])
        record('dbscan', 'xyz', True, {'eps':eps, 'min_samples':m}, X, labels, elapsed,
               ari, sd, eps_propuesto=float(proposed), vecinos_medios=mean_neighbors)
df = table(); display(df[['parametros','n_clusters','ruido','silhouette','DB','silhouette_comun']])
'''), md('### Una elección útil y un extremo\nElegimos el mayor silhouette común entre particiones con 2–30 grupos y ruido ≤40%; si ninguna cumple, mostramos la mejor válida y dejamos constancia. Contrastamos con el eps más pequeño. Las métricas excluyen ruido: siempre hay que leerlas junto con su porcentaje.'), code('''
valid = df[df.silhouette_comun.notna() & df.n_clusters.between(2,30) & (df.ruido <= .4)]
fallback = valid.empty
if fallback: valid = df[df.silhouette_comun.notna()]
assert not valid.empty, 'La grilla necesita ampliarse'
best = valid.sort_values('silhouette_comun', ascending=False).iloc[0]
worst = df.iloc[0]
for suffix, row in [('mejor',best),('eps_chico',worst)]:
    view(np.load(LOGS / 'clase_labels' / f'{row.id}.npy'), suffix,
         f'DBSCAN · {row.id} · ruido {row.ruido:.1%}')
fig, ax = plt.subplots(1,2,figsize=(10,3.5))
for m in (5,15):
    p = df[[json.loads(s)['min_samples']==m for s in df.parametros]]
    es = [json.loads(s)['eps'] for s in p.parametros]
    ax[0].plot(es,p.ruido,'o-',label=f'minPts={m}')
    ax[1].plot(es,p.n_clusters,'o-',label=f'minPts={m}')
ax[0].set(xlabel='eps',ylabel='Fracción de ruido'); ax[1].set(xlabel='eps',ylabel='Clusters')
ax[0].legend(); fig.tight_layout(); fig.savefig(FIG/'03_grilla.png',dpi=150); plt.show(); plt.close(fig)
finish({'mejor':best.to_dict(), 'extremo':worst.to_dict(), 'fallback':fallback,
        'configuraciones':len(df)})
''')]
    save(NAMES[2], c)

    c = base('''# 04 · AgglomerativeClustering y GaussianMixture
    **Objetivo.** Comparar cómo se unen grupos con single/complete/average/ward
    y cómo GMM describe nubes con covarianza full o diag. Reducimos el ajuste
    para respetar memoria y tiempo, y evaluamos las predicciones en la muestra común.''', helpers=True, nb='04')
    c += [md('### Para la jerarquía, una muestra de 3500 puntos\nUnimos primero puntos o grupos cercanos. Single mira el par más cercano; complete, el más lejano; average promedia distancias; ward minimiza el aumento de WCSS. Usamos distancias completas, sin conectividad kNN, para conservar esas definiciones. Probamos K=2,4,6,8 y registramos métricas locales, además de las comunes tras propagar por vecino espacial.'), code('''
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
sub_idx = np.sort(np.random.default_rng(0).choice(len(xyz), min(3500,len(xyz)), replace=False))
np.save(CACHE/'clase_jerarquico_idx.npy',sub_idx)
X = features('xyz')
for linkage in ('single','complete','average','ward'):
    for k in (2,4,6,8):
        tick = time.perf_counter()
        small_labels = AgglomerativeClustering(n_clusters=k,linkage=linkage).fit_predict(X[sub_idx])
        labels = propagate_labels(xyz[sub_idx],small_labels,xyz)
        elapsed = time.perf_counter()-tick
        ari, sd = stability_ari(lambda data, seed: labels, X, [0,1,2])
        record('agglomerative','xyz',True,{'n_clusters':k,'linkage':linkage},X,labels,
               elapsed,ari,sd, n_ajuste=len(sub_idx),
               silhouette_ajuste=silhouette_sample(X[sub_idx],small_labels,n=1200,seed=0),
               DB_ajuste=davies_bouldin(X[sub_idx],small_labels))
print('Jerarquía terminada')
'''), md('### GMM y BIC: ajustar mejor también cuesta parámetros\nUna mezcla gaussiana asigna probabilidades a componentes elípticas. Full permite correlaciones; diag solo varianzas por feature. Ajustamos 6000 puntos de las features completas, predecimos la muestra voxel y medimos estabilidad con semillas 0,1,2. BIC se calcula sobre los mismos 6000 puntos: menor es mejor dentro de esta representación, y penaliza complejidad. No lo comparamos con WCSS.'), code('''
gmm_idx = np.sort(np.random.default_rng(0).choice(len(xyz),min(6000,len(xyz)),replace=False))
np.save(CACHE/'clase_gmm_idx.npy',gmm_idx)
X = features('xyz+rgb+normales+height')
for cov in ('full','diag'):
    for k in range(2,9):
        tick = time.perf_counter()
        runs = {s: GaussianMixture(n_components=k,covariance_type=cov,random_state=s,
                    n_init=1,max_iter=200,reg_covar=1e-5).fit(X[gmm_idx]) for s in (0,1,2)}
        predictions = {s: model.predict(X) for s,model in runs.items()}
        ari, sd = stability_ari(lambda data, seed: predictions[seed],X,[0,1,2])
        elapsed = time.perf_counter()-tick
        record('gmm','xyz+rgb+normales+height',True,
               {'n_components':k,'covariance_type':cov,'n_init':1,'max_iter':200,'reg_covar':1e-5},
               X,predictions[0],elapsed,ari,sd,n_ajuste=len(gmm_idx),
               BIC=runs[0].bic(X[gmm_idx]), converged=all(m.converged_ for m in runs.values()))
    print('GMM',cov,'listo',flush=True)
df = table()
df['seleccionado_bic'] = False
for cov in ('full','diag'):
    mask = (df.metodo=='gmm') & df.parametros.str.contains(f'"covariance_type": "{cov}"')
    df.loc[df.loc[mask,'BIC'].idxmin(),'seleccionado_bic'] = True
df.to_csv(OUT/'metricas_04.csv',index=False)
display(df[['metodo','parametros','silhouette','DB','silhouette_comun','ARI','BIC','seleccionado_bic']])
'''), md('### Contrastemos los criterios\nLa jerarquía es determinista para estos puntos: ARI=1 entre semillas no mide sensibilidad al submuestreo. Para GMM señalamos el mínimo BIC de cada covarianza; silhouette y DB evalúan la partición dura, que no es el objetivo de máxima likelihood de GMM.'), code('''
gmm = df[df.metodo=='gmm']
fig, ax = plt.subplots(1,2,figsize=(11,3.5))
for cov in ('full','diag'):
    p = gmm[[json.loads(s)['covariance_type']==cov for s in gmm.parametros]]
    ax[0].plot([json.loads(s)['n_components'] for s in p.parametros],p.BIC,'o-',label=cov)
    display(p.loc[[p.BIC.idxmin()],['parametros','BIC','silhouette','DB','ARI']])
ax[0].set(xlabel='K',ylabel='BIC'); ax[0].legend()
for method in ('agglomerative','gmm'):
    p=df[df.metodo==method]
    ax[1].scatter(p.DB_comun,p.silhouette_comun,label=method)
ax[1].set(xlabel='DB común (menor mejor)',ylabel='Silhouette común'); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG/'04_comparacion.png',dpi=150); plt.show(); plt.close(fig)
best = df.sort_values('silhouette_comun',ascending=False).iloc[0]
view(np.load(LOGS/'clase_labels'/f'{best.id}.npy'),'mejor',f'Jerarquía / GMM · {best.id}')
finish({'mejor':best.to_dict(),'bic_mejor':gmm.loc[gmm.BIC.idxmin()].to_dict(),
        'configuraciones':len(df),'gmm_no_convergentes':int((~gmm.converged.astype(bool)).sum())})
''')]
    save(NAMES[3], c)

    c = base('''# 05 · PCA y mapas para visualizar
    **Objetivo.** Entender qué conservan PCA, t-SNE y (si está disponible) UMAP.
    Los colores vienen de k-means; no haremos clustering sobre el mapa ni
    tomaremos sus islas como prueba de objetos reales.''')
    c += [md('### PCA: las unidades cambian qué significa varianza\nPCA encuentra direcciones de mayor varianza. Comparamos las features completas con y sin estandarizar. Los coeficientes (`components_.T`) muestran qué aporta cada variable a cada dirección; el signo de un eje se puede invertir sin cambiar la proyección.'), code(FEATURES), code('''
from sklearn.decomposition import PCA
names = ['x','y','z','L*','a*','b*','nx','ny','nz','height']
pca_info = {}
fig, ax = plt.subplots(2,2,figsize=(12,8))
for j, scaled in enumerate((False,True)):
    X = features('xyz+rgb+normales+height',scaled)
    pca = PCA(svd_solver='full',random_state=0).fit(X)
    ratio = pca.explained_variance_ratio_
    pca_info[str(scaled)] = {'dos_pc':float(ratio[:2].sum()),
        'pc_90':int(np.searchsorted(np.cumsum(ratio),.90)+1), 'ratio':ratio.tolist()}
    ax[0,j].bar(range(1,len(ratio)+1),ratio,label='Individual')
    ax[0,j].plot(range(1,len(ratio)+1),np.cumsum(ratio),'o-',label='Acumulada')
    ax[0,j].set(xlabel='PC',ylabel='Varianza explicada',title=f'Estandarizado={scaled}'); ax[0,j].legend()
    im=ax[1,j].imshow(pca.components_[:3].T,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
    ax[1,j].set(yticks=range(10),yticklabels=names,xticks=range(3),xticklabels=['PC1','PC2','PC3'],title='Coeficientes de las direcciones')
    fig.colorbar(im,ax=ax[1,j])
    pd.DataFrame(pca.components_.T,index=names).to_csv(CACHE/f'clase_pca_loadings_{scaled}.csv')
fig.tight_layout(); fig.savefig(FIG/'05_pca_varianza_loadings.png',dpi=150); plt.show(); plt.close(fig)
print(pca_info)
'''), md('### Un mapa de 3000 puntos con colores fijos\nElegimos la corrida de k-means con mayor silhouette común, recuperamos sus etiquetas y dibujamos las mismas filas en todos los mapas. t-SNE preserva relaciones locales y puede deformar tamaños y distancias entre grupos. Probamos perplexity 15 y 40 para comprobar sensibilidad. Si no está umap-learn, lo omitimos sin instalar nada.'), code('''
from sklearn.manifold import TSNE, trustworthiness
km = pd.read_csv(OUT/'metricas_02.csv').sort_values('silhouette_comun',ascending=False).iloc[0]
labels = np.load(LOGS/'clase_labels'/f'{km.id}.npy')
vis_idx = np.sort(np.random.default_rng(0).choice(len(xyz),min(3000,len(xyz)),replace=False))
np.save(CACHE/'clase_visualizacion_idx.npy',vis_idx)
Xvis = EVAL[vis_idx]
colors = segment_rgb(labels[vis_idx])/255
maps = {'PCA':PCA(n_components=2,svd_solver='full',random_state=0).fit_transform(Xvis)}
for perplexity in (15,40):
    maps[f't-SNE p={perplexity}'] = TSNE(n_components=2,perplexity=perplexity,
        init='pca',learning_rate='auto',max_iter=750,random_state=0).fit_transform(Xvis)
umap_available = importlib.util.find_spec('umap') is not None
if umap_available:
    from umap import UMAP
    maps['UMAP'] = UMAP(n_components=2,n_neighbors=15,min_dist=.1,random_state=0,n_jobs=1).fit_transform(Xvis)
else:
    print('umap-learn no está instalado: usamos solo t-SNE de sklearn; no se instala nada.')
'''), md('### ¿Cuánto se puede confiar en el vecindario dibujado?\nTrustworthiness con 10 vecinos mide cuánto introduce el mapa vecinos que no eran cercanos en las features de origen. Es una medida del dibujo, no de calidad del clustering. PCA usa aquí las mismas 3000 filas para que la comparación sea directa.'), code('''
fig, axes = plt.subplots(1,len(maps),figsize=(5*len(maps),4))
scores = {}
for ax,(name,Z) in zip(np.atleast_1d(axes),maps.items()):
    scores[name] = float(trustworthiness(Xvis,Z,n_neighbors=10))
    ax.scatter(Z[:,0],Z[:,1],c=colors,s=5,alpha=.6,linewidths=0)
    ax.set(title=f'{name} · T={scores[name]:.3f}',xlabel='Eje 1',ylabel='Eje 2')
fig.tight_layout(); fig.savefig(FIG/'05_mapas.png',dpi=150); plt.show(); plt.close(fig)
payload = dict(pca=pca_info, trustworthiness=scores, umap=umap_available,
               kmeans_id=km.id,n_visualizacion=len(vis_idx),segundos_notebook=time.perf_counter()-START)
(CACHE/'clase_resumen_05.json').write_text(json.dumps(payload,indent=2))
print(payload)
''')]
    save(NAMES[4], c)

    c = base('''# 06 · Resultado de las técnicas de clase
    **Objetivo.** Elegir una partición con suficiente detalle para explorar columnas,
    tuberías, piso, techo, vigas y equipos, sin confundir silhouette alto con utilidad.
    Volvemos a ajustar y propagamos a los 1 647 021 puntos conservando normales.
    Los clusters pueden mezclar estructuras: no tenemos ground truth.''', helpers=True, nb='06')
    c += [md('''### La selección ahora responde al objetivo
    La primera regla permitía K=2 y acabó separando una franja superior del resto.
    Ese resultado podía tener silhouette alto y seguir siendo demasiado grueso.
    Conservamos todas las pruebas, incluidas K pequeños, para medir ese hallazgo.

    Exigimos **6–15 clusters, ruido ≤30% y ningún cluster >60% de la muestra**.
    Necesitamos silhouette común, DB común y ARI finitos para calcular el ranking.
    No añadimos un umbral mínimo de ARI ni de tamaño de cada cluster. BIC sigue
    siendo el diagnóstico de GMM en 04; aquí comparamos todas las configuraciones
    que cumplen estos límites con el mismo ranking, independientemente del método.

    Si no hay candidatos, relajamos primero el tamaño máximo a **70%**, conservando
    ruido ≤30%; solo si aún no hay candidatos permitimos **ruido ≤50%**, manteniendo
    el máximo de 70%. Nunca relajamos el rango 6–15. Registramos la etapa aplicada.

    Elegimos la media de tres ranks con igual peso: `silhouette_comun` descendente,
    `DB_comun` ascendente y `ARI` descendente. Desempatamos por silhouette común e ID.
    Las fracciones del filtro corresponden a la muestra voxel; también revisaremos
    tamaños en la nube completa tras propagar. ARI de métodos deterministas sigue
    siendo una limitación: 1 no demuestra estabilidad ante remuestreo.'''), code('''
results = pd.concat([pd.read_csv(OUT/f'metricas_{n}.csv') for n in ('02','03','04')],ignore_index=True)
finite = np.isfinite(results[['silhouette_comun','DB_comun','ARI']]).all(axis=1)
base_mask = finite & results.n_clusters.between(6,15)
levels = [('estricto',.6,.3), ('max_cluster_70',.7,.3), ('ruido_50',.7,.5)]
selection_steps = []
for selection_stage, max_frac, max_noise in levels:
    mask = base_mask & (results.max_cluster_frac<=max_frac) & (results.ruido<=max_noise)
    selection_steps.append(dict(etapa=selection_stage,max_cluster_frac=max_frac,
                                ruido=max_noise,candidatos=int(mask.sum())))
    if mask.any(): break
assert mask.any(), 'No hay candidatos ni tras las dos relajaciones: revisar experimentos'
results['elegible_estricto'] = base_mask & (results.max_cluster_frac<=.6) & (results.ruido<=.3)
results['elegible'] = mask
results['etapa_seleccion'] = selection_stage
results['limite_max_cluster_frac'] = max_frac
results['limite_ruido'] = max_noise
eligible = results[mask].copy()
eligible['rank_medio'] = (eligible.silhouette_comun.rank(ascending=False)
    + eligible.DB_comun.rank(ascending=True) + eligible.ARI.rank(ascending=False))/3
results['rank_medio'] = eligible.rank_medio
best = eligible.sort_values(['rank_medio','silhouette_comun','id'],ascending=[True,False,True]).iloc[0]
results['seleccionada'] = results.id.eq(best.id)
results.to_csv(ENT/'clase_metricas.csv',index=False)
display(pd.DataFrame(selection_steps))
display(eligible.sort_values(['rank_medio','silhouette_comun','id'],ascending=[True,False,True])[
    ['id','metodo','features','parametros','n_clusters','ruido','max_cluster_frac',
     'silhouette_comun','DB_comun','ARI','rank_medio']].head(10))
print('Elegido:',best.id,'etapa:',selection_stage,'candidatos:',len(eligible))
'''), md('''### Comparemos con los K pequeños que parecían mejores
    La tabla contrasta la configuración nueva con el mayor silhouette común de K=2
    y con complete K=2 (la elección anterior). Si el número mejora al agrupar casi
    todo en dos colores, todavía no satisface el detalle que buscamos. Esta es una
    limitación del criterio, no una razón para ocultar esos resultados.'''), code('''
k2 = results[results.n_clusters.eq(2)].sort_values('silhouette_comun',ascending=False).iloc[0]
previous = results[(results.metodo=='agglomerative') & results.n_clusters.eq(2)
                   & results.parametros.str.contains('"linkage": "complete"')].iloc[0]
comparison = pd.DataFrame([k2,previous,best])
display(comparison[['id','metodo','n_clusters','silhouette_comun','DB_comun','ARI','max_cluster_frac']])
comparison.to_csv(CACHE/'clase_comparacion_k2.csv',index=False)
'''), md('''### Repetimos el ajuste elegido
    Usamos los mismos hiperparámetros y semilla 0. Si corresponde recuperamos las
    submuestras de jerarquía o GMM. Jerarquía no tiene predict: primero propagamos
    de sus representantes a la muestra voxel. ARI=1 contra las etiquetas guardadas
    comprueba que el reajuste reprodujo esa corrida.'''), code('''
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
params = json.loads(best.parametros)
X = features(best.features, bool(best.estandarizado))
tick = time.perf_counter()
if best.metodo=='kmeans':
    labels = KMeans(**params,random_state=0).fit_predict(X)
elif best.metodo=='dbscan':
    labels = DBSCAN(**params,n_jobs=2).fit_predict(X)
elif best.metodo=='gmm':
    idx = np.load(CACHE/'clase_gmm_idx.npy')
    labels = GaussianMixture(**params,random_state=0).fit(X[idx]).predict(X)
elif best.metodo=='agglomerative':
    idx = np.load(CACHE/'clase_jerarquico_idx.npy')
    lab = AgglomerativeClustering(**params).fit_predict(X[idx])
    labels = propagate_labels(xyz[idx],lab,xyz)
else:
    raise ValueError(best.metodo)
refit_seconds = time.perf_counter()-tick
saved = np.load(LOGS/'clase_labels'/f'{best.id}.npy')
refit_ari = float(adjusted_rand_score(labels,saved))
assert refit_ari == 1.0
assert 6 <= cluster_summary(labels)['n_clusters'] <= 15
print('Reajuste (s):',refit_seconds,'ARI respecto a la corrida guardada:',refit_ari)
'''), md('''### Etiquetas y colores para todos los puntos
    El vecino espacial más cercano (`k=1`) lleva las etiquetas a la nube completa.
    Esta aproximación puede desplazar fronteras. Conservamos orden, coordenadas
    y normales, y usamos la misma paleta contrastante del notebook. El ruido es gris.
    Releemos el PLY para verificar conservación y al menos seis colores distintos.'''), code('''
tick = time.perf_counter()
labels_all = propagate_labels(xyz,labels,cloud['xyz'],k=1)
np.save(CACHE/'clase_labels_full.npy',labels_all.astype(np.int32))
write_ply(ENT/'clase_segmentos.ply',cloud['xyz'],segment_rgb(labels_all),cloud['normals'])
propagate_seconds = time.perf_counter()-tick
check = read_ply(ENT/'clase_segmentos.ply')
assert np.array_equal(check['xyz'],cloud['xyz'])
assert np.array_equal(check['normals'],cloud['normals'])
assert np.array_equal(check['rgb'],segment_rgb(labels_all))
assert check['rgb'].dtype == np.uint8
assert len(labels_all)==1647021
assert np.array_equal(labels_all[sample_idx],labels)
color_count = len(np.unique(check['rgb'],axis=0))
assert color_count >= 6
assert cluster_summary(labels_all)['n_clusters'] == int(best.n_clusters)
print('PLY verificado:',len(labels_all),'puntos, mismas coordenadas y normales;',color_count,'colores')
print('Muestra:',cluster_summary(labels)); print('Nube completa:',cluster_summary(labels_all))
'''), md('''### Revisemos el conjunto antes de separar los paneles
    La figura de la muestra y la de la nube completa usan puntos más visibles con
    alpha=0.6. Los colores identifican clusters; no implican categorías semánticas.'''), code('''
view(labels,'muestra',f'Resultado de clase · {best.metodo} · K={int(best.n_clusters)} · muestra')
path = FIG/'06_completo.png'
scatter_views(cloud['xyz'],labels_all,path,f'Resultado propagado · {best.metodo} · K={int(best.n_clusters)}',max_points=60000,seed=0)
display(Image(filename=str(path)))
'''), md('''### Ahora, cada cluster por separado: ¿qué parte captura?
    Cada panel lateral mantiene los mismos ejes x–z y muestra un solo cluster sobre
    una referencia gris tenue de la fábrica. El título indica el tamaño **completo**,
    aunque limitemos el dibujo a 12 mil puntos por cluster. La proyección superpone
    posiciones con distinto y: usaremos también los rangos y las vistas 3D para leerla.
    No asignamos nombres de objetos a partir del color: la interpretación visual
    posterior es tentativa y debe reconocer cuando un grupo mezcla estructuras.'''), code('''
cluster_ids = np.unique(labels_all[labels_all>=0])
cluster_rows = []
for label in cluster_ids:
    keep = labels_all==label
    pts = cloud['xyz'][keep]
    row = dict(cluster=int(label),n_puntos=int(keep.sum()),fraccion=float(keep.mean()),
               nz_mediana=float(np.median(cloud['normals'][keep,2])))
    for j,axis in enumerate('xyz'):
        for q,value in zip(('05','50','95'),np.quantile(pts[:,j],[.05,.5,.95])):
            row[f'{axis}_q{q}'] = float(value)
    cluster_rows.append(row)
cluster_table = pd.DataFrame(cluster_rows)
cluster_table.to_csv(CACHE/'clase_clusters.csv',index=False)
display(cluster_table)
'''), code('''
ncols = 3; nrows = int(np.ceil(len(cluster_ids)/ncols))
fig, axes = plt.subplots(nrows,ncols,figsize=(17,3.5*nrows),squeeze=False,sharex=True,sharey=True)
rng = np.random.default_rng(0)
context_idx = rng.choice(len(labels_all),10000,replace=False)
context = cloud['xyz'][context_idx]
for ax,label in zip(axes.flat,cluster_ids):
    idx = np.flatnonzero(labels_all==label)
    size = len(idx)
    if len(idx)>12000: idx = rng.choice(idx,12000,replace=False)
    pts = cloud['xyz'][idx]
    color = segment_rgb(np.array([label]))[0]/255.
    ax.scatter(context[:,0],context[:,2],c='gray',s=.5,alpha=.10,linewidths=0)
    ax.scatter(pts[:,0],pts[:,2],color=color,s=2.2,alpha=.6,linewidths=0)
    ax.set(title=f'Cluster {label} · {size:,} puntos ({size/len(labels_all):.1%})',
           xlabel='x',ylabel='z',aspect='equal')
for ax in list(axes.flat)[len(cluster_ids):]: ax.set_visible(False)
fig.suptitle(f'Clusters por separado · vista lateral x–z · {best.id}',y=1.002)
fig.tight_layout(); fig.savefig(FIG/'06_clusters_separados.png',dpi=180,bbox_inches='tight'); plt.show(); plt.close(fig)
finish({'mejor':best.to_dict(),'configuraciones':len(results),'elegibles':len(eligible),
        'etapa_seleccion':selection_stage,'pasos_seleccion':selection_steps,
        'referencia_k2':k2.to_dict(),'seleccion_anterior_k2':previous.to_dict(),
        'refit_ari':refit_ari,'refit_seconds':refit_seconds,'propagate_seconds':propagate_seconds,
        'n_colores':color_count,'clusters':cluster_rows,
        'full_summary':cluster_summary(labels_all),'sample_summary':cluster_summary(labels)})
''')]
    save(NAMES[5], c)


def conclusions():
    """Se llama únicamente después de ejecutar los seis notebooks y leer sus cifras."""
    data = {f'{i:02}':json.loads((ROOT/f'outputs/cache/clase_resumen_{i:02}.json').read_text()) for i in range(1,7)}
    a,b,d,h,p,r = (data[f'{i:02}'] for i in range(1,7))
    def metrics(row):
        return (f"silhouette común {row['silhouette_comun']:.3f}, DB común {row['DB_comun']:.3f} "
                f"y ARI {row['ARI']:.4f}")
    texts = [
        f"Con {a['n_original']:,} puntos, z va de {a['z_min']:.3f} a {a['z_max']:.3f}. "
        f"El voxel {a['voxel']:.5f} dejó {a['n_sample']:,} representantes, dentro del rango pedido. "
        "Conservamos puntos reales y sus normales, pero la muestra da un peso más parejo al espacio que a la densidad original. "
        "Los histogramas describen distribuciones, no clases; todavía no sabemos cuáles son objetos físicos.",
        f"Probamos {b['configuraciones']} configuraciones. Por silhouette común destacó {b['mejor']['id']} "
        f"({b['mejor']['features']}, estandarizado={b['mejor']['estandarizado']}, "
        f"{b['mejor']['parametros']}): {metrics(b['mejor'])}. "
        f"El ARI entre starts varió de {b['ari_min']:.3f} a {b['ari_max']:.3f}; "
        "por lo tanto, hay configuraciones más sensibles a la inicialización que otras. "
        "No podemos comparar WCSS de distintas escalas ni interpretar el mayor silhouette como segmentación semántica correcta.",
        f"En las {d['configuraciones']} pruebas, la elección de la grilla fue {d['mejor']['id']}: "
        f"{d['mejor']['n_clusters']} clusters, ruido {d['mejor']['ruido']:.1%}, {metrics(d['mejor'])}. "
        f"El extremo de eps pequeño dejó {d['extremo']['ruido']:.1%} de ruido y "
        f"{d['extremo']['n_clusters']} clusters. "
        + ('Fue necesario relajar el filtro para la figura. ' if d['fallback'] else 'No fue necesario relajar el filtro para la figura. ')
        + f"El grupo mayor reúne {d['mejor']['max_cluster_frac']:.1%} de la muestra: la figura está dominada por un solo color. "
        "Los puntos de ruido no entran en silhouette/DB; ignorar su fracción daría una comparación engañosa. "
        "ARI=1 es consecuencia del ajuste determinista y no demuestra robustez a eps.",
        f"Comparamos {h['configuraciones']} configuraciones. El mayor silhouette común de este notebook fue "
        f"{h['mejor']['id']} ({h['mejor']['metodo']}): {metrics(h['mejor'])}. "
        f"El menor BIC de GMM fue {h['bic_mejor']['BIC']:.1f} con {h['bic_mejor']['parametros']}; "
        f"su silhouette común fue {h['bic_mejor']['silhouette_comun']:.3f}. "
        f"Hubo {h['gmm_no_convergentes']} configuraciones de GMM sin convergencia en alguna semilla. "
        "BIC eligió K=8 para full y diag, el borde de la grilla 2–8; no sabemos si seguiría bajando fuera de ella. "
        "En 06 aplicamos el nuevo filtro de detalle y balance a todas las configuraciones; BIC queda como diagnóstico de GMM. "
        "La jerarquía se ajustó en 3500 puntos y GMM en 6000: las métricas comunes incluyen la extensión a la muestra voxel. "
        "Un ARI determinista de 1 no mide sensibilidad a esas submuestras.",
        f"Sin estandarizar, dos PC explicaron {p['pca']['False']['dos_pc']:.1%} de la varianza; "
        f"estandarizando explicaron {p['pca']['True']['dos_pc']:.1%}. "
        f"Para 90% se necesitaron {p['pca']['False']['pc_90']} y {p['pca']['True']['pc_90']} PC respectivamente. "
        f"En {p['n_visualizacion']} puntos, trustworthiness fue "
        + ', '.join(f"{k}: {v:.3f}" for k,v in p['trustworthiness'].items()) + '. '
        + ("También se ejecutó UMAP. " if p['umap'] else "UMAP no estaba instalado y no instalamos paquetes. ")
        + "La última PC tiene varianza prácticamente cero, coherente con que height duplica z tras centrar. "
        + "En los loadings sin escala, PC1 está dominada por L*; estandarizando, las normales también aportan. "
        + "Los colores son etiquetas de k-means, no ground truth. Los mapas ayudan a explorar, pero no validan el número de grupos.",
        f"De {r['configuraciones']} configuraciones, {r['elegibles']} cumplieron el filtro aplicado, en la etapa {r['etapa_seleccion']}. "
        + ('No hizo falta relajar los límites de 6–15 clusters, ruido ≤30% y cluster mayor ≤60%. ' if r['etapa_seleccion']=='estricto' else f"Se aplicó la relajación {r['etapa_seleccion']}: {r['pasos_seleccion']}. ")
        + f"El mejor silhouette común con K=2 fue {r['referencia_k2']['silhouette_comun']:.4f}; la elección anterior complete K=2 tenía {r['seleccion_anterior_k2']['silhouette_comun']:.4f}. "
        "Esos números más altos no daban el detalle buscado: techo frente al resto es una separación demasiado gruesa. "
        f"La regla de ranks eligió {r['mejor']['id']} ({r['mejor']['metodo']}, "
        f"{r['mejor']['features']}, {r['mejor']['parametros']}): {metrics(r['mejor'])}. "
        f"El reajuste reprodujo las etiquetas con ARI={r['refit_ari']:.3f}. "
        f"Al propagar quedaron {r['full_summary']['n_clusters']} grupos y "
        f"{r['full_summary']['noise_frac']:.1%} de ruido en la nube completa. "
        "Verificamos por relectura los 1 647 021 puntos, su orden, sus coordenadas, los colores uchar y las normales. "
        "El resultado es una partición según las features y el criterio elegido; sin ground truth no podemos asegurar que cada grupo sea un objeto. "
        f"El nuevo PLY tiene {r['n_colores']} colores distintos y su cluster mayor ocupa {max(r['full_summary']['sizes'].values())/1647021:.1%} de los puntos completos. "
        "Los paneles separados permiten revisar qué partes se mezclan: más colores no garantiza clases semánticas. "
        "La propagación espacial y las submuestras de ajuste pueden suavizar o desplazar fronteras."
    ]
    for name, text in zip(NAMES,texts):
        path=DEST/f'{name}.ipynb'
        nb=nbf.read(path,as_version=4)
        if name == NAMES[5]:
            inspection_path = ROOT/'outputs/cache/clase_inspeccion_visual.json'
            if inspection_path.exists():
                inspection = json.loads(inspection_path.read_text())
                assert inspection['configuracion'] == r['mejor']['id']
                assert inspection['sizes'] == r['full_summary']['sizes']
                text += '\n\nLectura tentativa de las figuras (los grupos pueden mezclar estructuras):\n\n'
                text += '| Cluster | Puntos completos | Qué se observa |\n|---|---:|---|\n'
                for label, size in r['full_summary']['sizes'].items():
                    text += f"| {label} | {size:,} | {inspection['clusters'][label]} |\n"
        nb.cells[-1].source='## Qué aprendimos / qué falló\n\n'+text
        mode = nb.metadata.get('clase',{}).get('ejecucion','nbconvert')
        if 'IPython' in mode:
            note = '\n\n**Nota de ejecución.** Celdas ejecutadas en orden con IPython dentro de `.venv`, porque nbconvert no pudo abrir los sockets del kernel en este sandbox. El usuario había verificado la versión previa con nbconvert fuera de esta restricción. Logs de esta revisión en `outputs/logs/`; auxiliares en `outputs/cache/`.'
        else:
            note = '\n\n**Nota de ejecución.** Notebook ejecutado con nbconvert dentro de `.venv`. Logs en `outputs/logs/`; auxiliares en `outputs/cache/`.'
        nb.cells[0].source = nb.cells[0].source.split('\n\n**Nota de ejecución.**')[0] + note
        nbf.write(nb,path)
        print(name, text)

def execute_inprocess(name):
    """Alternativa explícita cuando el sandbox impide iniciar kernels de Jupyter.

    IPython ejecuta las celdas reales en orden; cada notebook usa un proceso nuevo.
    No simula una ejecución por nbconvert ni oculta excepciones.
    """
    import os
    import time
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    from nbformat.v4 import new_output
    path = DEST / f'{name}.ipynb'
    nb = nbf.read(path, as_version=4)
    os.chdir(DEST)
    shell = TerminalInteractiveShell.instance()
    shell.run_line_magic('matplotlib', 'inline')
    start = time.perf_counter()
    count = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'{name} celda {count}', flush=True)
        with capture_output(display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(new_output('display_data', data=output.data, metadata=output.metadata))
        if result.error_before_exec or result.error_in_exec:
            error = result.error_before_exec or result.error_in_exec
            print(captured.stdout, captured.stderr, flush=True)
            raise RuntimeError(f'{name}, celda {count}: {error}') from error
        # Checkpoint de celdas realmente ejecutadas.
        nbf.write(nb, path)
    nb.metadata['clase']['ejecucion'] = 'IPython in-process; nbconvert bloqueado por sockets del sandbox'
    nb.metadata['clase']['segundos_ejecucion'] = time.perf_counter() - start
    nbf.validate(nb)
    nbf.write(nb, path)
    print('COMPLETE', name, nb.metadata['clase']['segundos_ejecucion'], flush=True)

if __name__=='__main__':
    if '--execute-inprocess' in sys.argv:
        execute_inprocess(sys.argv[sys.argv.index('--execute-inprocess')+1])
    elif '--conclusions' in sys.argv:
        conclusions()
    else:
        build()
