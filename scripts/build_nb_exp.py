"""Generador de experimentación: orientación local, sin modificar clase."""
from pathlib import Path
import sys
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'notebooks/experimentacion'
DEST.mkdir(exist_ok=True)
M=nb.v4.new_markdown_cell
C=nb.v4.new_code_cell

def write(name,cells):
    if '--execute' in sys.argv: return
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (.venv)','language':'python','name':'python3'}})
    nb.write(n,DEST/f'{name}.ipynb')

COMMON = r'''import os, sys, json, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path.cwd() if Path('src/pcseg').exists() else Path.cwd().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/cache/exp_matplotlib')
os.environ['IPYTHONDIR']=str(ROOT/'outputs/cache/exp_ipython')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, Image
from threadpoolctl import threadpool_limits
from pcseg.io import read_ply, write_ply, labels_to_rgb
from pcseg.features import voxel_downsample, build_features
from pcseg.metrics import silhouette_sample, davies_bouldin
from pcseg.stages import *
plt.switch_backend('Agg')
thread_limit = threadpool_limits(limits=2)
OUT=ROOT/'outputs'; CACHE=OUT/'cache'; FIG=OUT/'figs'; LOG=OUT/'logs'; ENT=ROOT/'entregable'; ENT.mkdir(exist_ok=True)
START=time.perf_counter()
cloud=read_ply(ROOT/'data/FARO_Scan_Full_Sub1_down3_with_normals.ply')
common_idx=np.load(OUT/'sample_idx.npy')
xyz,rgb,normals=(cloud[k][common_idx] for k in ('xyz','rgb','normals'))
EVAL=build_features(xyz,rgb,normals,use_rgb=True,use_normals=True,use_height=True)
base_table=pd.read_csv(ENT/'clase_metricas.csv')
base=base_table[base_table.seleccionada].iloc[0]
base_labels=np.load(LOG/'clase_labels'/f'{base.id}.npy')
assert len(xyz)==93903 and len(cloud['xyz'])==1647021
print('Muestra común',len(xyz),'Python',sys.executable,sys.version.split()[0])
CLASS_COLORS={'piso':(70,155,70),'techo':(90,140,230),'pared':(190,145,90),
 'plano_horizontal':(145,190,220),'columna':(230,60,55),'viga_horizontal':(245,170,35),
 'viga_o_tuberia':(225,125,25),'tuberia':(25,180,185),
 'equipo_bajo':(155,75,190),'compacto_plausible':(30,200,190),
 'mezcla_no_identificada':(210,110,160),'ruido':(45,45,45)}
def view(points,labels,name,title,classes=None):
    idx=np.random.default_rng(0).choice(len(points),min(35000,len(points)),replace=False)
    p=points[idx]; ls=labels[idx]
    if classes is None: colors=labels_to_rgb(ls)/255.
    else: colors=np.array([CLASS_COLORS[classes[int(v)]] for v in ls])/255.
    fig=plt.figure(figsize=(16,5))
    for i,j in enumerate((1,2)):
        ax=fig.add_subplot(1,3,i+1)
        ax.scatter(p[:,0],p[:,j],c=colors,s=2,alpha=.65,linewidths=0)
        ax.set(xlabel='x (m)',ylabel='y (m)' if j==1 else 'z (m)',aspect='equal',title='Superior' if j==1 else 'Lateral')
    ax=fig.add_subplot(1,3,3,projection='3d'); ax.scatter(*p.T,c=colors,s=1,alpha=.65)
    ax.set(xlabel='x',ylabel='y',zlabel='z',title='3D'); ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),.1))
    if classes:
        names=sorted(set(classes.values()))
        fig.legend(handles=[Patch(color=np.array(CLASS_COLORS[n])/255,label=n) for n in names],loc='lower center',ncol=4,fontsize=8)
    fig.suptitle(title); fig.tight_layout(rect=(0,.12,1,.94) if classes else (0,0,1,.94))
    path=FIG/f'exp_{name}.png'; fig.savefig(path,dpi=140); plt.close(fig); display(Image(filename=str(path)))
def save_run(name,run):
    np.savez_compressed(CACHE/f'exp_{name}.npz',labels=run['labels'],idx=run['idx'],sample_labels=run['sample_labels'])
    serial=[]
    for p in run['planes']:
        serial.append({k:v for k,v in p.items() if k!='inliers'})
    (CACHE/f'exp_{name}_planes.json').write_text(json.dumps(serial,indent=2))
    supports={f'plano_{j}':run['idx'][p['inliers']] for j,p in enumerate(run['planes'])}
    sampled=cloud['xyz'][run['idx']]
    for j,p in enumerate(run['planes']):
        for k,coef in enumerate(p.get('patches',[])):
            mask=np.abs(sampled@np.asarray(coef[:3])+coef[3])<.08
            supports[f'plano_{j}_parche_{k}']=run['idx'][mask]
    np.savez_compressed(CACHE/f'exp_{name}_inliers.npz',**supports)
def load_run(name):
    a=dict(np.load(CACHE/f'exp_{name}.npz'))
    a['planes']=json.loads((CACHE/f'exp_{name}_planes.json').read_text())
    return a
def finish(name, payload):
    payload['segundos_notebook']=time.perf_counter()-START
    (CACHE/f'exp_resumen_{name}.json').write_text(json.dumps(payload,indent=2,default=float))
    print(payload)

from pcseg.propagate import propagate_labels
from sklearn.preprocessing import StandardScaler
BASE_FULL=propagate_labels(xyz,base_labels,cloud['xyz'])
def maps_and_totals(run):
    desc,totals=summarize_classes(cloud['xyz'],run['labels'],run['planes'])
    classes=dict(zip(desc.id,desc.clase_heuristica)); classes[-1]='ruido'
    return desc,totals,classes
def mixture_metrics(desc,totals):
    def pct(name):
        row=totals[totals.clase_heuristica==name]
        return float(row.porcentaje.sum())
    columns=desc[desc.clase_heuristica=='columna']
    high=columns[columns.extension_vertical>=5]
    from sklearn.cluster import DBSCAN
    xy=(high[['x_min','y_min']].to_numpy()+high[['x_max','y_max']].to_numpy())/2
    footprints=DBSCAN(eps=1.5,min_samples=1).fit_predict(xy) if len(high) else []
    distinct=len(set(footprints))
    return dict(n_segmentos_completo=int((desc.id>=0).sum()),mezcla_pct=pct('mezcla_no_identificada'),ruido_completo_pct=pct('ruido'),
        no_resuelto_pct=pct('mezcla_no_identificada')+pct('ruido'),
        n_columnas=int(len(columns)),n_columnas_altas=int((columns.extension_vertical>=5).sum()),n_columnas_altas_distintas=distinct)
def geometry_space():
    path=CACHE/'exp_features_geo.npy'
    if path.exists(): return np.load(path)
    f=local_pca_features(xyz,k=96)
    X=StandardScaler().fit_transform(np.c_[xyz,normals,f['linealidad']])
    np.save(path,X)
    return X
'''

def execute_ipython(paths):
    """Alternativa sin sockets; conserva outputs y aborta en el primer error."""
    import contextlib
    import os
    import time
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    for path in paths:
        book=nb.read(path,as_version=4)
        book.cells=[c for c in book.cells if not (c.cell_type=='markdown' and c.source.startswith('**Ejecución:**'))]
        shell=InteractiveShell()
        start=time.perf_counter()
        log=ROOT/'outputs/logs'/f'exp_{path.stem}_ipython.log'
        with log.open('w') as stream:
            for index,cell in enumerate(book.cells):
                if cell.cell_type!='code': continue
                with capture_output() as cap:
                    result=shell.run_cell(cell.source,store_history=True)
                outputs=[]
                if cap.stdout: outputs.append(nb.v4.new_output('stream',name='stdout',text=cap.stdout))
                if cap.stderr: outputs.append(nb.v4.new_output('stream',name='stderr',text=cap.stderr))
                for item in cap.outputs:
                    outputs.append(nb.v4.new_output('display_data',data=item.data,metadata=item.metadata))
                cell.outputs=outputs; cell.execution_count=shell.execution_count-1
                stream.write(f'Celda {index}\n'+cap.stdout+cap.stderr+'\n'); stream.flush()
                nb.write(book,path)
                if result.error_before_exec or result.error_in_exec:
                    raise RuntimeError(f'{path}: celda {index}') from (result.error_in_exec or result.error_before_exec)
        note='**Ejecución:** IPython en `.venv`, celdas en orden, sin kernel de sockets (nbconvert bloqueado por sandbox). Tiempo: %.2f s. Logs en `outputs/logs/`; cachés en `outputs/cache/`.'%(time.perf_counter()-start)
        book.cells.insert(1,M(note)); nb.write(book,path)
        print(path.name,round(time.perf_counter()-start,2),'s',flush=True)

PLAN='''# 00 · Plan corregido: orientación antes de distancia
La revisión visual del primer intento encontró 57.43% como mezcla, solo dos columnas, y dos falsos planos de pared que eran acero del truss. DBSCAN agrupa estructuras conectadas: cambiar eps no basta para separarlas. Tampoco debemos decidir calidad por silhouette al comparar once grupos con cientos de segmentos.

| Hipótesis | Prueba | Criterio previo |
|---|---|---|
| H1: una pared debe ocupar superficie, no solo tener bbox grande | RANSAC; área ocupada y cobertura de grilla tangente | pared solo si área ocupada≥20 m² y cobertura≥55%; revisar candidatos rechazados y vista |
| H2: separar orientaciones rompe conexiones columna/viga | PCA local k=24/48/96; linealidad 0.30/0.45/0.60, planaridad 0.35/0.50/0.65 | medir distribución de orientaciones, instancias, mezcla y residuo; contrastar columnas en xy y lateral |
| H3: el crecimiento y la fusión reducen fragmentos | reasignación a <0.25/0.45/0.65 m; fusión colineal con hueco y distancia transversal acotados | evitar contar caras de una misma columna como objetos diferentes; curvas, cantidad de segmentos y posiciones |
| H4: orientación es el cambio decisivo | mismo pipeline con/sin orientación; DBSCAN y HDBSCAN, normales, color, height y voxel | mezcla<25%, ≥6 columnas; además ruido≤5% y mezcla+ruido<25%, para no esconder mezcla como ruido |
| H5: comparar en un espacio común y con control de K cambia la lectura | xyz+normales+linealidad estandarizados; segmentos≥500 puntos; soporte pareado; agregación de segmentos a K=11 | reportar silhouette/DB, cobertura, número de grupos y mayor segmento; no asumir que el filtro de tamaño elimina el sesgo por K |

Secuencia: 01 piso/techo y auditoría de pared; 02 sensibilidad de orientación y agrupación; 03 reglas honestas y color; 04 ablaciones, estabilidad y comparación; 05 propagación completa y revisión visual. La altura del piso sigue como unión de parches bajos a ≤0.40 m, con distancia <0.08 m. El eje e1 usa signo absoluto. Lineales verticales: |e1_z|>0.85; horizontales: |e1_z|<0.30, separados por dirección x/y. Planos se separan por normal vertical/horizontal; el resto es otro. PCA usa cKDTree solo sobre voxel.

Reglas físicas: columna si alto>2 m, alto/ancho≥3 y eje vertical>0.85; reportaremos además cuántas tienen alto≥5 m y sus posiciones para distinguir soportes parciales de columnas de la nave. Lineal horizontal de sección PCA aplanada (λ2/λ3>4) recibe viga_horizontal; sección ambigua o perfil inclinado recibe viga_o_tuberia. No llamaremos tubería a un perfil solo por baja planaridad/esfericidad: eso no prueba sección circular ni distingue tubo de perfil cerrado. No existe clase truss. Equipo bajo requiere extensión z<3, altura media sobre piso<3 y esfericidad≥0.03. Compacto plausible solo describe tamaño 0.4–2 m alto y 0.3–1 m ancho; no identifica persona/vehículo.

**Selección de la corrida final (incluye la enmienda de auditoría):** exigir piso fijo≥95%, mezcla<25%, ≥6 columnas, ruido≤5%, mezcla+ruido<25% y ≥6 columnas de alto≥5 m **en huellas xy separadas**. Para contar huellas, agrupamos centros xy a ≤1.5 m y contamos una sola por grupo; no altera etiquetas ni utiliza posiciones conocidas. Entre elegibles minimizar el número TOTAL de segmentos (penalizar fragmentación también por debajo de 500 puntos), después mezcla y después ARI de remuestreo descendente. Los controles sin orientación participan con el mismo resto de etapas. Si no hay elegibles, reportar la meta fallida y ordenar por mezcla+ruido, sin inventar cumplimiento. Las métricas de clustering se informan, no gobiernan esta selección. ARI/NMI entre clases finales colapsadas y los once grupos de clase solo son concordancia, no calidad ni exactitud.

La comparación geométrica usa un StandardScaler común en los 93 903 puntos de clase y linealidad calculada una vez con k=96. El soporte pareado excluye segmentos con <500 puntos en la nube completa de cualquiera de las dos particiones; se reporta cobertura y tamaño de ese soporte. Agregación a K=11: k-means sobre centroides geométricos de segmentos ponderados, conservando cada segmento entero. Es un control de resolución, no una segmentación física nueva. No se comparará silhouette sin informar K.

**Enmienda tras auditoría visual del primer ganador:** HDBSCAN100 tenía seis etiquetas altas pero solo cinco grupos de huellas, y faltaban columnas visibles alrededor de x≈3.5 y 13.5. La restricción de seis huellas distintas corrige ese conteo, sin cambiar métricas ni ajustar la segmentación a esas coordenadas. Se registra este cambio de selección, no se presenta como una regla fijada antes de la primera corrida.

Semilla fija 0. Estabilidad con tres réplicas voxel×{0.9,1,1.1} y retención aleatoria del 90%, evaluadas sobre la muestra común. Se repite RANSAC y PCA por réplica. Las tres comparaciones ARI no son intervalos de confianza. No hay ground truth.
'''
write('00_plan_experimentos',[M(PLAN)])

write('01_etapa1_estructura',[M('''# 01 · Piso y techo; revisar falsos planos de pared
Conservamos RANSAC numpy y la unión de parches bajos para el piso. El error anterior fue llamar pared a un bbox grande de puntos dispersos del truss. Ahora proyectamos los inliers a un sistema tangente y medimos celdas ocupadas de 0.30 m: exigimos área ocupada≥20 m² y cobertura≥55%. Un rectángulo grande vacío no cumple. No usamos etiquetas para tomar esa decisión.

La curva de distancia conserva el umbral 0.08 m. Dibujamos también el área y cobertura de los candidatos verticales, con los límites declarados. Rechazar una pared real escasamente observada sigue siendo posible.'''),C(COMMON),C('''idx=voxel_downsample(cloud['xyz'],.15)
p=cloud['xyz'][idx]; n=cloud['normals'][idx]; diagnostic=[]
planes=extract_planes(p,n,.08,diagnostics=diagnostic)
wall=pd.DataFrame(diagnostic);wall.to_csv(OUT/'exp_paredes_auditoria.csv',index=False);display(wall)
rows=[]
for threshold in (.03,.05,.08,.10,.15):
    labels=assign_planes(cloud['xyz'],planes,threshold)
    row={'umbral':threshold,'planos_pct':100*np.mean(labels>=0),**floor_metrics(cloud['xyz'],labels,planes)}
    for j in range(len(planes)):row[f'plano_{j}_pct']=100*np.mean(labels==j)
    rows.append(row)
sensitivity=pd.DataFrame(rows);sensitivity.to_csv(OUT/'exp_sensibilidad_planos.csv',index=False);display(sensitivity)
labels=assign_planes(cloud['xyz'],planes,.08)
plane_rows=[]
for j,pl in enumerate(planes):
    mask=labels==j
    plane_rows.append(dict(id=j,clase=pl['class'],porcentaje=100*mask.mean(),n_puntos=int(mask.sum()),
        residual_p95_m=float(np.quantile(plane_distance(cloud['xyz'][mask],pl),.95)),
        area_ocupada=pl.get('area_ocupada',np.nan),cobertura_grid=pl.get('cobertura_grid',np.nan)))
table=pd.DataFrame(plane_rows);table.to_csv(OUT/'exp_planos.csv',index=False);display(table)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for j,pl in enumerate(planes):axes[0].plot(sensitivity.umbral,sensitivity[f'plano_{j}_pct'],'o-',label=pl['class'])
axes[0].set(xlabel='Distancia (m)',ylabel='% nube');axes[0].legend()
if len(wall):
    axes[1].bar(np.arange(len(wall)),wall.area_ocupada);axes[1].axhline(20,c='r',ls='--')
    axes[2].bar(np.arange(len(wall)),wall.cobertura_grid);axes[2].axhline(.55,c='r',ls='--')
axes[1].set(title='Área ocupada candidatos verticales',ylabel='m²');axes[2].set(title='Cobertura de grilla',ylim=(0,1))
path=FIG/'exp_01_sensibilidad.png';fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig);display(Image(filename=str(path)))
run=dict(labels=labels,idx=idx,sample_labels=assign_planes(p,planes,.08),planes=planes)
save_run('estructura',run)
view(cloud['xyz'],labels,'01_estructura','Solo superficies aceptadas; gris = residuo')
finish('01',dict(n_planos=len(planes),n_paredes=int(sum(pl['class']=='pared' for pl in planes)),
    n_candidatos_pared=len(wall),**floor_metrics(cloud['xyz'],labels,planes)))''')])

write('02_etapa2_objetos',[M('''# 02 · Orientación local antes de agrupar
PCA local busca la dirección e1 de mayor variación; su signo no tiene significado. Con λ1≥λ2≥λ3 calculamos linealidad=(λ1−λ2)/λ1, planaridad=(λ2−λ3)/λ1 y esfericidad=λ3/λ1. Para planos usamos la normal e3, porque e1 puede rotar dentro de una superficie plana.

Probamos k y umbrales de forma en la misma muestra voxel. Luego DBSCAN/HDBSCAN se ajustan por orientación. Otro/ruido crece hacia un segmento solo si está cerca; el resto se agrupa para no perder equipos aislados. Dos fusiones reúnen fragmentos colineales del mismo eje, antes y después del crecimiento. La segunda permite unir una cara planar con un fuste lineal si su PCA de instancia confirma ese eje. Se limita el tamaño menor del par a 1200 representantes, la distancia transversal y el hueco longitudinal. No basta que dos objetos se toquen.

La selección preliminar busca una zona con mezcla voxel<25%, ≥6 columnas altas y menos segmentos: no depende de silhouette. Los porcentajes de esta sensibilidad son voxel, no porcentajes finales de la nube.'''),C(COMMON),C('''structure=load_run('estructura');idx=structure['idx'];planes=structure['planes']
p=cloud['xyz'][idx];plabel=assign_planes(p,planes,.08);residual=plabel<0;q=p[residual]
shape_rows=[];trial_rows=[];feature_cache={}
for k in (24,48,96):
    f=local_pca_features(q,k);feature_cache[k]=f
    for linear in (.30,.45,.60):
        for planar in (.35,.50,.65):
            group=orientation_classes(f,linear,planar)
            row=dict(k=k,linealidad=linear,planaridad=planar,radio_mediana=float(np.median(f['radio'])))
            for name in ORIENTATION_NAMES:row[name]=100*np.mean(group==name)
            shape_rows.append(row)
shapes=pd.DataFrame(shape_rows);shapes.to_csv(OUT/'exp_sensibilidad_orientacion.csv',index=False)
# Un factor por vez alrededor de la configuración de referencia.
trials=[dict(nombre=f'k{k}',k=k) for k in (24,48,96)]
trials += [dict(nombre=f'linear{v}',k=96,linear_threshold=v) for v in (.30,.60)]
trials += [dict(nombre=f'planar{v}',k=96,planar_threshold=v) for v in (.35,.65)]
trials += [dict(nombre=f'crecer{v}',k=96,reassign_distance=v) for v in (.25,.65)]
trials += [dict(nombre=f'fusion{lat}',k=96,merge_lateral=lat,merge_gap=gap) for lat,gap in ((.35,.4),(.55,.85),(.75,1.2))]
trials += [dict(nombre='HDBSCAN',k=96,method='hdbscan',min_cluster_size=40),
           dict(nombre='kmeans_por_orientacion',k=96,method='kmeans'),
           dict(nombre='xy_z_por_orientacion',k=96,method='two_level')]
for options in trials:
    name=options['nombre'];k=options['k'];kwargs={key:value for key,value in options.items() if key not in ('nombre','k')}
    params=dict(method='dbscan',eps=.35,min_samples=6,linear_threshold=.45,planar_threshold=.50,
        reassign_distance=.45,merge_gap=1.2,merge_lateral=.75)
    params.update(kwargs)
    lab,groups,detail=cluster_orientations(q,feature_cache[k],**params)
    labels=plabel.copy();labels[residual]=np.where(lab<0,-1,lab+len(planes))
    desc,totals=summarize_classes(p,labels,planes);met=mixture_metrics(desc,totals)
    trial_rows.append(dict(nombre=name,k=k,**met,n_segmentos=len(desc[desc.id>=0]),
        parametros=json.dumps(dict(k=k,**params))))
trials_table=pd.DataFrame(trial_rows);trials_table.to_csv(OUT/'exp_sensibilidad_agrupacion.csv',index=False);display(trials_table)
# La sensibilidad informa el ajuste; no se usa la mezcla sola para premiar descarte.
# Preferimos perfiles de radio y cantidad de fragmentos estables, con crecimiento acotado.
params=dict(voxel=.15,k=96,linear_threshold=.45,planar_threshold=.50,eps=.35,min_samples=6,
    reassign_distance=.45,merge_gap=1.2,merge_lateral=.75)
(CACHE/'exp_config_referencia.json').write_text(json.dumps(params,indent=2))
f=feature_cache[96];groups=orientation_classes(f,.45,.50)
group_ids=np.array([ORIENTATION_NAMES.index(name) for name in groups],dtype=np.int32)
view(q,group_ids,'02_orientaciones','Orientaciones locales (leyenda en tabla)')
display(pd.DataFrame({'id':np.arange(len(ORIENTATION_NAMES)),'orientacion':ORIENTATION_NAMES}))
distance=cKDTree(q).query(q,k=6,workers=2)[0][:,-1]
fig,axes=plt.subplots(1,3,figsize=(16,4))
for k in (24,48,96):
    sub=shapes[(shapes.k==k)&(shapes.planaridad==.5)]
    axes[0].plot(sub.linealidad,100-sub.otro,'o-',label=f'k={k}')
axes[0].set(xlabel='Umbral linealidad',ylabel='% con orientación');axes[0].legend()
axes[1].plot(np.sort(distance));axes[1].axhline(.35,c='r',ls='--');axes[1].set(ylim=(0,1),ylabel='k-distance (6)',xlabel='Puntos ordenados')
for k in (24,48,96):axes[2].hist(feature_cache[k]['radio'],bins=60,range=(0,2),histtype='step',label=f'k={k}')
axes[2].set(xlabel='Radio hasta vecino k (m)',ylabel='Puntos');axes[2].legend()
path=FIG/'exp_02_sensibilidad.png';fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig);display(Image(filename=str(path)))
run=pipeline(cloud,cloud['xyz'],**params);save_run('referencia',run)
desc,totals,classes=maps_and_totals(run);totals.to_csv(OUT/'exp_referencia_clases.csv',index=False);display(totals)
view(cloud['xyz'],run['labels'],'02_referencia','Primera partición por orientación',classes)
finish('02',dict(**mixture_metrics(desc,totals),configuracion=params,n_segmentos=int((desc.id>=0).sum())))''')])

write('03_etapa3_refinamiento',[M('''# 03 · Reglas honestas y prueba de color
Clasificamos cada instancia por su geometría completa tras propagar, sin convertir automáticamente una orientación en un objeto físico. Columna: alto>2 m, alto/ancho≥3 y eje vertical>0.85. Reportamos por separado fustes de alto≥5 m. Una cara o tubería vertical puede cumplirlo: la vista xy y la lateral son parte de la revisión.

Los lineales horizontales de sección PCA aplanada (λ2/λ3>4) se llaman viga_horizontal; los demás quedan como viga_o_tuberia. Los perfiles inclinados también conservan esa ambigüedad. Linealidad alta con baja planaridad/esfericidad no demuestra una sección circular: un perfil cuadrado puede dar atributos similares. No emitimos tuberia ni truss como clases confirmadas. Equipo bajo requiere volumen PCA no degenerado. Compacto plausible no significa persona/vehículo.

Probamos k-means y GMM en Lab solo dentro de mezclas/equipos grandes; aceptamos k-means si silhouette Lab≥0.35, hijos≥30 y dispersión espacial relativa≤0.9. La ablación de color usa exactamente el mismo criterio. No usamos el color para pintar como conocido lo que no entendemos.'''),C(COMMON),C('''params=json.loads((CACHE/'exp_config_referencia.json').read_text())
run=pipeline(cloud,cloud['xyz'],**params);desc,totals,classes=maps_and_totals(run)
# Descriptores locales de los representantes no-planos, alineados por máscara.
f=run['local_features'];idx=run['idx'];p=cloud['xyz'][idx];residual=run['residual_mask']
local_table=pd.DataFrame({key:f[key] for key in ('linealidad','planaridad','esfericidad','radio')})
local_table['orientacion']=run['orientations'];local_table['indice_original']=idx[residual]
for j,axis in enumerate('xyz'):local_table[f'e1_{axis}']=f['e1'][:,j]
local_table.to_csv(OUT/'exp_features_locales.csv',index=False)
desc.to_csv(OUT/'exp_features_instancias.csv',index=False);display(totals)
colored=pipeline(cloud,cloud['xyz'],**params,color=True);save_run('refinamiento_color',colored)
diagnostic=colored['color_diagnostic']
if diagnostic.empty:diagnostic=pd.DataFrame(columns=['id','metodo','silhouette_Lab','dispersion_ratio','min_hijo','aceptada'])
diagnostic.to_csv(OUT/'exp_color_diagnostico.csv',index=False);display(diagnostic)
color_desc,color_totals,color_classes=maps_and_totals(colored)
compare=pd.DataFrame([dict(configuracion='sin_color',**mixture_metrics(desc,totals)),dict(configuracion='con_color',**mixture_metrics(color_desc,color_totals))])
compare.to_csv(OUT/'exp_refinamiento.csv',index=False);display(compare)
view(cloud['xyz'],run['labels'],'03_antes','Reglas sin color',classes)
view(cloud['xyz'],colored['labels'],'03_despues','Mismas reglas con refinamiento Lab',color_classes)
finish('03',dict(**mixture_metrics(desc,totals),n_compactos=int((desc.clase_heuristica=='compacto_plausible').sum()),
    n_divisiones_color=int(diagnostic.aceptada.sum()) if len(diagnostic) else 0))''')])

write('04_ablaciones',[M('''# 04 · Ablación de orientación y comparación de resolución
Los controles emparejados de DBSCAN y HDBSCAN100 cambian **solo orientation=True/False**, conservando RANSAC corregido, voxel, eps, min_samples y crecimiento por distancia. Ambos brazos ejecutan las mismas dos fusiones, infiriendo los ejes por PCA de instancia con el mismo criterio. También comparten el crecimiento y clustering del residuo; solo cambia la partición inicial por orientación. Incluimos también un control sin fusión para distinguir ese efecto. No reutilizamos la antigua separación especial de fustes.

Reportamos silhouette_comun/DB_comun como diagnóstico histórico, con K explícito, sin usarlos para afirmar mejora. La evaluación geométrica usa xyz+normales+linealidad estandarizados una sola vez y segmentos con ≥500 puntos **de la nube completa**. Cada pipeline y clase se evalúan en el mismo soporte pareado de esos segmentos; reportamos la cobertura para no premiar descarte. El filtro de tamaño no corrige por sí solo el sesgo por K: añadimos una agregación a K=11 de los centroides de segmentos (k-means ponderado, seed=0), conservando cada segmento completo. No se interpreta esa agregación como instancias físicas.

ARI y NMI comparan la partición colapsada a clases heurísticas con los once grupos de clase: son concordancia, no calidad, y no esperamos que sean altos al corregir el piso. El ruido permanece como clase en esa referencia. Por separado, ARI_media/std mide estabilidad entre tres réplicas de voxel y remuestreo, no concordancia con clase.

La selección usa la enmienda visual documentada en 00: se cuentan huellas altas distintas (centros xy agrupados a 1.5 m), no caras duplicadas. Esto no modifica las etiquetas. La regla de selección de 00 prioriza las metas geométricas y menor fragmentación total; nunca silhouette entre K distintos. Usamos el número total de segmentos para seleccionar, no solo los ≥500: de otro modo se premiaría esconder fragmentos bajo el filtro de evaluación. Las métricas de piso usan el mismo soporte de etapa 1 para todas las configuraciones. Los segundos suman ajuste nominal y tres réplicas, excluyendo evaluación y figuras.'''),C(COMMON),C('''reference=json.loads((CACHE/'exp_config_referencia.json').read_text());structure=load_run('estructura');reference_planes=structure['planes']
XGEO=geometry_space()
changes=[('orientacion',{}),('sin_orientacion',{'orientation':False}),('sin_fusion',{'merge_gap':0.,'merge_lateral':0.}),
 ('sin_normales',{'normals':False}),('con_color',{'color':True}),('con_height',{'height':True}),
 ('voxel010',{'voxel':.10}),('voxel025',{'voxel':.25}),('eps025',{'eps':.25}),('eps045',{'eps':.45}),
 ('min12',{'min_samples':12}),('HDBSCAN40',{'method':'hdbscan','min_cluster_size':40}),
 ('HDBSCAN100',{'method':'hdbscan','min_cluster_size':100,'min_samples':12}),
 ('HDBSCAN100_sin_orientacion',{'method':'hdbscan','min_cluster_size':100,'min_samples':12,'orientation':False}),('k48',{'k':48}),('crecer025',{'reassign_distance':.25})]
rows=[];replica_rows=[]
basemetrics=evaluate(EVAL,xyz,base_labels,reference_planes)
basegeo=geometric_comparison(XGEO,BASE_FULL,common_idx,BASE_FULL)
rows.append(dict(configuracion='linea_base_clase',**basemetrics,**basegeo,ARI_media=float(base.ARI),ARI_std=float(base.ARI_std),
    protocolo_ARI='semillas_histórico',segundos=0.,mezcla_pct=np.nan,ruido_completo_pct=0.,no_resuelto_pct=np.nan,n_columnas=np.nan,n_columnas_altas=np.nan,n_columnas_altas_distintas=np.nan,parametros='k-means K=11 original'))
for name,change in changes:
    params=dict(reference);params.update(change)
    run=pipeline(cloud,cloud['xyz'],**params);save_run(name,run)
    desc,totals,classes=maps_and_totals(run)
    desc.to_csv(CACHE/f'exp_descriptores_{name}.csv',index=False)
    totals.to_csv(CACHE/f'exp_clases_{name}.csv',index=False)
    common_labels=run['labels'][common_idx]
    metrics=evaluate(EVAL,xyz,common_labels,run['planes'])
    floor=floor_metrics(xyz,common_labels,reference_planes)
    metrics['%piso_en_un_segmento']=floor['piso_en_un_segmento'];metrics['piso_n_segmentos']=floor['piso_n_segmentos']
    geo=geometric_comparison(XGEO,run['labels'],common_idx,BASE_FULL,classes)
    replicas=[];seconds=run['seconds']
    for scale in (.9,1.,1.1):
        options=dict(params);options['voxel']=params['voxel']*scale
        rr=pipeline(cloud,xyz,**options,subsample=True);seconds+=rr['seconds'];replicas.append(rr['labels'])
        np.save(CACHE/f'exp_estabilidad_{name}_{scale}.npy',rr['labels'])
        replica_rows.append(dict(configuracion=name,voxel=options['voxel'],n_ajuste=len(rr['idx']),segundos=rr['seconds'],
            **floor_metrics(xyz,rr['labels'],reference_planes)))
    ari,std=stability(replicas)
    nonplan=assign_planes(xyz,reference_planes,.08)<0
    ari_res,std_res=stability([labels[nonplan] for labels in replicas])
    row=dict(configuracion=name,**metrics,**geo,**mixture_metrics(desc,totals),ARI_media=ari,ARI_std=std,
        ARI_residuo=ari_res,ARI_residuo_std=std_res,protocolo_ARI='remuestreo_voxel',segundos=seconds,parametros=json.dumps(params,sort_keys=True))
    rows.append(row)
    pd.DataFrame(rows).to_csv(ENT/'exp_ablaciones.csv',index=False)
    print(name,'mezcla',round(row['mezcla_pct'],2),'columnas',row['n_columnas'],'altas',row['n_columnas_altas'],
        'K500',row['n_segmentos_500'],'ARI',round(ari,3),flush=True)
abl=pd.DataFrame(rows);abl.to_csv(ENT/'exp_ablaciones.csv',index=False)
pd.DataFrame(replica_rows).to_csv(OUT/'exp_estabilidad_replicas.csv',index=False)
valid=abl[abl.configuracion!='linea_base_clase'].copy()
mask=(valid['%piso_en_un_segmento']>=95)&(valid.mezcla_pct<25)&(valid.no_resuelto_pct<25)&(valid.ruido_completo_pct<=5)&(valid.n_columnas>=6)&(valid.n_columnas_altas_distintas>=6)
valid['cumple_metas']=mask
if mask.any():
    eligible=valid[mask].sort_values(['n_segmentos_completo','mezcla_pct','ARI_media','configuracion'],ascending=[True,True,False,True]);stage='metas_cumplidas'
else:
    eligible=valid.sort_values(['no_resuelto_pct','configuracion']);stage='meta_fallida'
best_exp=eligible.iloc[0];valid['seleccionada']=valid.configuracion==best_exp.configuracion
valid.to_csv(OUT/'exp_ranking.csv',index=False)
selection=dict(configuracion=best_exp.configuracion,parametros=json.loads(best_exp.parametros),etapa=stage,candidatos=int(mask.sum()))
(CACHE/'exp_mejor_config.json').write_text(json.dumps(selection,indent=2))
display(abl[['configuracion','n_segmentos','n_segmentos_500','mezcla_pct','ruido_completo_pct','n_columnas','n_columnas_altas','n_columnas_altas_distintas','silhouette_geo','DB_geo','silhouette_geo_K11','DB_geo_K11','ARI_media']])
print('Selección:',selection)
# Tabla de resolución comparable, con soporte y concordancia explícitos.
fcols=['configuracion','n_segmentos_500','cobertura_500_pct','n_soporte_pareado','mayor_segmento_pct','silhouette_geo','DB_geo','silhouette_base_pareado','DB_base_pareado','n_geo_K11','silhouette_geo_K11','DB_geo_K11','ARI_clases_base','NMI_clases_base']
abl[fcols].to_csv(ENT/'exp_comparacion_justa.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(17,7));y=np.arange(len(valid))
for ax,col,title in zip(axes,['mezcla_pct','n_columnas_altas_distintas','n_segmentos_completo'],['% mezcla','Huellas de columnas ≥5 m','Segmentos totales']):
    ax.barh(y,valid[col],color=['#287e83' if value!=best_exp.configuracion else '#d7782f' for value in valid.configuracion]);ax.set(yticks=y,yticklabels=valid.configuracion if ax is axes[0] else [],title=title)
axes[0].axvline(25,c='r',ls='--');axes[1].axvline(6,c='r',ls='--')
path=FIG/'exp_04_resumen.png';fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig);display(Image(filename=str(path)))
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,base_col,new_col,title in zip(axes,['silhouette_base_pareado','DB_base_pareado'],['silhouette_geo_K11','DB_geo_K11'],['Silhouette geométrico · K=11','DB geométrico · K=11']):
    ax.bar(['Clase','Final agregado'],[best_exp[base_col],best_exp[new_col]],color=['#777777','#287e83']);ax.set(title=title)
fig.suptitle('Mismo soporte pareado y resolución; no mide exactitud');fig.tight_layout()
path=FIG/'exp_04_comparacion_K11.png';fig.savefig(path,dpi=140);plt.close(fig);display(Image(filename=str(path)))
finish('04',selection)''')])

write('05_resultado_final',[M('''# 05 · Resultado final y contraste visual
Aplicamos el ganador de 04 a todos los puntos. Colores de clase en PLY y panel; IDs de instancia conservados en el CSV y en cache/exp_final_ids.npy. Verificamos igualdad exacta de xyz, orden y normales al volver a leer el PLY.

La vista de columnas asigna **un color único por instancia**, sin reciclar la paleta de veinte colores. Incluye xy y lateral para detectar dos filas de columnas superpuestas y caras duplicadas. La tabla de columnas muestra centro, bbox y altura; alto≥5 m distingue fustes largos de soportes parciales. Las posiciones citadas por la revisión visual son una referencia de inspección, no etiquetas usadas por el algoritmo. No afirmamos un censo real ni exactitud.

La tabla base/final incluye la evaluación geométrica con soporte pareado y el control K=11, tamaño de segmentos, cobertura y mayor segmento. ARI/NMI de clases colapsadas se guardan solo como concordancia.'''),C(COMMON),C('''choice=json.loads((CACHE/'exp_mejor_config.json').read_text());print(choice)
run=pipeline(cloud,cloud['xyz'],**choice['parametros']);save_run('final',run)
labels=run['labels'];np.save(CACHE/'exp_final_ids.npy',labels)
summary,totals,classes=maps_and_totals(run)
summary['bbox']=summary.apply(lambda row:json.dumps([[row.x_min,row.y_min,row.z_min],[row.x_max,row.y_max,row.z_max]]),axis=1)
summary.to_csv(ENT/'final_segmentos.csv',index=False);totals.to_csv(OUT/'exp_porcentajes_clases.csv',index=False)
legend=pd.DataFrame([dict(clase_heuristica=name,R=CLASS_COLORS[name][0],G=CLASS_COLORS[name][1],B=CLASS_COLORS[name][2]) for name in sorted(totals.clase_heuristica)])
legend.to_csv(ENT/'final_leyenda.csv',index=False)
colors=np.empty((len(labels),3),np.uint8)
for label in np.unique(labels):colors[labels==label]=CLASS_COLORS[classes[int(label)]]
write_ply(ENT/'final_segmentos.ply',cloud['xyz'],colors,cloud['normals'])
restored=read_ply(ENT/'final_segmentos.ply')
assert np.array_equal(restored['xyz'],cloud['xyz']) and np.array_equal(restored['normals'],cloud['normals'])
assert np.array_equal(restored['rgb'],colors) and colors.dtype==np.uint8 and summary.n_puntos.sum()==1647021
view(cloud['xyz'],labels,'05_final','Orientación → instancias → clases heurísticas',classes)
# Panel por clase con escala y colores comunes.
names=totals.clase_heuristica.tolist();fig,axes=plt.subplots(int(np.ceil(len(names)/3)),3,figsize=(16,4*int(np.ceil(len(names)/3))),squeeze=False)
for ax,name in zip(axes.flat,names):
    ids=summary.loc[summary.clase_heuristica==name,'id'].to_numpy();points=np.flatnonzero(np.isin(labels,ids))
    take=np.random.default_rng(0).choice(points,min(18000,len(points)),replace=False);p=cloud['xyz'][take]
    pct=float(totals.loc[totals.clase_heuristica==name,'porcentaje'].iloc[0])
    ax.scatter(p[:,0],p[:,2],s=2,c=[np.array(CLASS_COLORS[name])/255],alpha=.65)
    ax.set(xlim=(cloud['xyz'][:,0].min(),cloud['xyz'][:,0].max()),ylim=(cloud['xyz'][:,2].min(),cloud['xyz'][:,2].max()),xlabel='x (m)',ylabel='z (m)',title=f'{name}: {pct:.2f}% · {len(ids)} segmentos')
    ax.legend(handles=[Patch(color=np.array(CLASS_COLORS[name])/255,label=name)],fontsize=7)
for ax in list(axes.flat)[len(names):]:ax.axis('off')
path=FIG/'exp_05_panel_clases.png';fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig);display(Image(filename=str(path)))
columns=summary[summary.clase_heuristica=='columna'].copy()
for axis in 'xyz':columns[f'{axis}_centro']=(columns[f'{axis}_min']+columns[f'{axis}_max'])/2
columns['fuste_alto']=columns.extension_vertical>=5
columns['grupo_huella_alta']=-1
from sklearn.cluster import DBSCAN
high=columns[columns.fuste_alto]
if len(high):
    footprint=DBSCAN(eps=1.5,min_samples=1).fit_predict(high[['x_centro','y_centro']])
    columns.loc[high.index,'grupo_huella_alta']=footprint
columns.to_csv(OUT/'exp_columnas_auditoria.csv',index=False)
# Paleta hsv con un tono por ID, guardada para consultar la figura sin ambigüedad.
colpalette=(plt.get_cmap('hsv')(np.linspace(0,.95,max(len(columns),1)))[:,:3]*255).astype(np.uint8)
fig,axes=plt.subplots(1,2,figsize=(17,8));handles=[];colorrows=[]
for i,row in enumerate(columns.itertuples()):
    pointids=np.flatnonzero(labels==row.id);take=np.random.default_rng(0).choice(pointids,min(5000,len(pointids)),replace=False);p=cloud['xyz'][take];c=colpalette[i]/255
    for ax,axis in zip(axes,(1,2)):ax.scatter(p[:,0],p[:,axis],s=3,c=[c],alpha=.7)
    axes[0].text(row.x_centro,row.y_centro,str(row.id),fontsize=6)
    axes[1].text(row.x_centro,row.z_max+.1,str(row.id),fontsize=6)
    handles.append(Patch(color=c,label=f'{row.id} ({row.extension_vertical:.1f} m)'))
    colorrows.append(dict(id=row.id,R=int(colpalette[i,0]),G=int(colpalette[i,1]),B=int(colpalette[i,2])))
axes[0].set(xlabel='x (m)',ylabel='y (m)',title='Instancias de columnas · xy',aspect='equal');axes[1].set(xlabel='x (m)',ylabel='z (m)',title='Instancias de columnas · lateral',aspect='equal')
fig.legend(handles=handles,loc='lower center',ncol=8,fontsize=6)
fig.suptitle('Un color por instancia; candidatas geométricas, no censo de columnas')
fig.tight_layout(rect=(0,.18,1,.94));path=FIG/'exp_05_columnas_instancias.png';fig.savefig(path,dpi=160);plt.close(fig);display(Image(filename=str(path)))
pd.DataFrame(colorrows,columns=['id','R','G','B']).to_csv(OUT/'exp_columnas_colores.csv',index=False)
XGEO=geometry_space();fair=geometric_comparison(XGEO,labels,common_idx,BASE_FULL,classes)
abl=pd.read_csv(ENT/'exp_ablaciones.csv');selected=abl[abl.configuracion==choice['configuracion']].iloc[0]
base_row=abl.iloc[0].to_dict()
# En la tabla final ambas filas usan exactamente el soporte pareado del final.
base_row.update(silhouette_geo=fair['silhouette_base_pareado'],DB_geo=fair['DB_base_pareado'],
    silhouette_geo_K11=fair['silhouette_base_pareado'],DB_geo_K11=fair['DB_base_pareado'],
    n_soporte_pareado=fair['n_soporte_pareado'])
comparison=pd.DataFrame([base_row,dict(configuracion='final_'+choice['configuracion'],
    **evaluate(EVAL,xyz,labels[common_idx],run['planes']),**fair,**mixture_metrics(summary,totals),ARI_media=selected.ARI_media,ARI_std=selected.ARI_std,protocolo_ARI='remuestreo_voxel')])
comparison.to_csv(OUT/'exp_comparacion_final.csv',index=False)
display(comparison[['configuracion','n_segmentos','n_segmentos_500','mayor_segmento_pct','silhouette_geo','DB_geo','silhouette_geo_K11','DB_geo_K11','ARI_clases_base','NMI_clases_base']])
display(totals);display(columns[['id','x_centro','y_centro','extension_vertical','n_puntos','fuste_alto']])
met=mixture_metrics(summary,totals)
finish('05',dict(configuracion=choice['configuracion'],n_puntos=len(labels),n_segmentos=int((summary.id>=0).sum()),
    clases=totals.clase_heuristica.tolist(),**met,n_compactos=int((summary.clase_heuristica=='compacto_plausible').sum()),
    segundos_pipeline=run['seconds'],**floor_metrics(cloud['xyz'],labels,run['planes']),**fair))''')])

# Las conclusiones se rellenan únicamente después de revisar resultados reales.
# Un rerun del generador conserva las últimas conclusiones verificadas, si existen.
learned_path=ROOT/'outputs/cache/exp_conclusiones_vigentes.json'
if '--execute' not in sys.argv and learned_path.exists():
    import json
    for name,text in json.loads(learned_path.read_text()).items():
        path=DEST/f'{name}.ipynb';book=nb.read(path,as_version=4);book.cells.append(M(text));nb.write(book,path)

if __name__=='__main__':
    if '--execute' in sys.argv:
        execute_ipython([DEST/f'{name}.ipynb' for name in sys.argv[sys.argv.index('--execute')+1:]])
    else: print('Plan corregido y notebooks preparados; conclusiones pendientes de cifras reales.')
