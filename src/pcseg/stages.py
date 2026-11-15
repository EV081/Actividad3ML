"""Pipeline exploratorio por etapas. Reglas geométricas, nunca ground truth."""
from itertools import combinations
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from .features import voxel_downsample, build_features
from .metrics import silhouette_sample, davies_bouldin, cluster_summary


def ransac_plane(xyz, normals=None, kind='horizontal', threshold=.08,
                 iterations=180, random_state=0):
    """RANSAC numpy con tripletas y candidatos filtrados por normales.

    El scoring usa hasta 12000 candidatos; el refinamiento SVD usa todos.
    Devuelve plano unitario (n,d) y máscara de soporte con normales compatibles.
    """
    xyz = np.asarray(xyz, dtype=float)
    if threshold <= 0 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz (N,3), threshold positivo')
    horizontal = kind == 'horizontal'
    nz = .9 if horizontal else .2
    keep = np.ones(len(xyz), dtype=bool) if normals is None else (
        np.abs(normals[:, 2]) > nz if horizontal else np.abs(normals[:, 2]) < nz)
    candidates = np.flatnonzero(keep)
    if len(candidates) < 3:
        return None, np.zeros(len(xyz), bool)
    rng = np.random.default_rng(random_state)
    score_idx = rng.choice(candidates, min(12000, len(candidates)), replace=False)
    q = xyz[score_idx]
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = xyz[rng.choice(candidates, 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if (horizontal and abs(normal[2]) < .97) or (not horizontal and abs(normal[2]) > .12):
            continue
        d = -a @ normal
        count = np.count_nonzero(np.abs(q @ normal+d) < threshold)
        if count > best_count:
            best, best_count = (normal, d), count
    if best is None:
        return None, np.zeros(len(xyz), bool)
    normal, d = best
    for _ in range(2):
        mask = keep & (np.abs(xyz @ normal+d) < threshold)
        if mask.sum() < 3:
            break
        center = xyz[mask].mean(axis=0)
        _, _, vt = np.linalg.svd(xyz[mask]-center, full_matrices=False)
        normal = vt[-1]
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        d = -center @ normal
    return np.r_[normal, d], keep & (np.abs(xyz @ normal+d) < threshold)


def extract_planes(xyz, normals=None, threshold=.08, wall_min_area=20.,
                   wall_min_coverage=.55, diagnostics=None):
    """Hasta 4 horizontales y 4 verticales significativos, con soporte amplio."""
    planes = []
    remaining = np.ones(len(xyz), bool)
    for kind in ('horizontal', 'vertical'):
        for _ in range(4):
            idx = np.flatnonzero(remaining)
            plane, support = ransac_plane(xyz[idx], None if normals is None else normals[idx],
                                          kind, threshold)
            if plane is None or support.sum() < max(180, .004*len(xyz)):
                break
            ids = idx[support]
            ext = np.ptp(xyz[ids], axis=0)
            remaining[ids] = False
            # Dos direcciones extensas: no retirar una cara estrecha de columna.
            if np.sort(ext)[-2] < 3.0:
                continue
            area=plane_support_area(xyz[ids],plane)
            if kind=='vertical':
                accepted=(area['area_ocupada']>=wall_min_area and area['cobertura_grid']>=wall_min_coverage)
                if diagnostics is not None:
                    diagnostics.append(dict(**area,aceptada=accepted,n_inliers=len(ids)))
                if not accepted: continue
            planes.append(dict(coef=plane.tolist(), kind=kind, **area,
                               z=float(np.median(xyz[ids, 2])), inliers=ids))
    horizontal = sorted([p for p in planes if p['kind']=='horizontal'], key=lambda p:p['z'])
    for p in planes:
        p['class'] = ('piso' if p is horizontal[0] else 'techo' if p is horizontal[-1]
                      else 'plano_horizontal') if p['kind']=='horizontal' else 'pared'
    # La primera prueba dividió el piso: agrupamos sus parches bajos próximos.
    # Se mantienen ecuaciones e inliers; es un segmento con varias superficies.
    if horizontal:
        low = horizontal[0]
        patches = [p for p in horizontal if p['z']-low['z'] <= .40]
        low['patches'] = [p['coef'] for p in patches]
        low['inliers'] = np.unique(np.concatenate([p['inliers'] for p in patches]))
        planes = [p for p in planes if p is low or all(p is not q for q in patches)]
    # Piso primero: tiene prioridad para unir el soporte geométrico bajo.
    planes.sort(key=lambda p:(p['class']!='piso', p['z']))
    return planes


def plane_distance(xyz, plane):
    """Distancia mínima a los parches de un mismo segmento plano."""
    return np.min([np.abs(xyz@np.asarray(c[:3])+c[3])
                   for c in plane.get('patches',[plane['coef']])],axis=0)


def assign_planes(xyz, planes, threshold=.08):
    """Asignación por menor distancia al plano; no usa nearest neighbor."""
    labels = np.full(len(xyz), -1, np.int32)
    best = np.full(len(xyz), threshold, dtype=float)
    for label, p in enumerate(planes):
        distance = plane_distance(xyz,p)
        keep = (distance < best) & (labels != 0)
        labels[keep] = label
        best[keep] = distance[keep]
    return labels


def local_pca_features(xyz, k=48):
    """PCA local con cKDTree: índices propios descendentes, e1 y normal e3.

    El signo de cada eigenvector es arbitrario; las reglas usan valores absolutos.
    Se procesan bloques para limitar memoria. Vecindarios degenerados no reciben
    linealidad/planaridad artificial: sus tres descriptores quedan en cero.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('xyz debe ser (N,3), finito')
    if not isinstance(k, (int, np.integer)) or k < 3:
        raise ValueError('k debe ser entero >=3')
    n = len(xyz)
    result = {key: np.zeros(n) for key in ('linealidad','planaridad','esfericidad','radio')}
    result.update(e1=np.zeros((n,3)), normal=np.zeros((n,3)), eigenvalues=np.zeros((n,3)))
    if n < 3:
        return result
    tree = cKDTree(xyz)
    for start in range(0,n,4000):
        dist, neighbors = tree.query(xyz[start:start+4000], k=min(k,n),workers=2)
        q=xyz[neighbors]
        q=q-q.mean(axis=1,keepdims=True)
        eig, vec = np.linalg.eigh(np.einsum('nki,nkj->nij',q,q)/max(min(k,n)-1,1))
        eig=np.maximum(eig[:,::-1],0)
        scale=np.maximum(eig[:,0],1e-12)
        valid=eig[:,0]>1e-12
        end=start+len(q)
        result['linealidad'][start:end]=np.where(valid,(eig[:,0]-eig[:,1])/scale,0)
        result['planaridad'][start:end]=np.where(valid,(eig[:,1]-eig[:,2])/scale,0)
        result['esfericidad'][start:end]=np.where(valid,eig[:,2]/scale,0)
        result['e1'][start:end]=vec[:,:,-1]
        result['normal'][start:end]=vec[:,:,0]
        result['eigenvalues'][start:end]=eig
        result['radio'][start:end]=dist[:,-1]
    return result


def local_pca(xyz, k=20):
    """Compatibilidad: tres descriptores; usar local_pca_features para e1."""
    f=local_pca_features(xyz,k)
    return np.column_stack([f[key] for key in ('linealidad','planaridad','esfericidad')])


ORIENTATION_NAMES = ('lineal_vertical','lineal_horizontal_x','lineal_horizontal_y',
                     'planar_normal_vertical','planar_normal_horizontal','otro')


def orientation_classes(features, linear_threshold=.45, planar_threshold=.5,
                        vertical_threshold=.85, horizontal_threshold=.3):
    """Partición de puntos; primero lineales, luego planares, resto otro.

    No interpreta el nombre físico del objeto a partir de la orientación.
    Una superficie vertical tiene normal horizontal. Un plano isotrópico no
    tiene e1 estable; por eso se usa su normal e3 para clasificarlo.
    """
    if not 0<=linear_threshold<=1 or not 0<=planar_threshold<=1:
        raise ValueError('umbrales de forma deben estar entre 0 y 1')
    if not 0<=horizontal_threshold<vertical_threshold<=1:
        raise ValueError('umbrales de dirección incompatibles')
    linear=np.asarray(features['linealidad']); planar=np.asarray(features['planaridad'])
    e1=np.asarray(features['e1']); normal=np.asarray(features['normal'])
    if e1.shape!=(len(linear),3) or normal.shape!=e1.shape or planar.shape!=linear.shape:
        raise ValueError('features incompatibles')
    out=np.full(len(linear),'otro',dtype='<U27')
    line=linear>=linear_threshold
    out[line & (np.abs(e1[:,2])>vertical_threshold)]='lineal_vertical'
    horiz=line & (np.abs(e1[:,2])<horizontal_threshold)
    out[horiz & (np.abs(e1[:,0])>=np.abs(e1[:,1]))]='lineal_horizontal_x'
    out[horiz & (np.abs(e1[:,0])<np.abs(e1[:,1]))]='lineal_horizontal_y'
    flat=(out=='otro') & (planar>=planar_threshold)
    out[flat & (np.abs(normal[:,2])>vertical_threshold)]='planar_normal_vertical'
    out[flat & (np.abs(normal[:,2])<horizontal_threshold)]='planar_normal_horizontal'
    return out


def plane_support_area(xyz, coef, cell=.30):
    """Área ocupada y cobertura de grilla en coordenadas tangentes al plano.

    El bbox vacío de un truss no es área de pared observada.
    """
    n=np.asarray(coef[:3],float); n/=np.linalg.norm(n)
    basis=np.eye(3)[np.argmin(np.abs(n))]
    u=np.cross(n,basis); u/=np.linalg.norm(u); v=np.cross(n,u)
    uv=np.asarray(xyz)@np.column_stack([u,v])
    ext=np.ptp(uv,axis=0)
    area=float(np.prod(ext))
    grid=np.floor((uv-uv.min(axis=0))/cell).astype(np.int32)
    occupied=len(np.unique(grid,axis=0))*cell**2
    slots=np.prod(np.floor(ext/cell)+1)
    return dict(area_bbox=area,area_ocupada=float(occupied),
                cobertura_grid=float(occupied/(slots*cell**2)))


def describe_instances(xyz, labels, planes, local=None):
    """Features PCA globales y locales, bbox y clase heurística declarada."""
    floor = next((p['z'] for p in planes if p['class']=='piso'), float(xyz[:,2].min()))
    rows = []
    labels=np.asarray(labels)
    order=np.argsort(labels,kind='stable')
    unique,starts=np.unique(labels[order],return_index=True)
    ends=np.r_[starts[1:],len(labels)]
    for label,start,end in zip(unique,starts,ends):
        members=order[start:end]
        q = xyz[members].astype(float)
        lo, hi = q.min(axis=0), q.max(axis=0)
        ext = hi-lo
        eigen, vectors = np.linalg.eigh(np.cov(q.T) if len(q)>2 else np.zeros((3,3)))
        a,b,c = np.maximum(eigen[::-1], 0)
        linear, planar, sphere = (a-b)/max(a,1e-12), (b-c)/max(a,1e-12), c/max(a,1e-12)
        vertical = abs(vectors[2,-1])
        width = max(ext[:2])
        height = q[:,2].mean()-floor
        if label < 0:
            cls = 'ruido'
        elif label < len(planes):
            cls = planes[label]['class']
        elif ext[2]>2 and ext[2]/max(width,.01)>=3 and vertical>.85:
            cls = 'columna'
        elif linear>=.60 and vertical<.30:
            cls = 'viga_horizontal' if b/max(c,1e-12)>4 else 'viga_o_tuberia'
        elif linear>=.60 and vertical<.80:
            cls = 'viga_o_tuberia'
        elif .4<=ext[2]<=2 and .3<=width<=1:
            cls = 'compacto_plausible'
        elif height<3 and ext[2]<3 and linear<.65 and sphere>=.03:
            cls = 'equipo_bajo'
        else:
            cls = 'mezcla_no_identificada'
        row = dict(id=int(label), clase_heuristica=cls, n_puntos=len(q),
                   linealidad=linear, planaridad=planar, esfericidad=sphere,
                   extension_vertical=ext[2], altura_media=height,
                   aspecto_huella=width/max(min(ext[:2]),.01), eje_vertical=vertical)
        for j, axis in enumerate('xyz'):
            row[f'{axis}_min'], row[f'{axis}_max'] = lo[j], hi[j]
        if local is not None:
            for j, name in enumerate(('linealidad_local','planaridad_local','esfericidad_local')):
                row[name] = float(np.median(local[members,j]))
        rows.append(row)
    return pd.DataFrame(rows)


def object_labels(xyz, normals=None, method='dbscan', eps=.30, min_samples=6,
                  min_cluster_size=50, height=False):
    X = xyz.astype(float).copy()
    # height opcional repite z con peso 0.5, conservando xyz en ambas variantes.
    if height and len(xyz):
        X = np.c_[X, .5*(xyz[:,2]-np.percentile(xyz[:,2],1))]
    if normals is not None:
        # Peso pequeño en metros; normales de signo opuesto pueden separar superficies.
        X = np.c_[X, .12*normals]
    if len(X)<min_samples:
        return np.full(len(X),-1,np.int32)
    if method == 'hdbscan':
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
            min_samples=min_samples, core_dist_n_jobs=2).fit_predict(X)
    if method == 'kmeans':
        return KMeans(n_clusters=min(11,len(X)), n_init=3, random_state=0).fit_predict(X)
    if method == 'two_level':
        footprints = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=2).fit_predict(X[:,:2])
        labels = np.full(len(X),-1,np.int32)
        next_id = 0
        for label in np.unique(footprints[footprints>=0]):
            ids = np.flatnonzero(footprints==label)
            zlabels = density_1d(xyz[ids,2],eps,min_samples)
            good = zlabels>=0
            labels[ids[good]] = zlabels[good]+next_id
            if good.any(): next_id += int(zlabels[good].max())+1
        return labels
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=2).fit_predict(X)


def refine_color(xyz, rgb, labels, planes, enabled=True):
    """Divide solo mezclas grandes con controles espaciales y en Lab."""
    result = labels.copy()
    diagnostics = []
    if not enabled:
        return result, pd.DataFrame()
    desc = describe_instances(xyz, labels, planes)
    next_id = int(max(labels.max()+1,len(planes)))
    for row in desc.itertuples():
        if row.id < len(planes) or row.n_puntos<150 or row.clase_heuristica not in ('mezcla_no_identificada','equipo_bajo'):
            continue
        idx = np.flatnonzero(labels==row.id)
        if max(row.x_max-row.x_min,row.y_max-row.y_min)<2:
            continue
        lab = build_features(xyz[idx],rgb[idx],None,use_xyz=False,use_rgb=True,standardize=False)
        parent_disp = np.sum(np.var(xyz[idx],axis=0))
        for method in ('kmeans','gmm'):
            if method=='kmeans':
                split = KMeans(n_clusters=2,n_init=3,random_state=0).fit_predict(lab)
            else:
                split = GaussianMixture(n_components=2,random_state=0,reg_covar=1e-4).fit_predict(lab)
            sil = silhouette_sample(lab,split,n=800,seed=0)
            sizes = np.bincount(split,minlength=2)
            disp = sum(np.sum(np.var(xyz[idx[split==j]],axis=0))*sizes[j]/len(idx) for j in range(2) if sizes[j])
            ratio = disp/max(parent_disp,1e-12)
            accepted = bool(method=='kmeans' and sil>=.35 and sizes.min()>=30 and ratio<=.9)
            diagnostics.append(dict(id=row.id,metodo=method,silhouette_Lab=sil,
                                    dispersion_ratio=ratio,min_hijo=int(sizes.min()),aceptada=accepted))
            if accepted:
                result[idx[split==1]] = next_id
                next_id += 1
    return result,pd.DataFrame(diagnostics)


def pipeline(cloud, target_xyz, voxel=.15, eps=.35, min_samples=6,
             min_cluster_size=40, method='dbscan', normals=True, color=False,
             height=False, threshold=.08, subsample=False, split_columns=False,
             orientation=True, k=48, linear_threshold=.45, planar_threshold=.50,
             reassign_distance=.45, merge_gap=.85, merge_lateral=.35):
    """Planos verificados → PCA local → orientación → densidad → reglas.

    Las normales de entrada filtran RANSAC. En objetos, e1 procede de PCA;
    añadir normales al clustering es una ablación opcional con peso 0.12 m.
    """
    start=time.perf_counter()
    idx=voxel_downsample(cloud['xyz'],voxel)
    if subsample:
        rng=np.random.default_rng(0)
        idx=idx[rng.choice(len(idx),int(.90*len(idx)),replace=False)]
    xyz,rgb,nrm=(cloud[key][idx] for key in ('xyz','rgb','normals'))
    rejected=[]
    planes=extract_planes(xyz,nrm if normals else None,threshold,diagnostics=rejected)
    labels=assign_planes(xyz,planes,threshold); residual=labels<0
    f=local_pca_features(xyz[residual],k)
    lab,groups,detail=cluster_orientations(xyz[residual],f,method,eps,min_samples,
        min_cluster_size,linear_threshold,planar_threshold,reassign_distance,
        merge_gap,merge_lateral,orientation,None,height)
    labels[residual]=np.where(lab<0,-1,lab+len(planes))
    if split_columns: labels=split_verticals(xyz,labels,planes)
    labels,diagnostic=refine_color(xyz,rgb,labels,planes,color)
    target=assign_planes(target_xyz,planes,threshold); missing=target<0
    if residual.any() and missing.any():
        _,near=cKDTree(xyz[residual]).query(target_xyz[missing],workers=2)
        target[missing]=labels[residual][near]
    return dict(labels=target,idx=idx,sample_labels=labels,planes=planes,
        residual_mask=residual,local_features=f,orientations=groups,
        color_diagnostic=diagnostic,wall_diagnostic=pd.DataFrame(rejected),
        orientation_diagnostic=detail,seconds=time.perf_counter()-start)


def floor_metrics(xyz, labels, planes):
    plane = next((p for p in planes if p['class']=='piso'),None)
    if plane is None:
        return dict(piso_en_un_segmento=0.,piso_n_segmentos=0,piso_soporte=0)
    mask = plane_distance(xyz,plane)<.08
    valid = labels[mask & (labels>=0)]
    counts = np.unique(valid,return_counts=True)[1]
    return dict(piso_en_un_segmento=100*counts.max()/max(mask.sum(),1) if len(counts) else 0.,
                piso_n_segmentos=len(counts),piso_soporte=int(mask.sum()))


def evaluate(X, xyz, labels, planes):
    summary = cluster_summary(labels)
    obj = labels>=len(planes)
    counts = np.unique(labels[obj],return_counts=True)[1]
    floor = floor_metrics(xyz,labels,planes)
    return dict(n_segmentos=summary['n_clusters'],ruido=summary['noise_frac'],
                silhouette_comun=silhouette_sample(X,labels,n=1200,seed=0),
                DB_comun=davies_bouldin(X,labels),
                silhouette_soporte_fijo=silhouette_sample(X,labels+1,n=1200,seed=0),
                mayor_objeto=counts.max()/max(obj.sum(),1) if len(counts) else 0.,
                **{'%piso_en_un_segmento':floor['piso_en_un_segmento']},
                piso_n_segmentos=floor['piso_n_segmentos'])


def stability(runs):
    scores = [adjusted_rand_score(a,b) for a,b in combinations(runs,2)]
    return float(np.mean(scores)),float(np.std(scores))


def split_verticals(xyz, labels, planes):
    """Busca fustes verticales dentro del residuo conectado y conserva IDs únicos.

    PCA local filtra ejes; DBSCAN en xy reúne el mismo fuste a alturas distintas.
    Exige alto >=2, ancho <=0.8 y altura/ancho >=3 antes de crecer 0.30 m en xy.
    Es una regla exploratoria; podría confundir soportes de equipos con columnas.
    """
    out = labels.copy()
    residual = labels>=len(planes)
    ids = np.flatnonzero(residual)
    q = xyz[ids].astype(float)
    if len(q)<20: return out
    _, near = cKDTree(q).query(q,k=20,workers=2)
    candidates = np.zeros(len(q),bool)
    for start in range(0,len(q),5000):
        v=q[near[start:start+5000]]
        v-=v.mean(axis=1,keepdims=True)
        eig,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',v,v))
        candidates[start:start+5000]=((eig[:,2]-eig[:,1])/np.maximum(eig[:,2],1e-12)>.45)&(np.abs(vec[:,2,2])>.90)
    floor=next((p['z'] for p in planes if p['class']=='piso'),float(xyz[:,2].min()))
    candidates &= (q[:,2]>floor+.4)
    if candidates.sum()<12: return out
    xy_labels=DBSCAN(eps=.30,min_samples=6,n_jobs=2).fit_predict(q[candidates,:2])
    next_id=int(max(out.max()+1,len(planes)))
    for label in np.unique(xy_labels[xy_labels>=0]):
        seed=q[candidates][xy_labels==label]
        ext=np.ptp(seed,axis=0); width=max(ext[:2])
        if ext[2]<2 or width>.8 or ext[2]/max(width,.01)<3: continue
        center=np.median(seed[:,:2],axis=0)
        take=(np.linalg.norm(q[:,:2]-center,axis=1)<max(.30,width/2)) & (q[:,2]>=seed[:,2].min()-.15)&(q[:,2]<=seed[:,2].max()+.15)
        target=ids[take]
        if len(target)<20: continue
        out[target]=next_id; next_id+=1
    return out


def merge_collinear_fragments(xyz, labels, axes, gap=.85, lateral=.35, small_size=600):
    """Fusiona solo fragmentos del mismo eje, con hueco y desvío limitados.

    No conecta ejes distintos aunque se toquen. Medianas transversales evitan
    fusionar dos columnas vecinas por un solo punto de borde.
    """
    labels=labels.copy(); ids=np.unique(labels[labels>=0])
    info={}
    for label in ids:
        axis=axes.get(int(label))
        if axis not in (0,1,2): continue
        q=xyz[labels==label]
        other=[j for j in range(3) if j!=axis]
        info[int(label)]=(axis,np.median(q[:,other],axis=0),float(q[:,axis].min()),
                          float(q[:,axis].max()),len(q))
    parent={int(label):int(label) for label in ids}
    def root(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    keys=list(info)
    for i,a in enumerate(keys):
        axis,center,lo,hi,n=info[a]
        for b in keys[i+1:]:
            ax2,c2,l2,h2,n2=info[b]
            if axis!=ax2 or min(n,n2)>small_size: continue
            if np.linalg.norm(center-c2)>lateral: continue
            distance=max(lo-h2,l2-hi,0)
            if distance>gap: continue
            ra,rb=root(a),root(b)
            if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    mapping={label:root(label) for label in parent}
    for label,new in mapping.items(): labels[labels==label]=new
    return labels


def cluster_orientations(xyz, features=None, method='dbscan', eps=.35, min_samples=6,
                         min_cluster_size=40, linear_threshold=.45, planar_threshold=.50,
                         reassign_distance=.45, merge_gap=.85, merge_lateral=.35,
                         enabled=True, normals=None, height=False):
    """Clustering por orientación; crecimiento limitado al vecino y fusión de ejes.

    Otro no se transforma en una clase física: se reasigna solo si hay soporte
    espacial cercano. Su residuo se agrupa por densidad para describir equipos
    y mezcla. Los puntos de ruido que queden no se ocultan en una clase.
    """
    if features is None: features=local_pca_features(xyz)
    orientation=orientation_classes(features,linear_threshold,planar_threshold)
    labels=np.full(len(xyz),-1,np.int32); axes={}; next_id=0
    groups=ORIENTATION_NAMES[:-1] if enabled else ('sin_orientacion',)
    for group in groups:
        take=orientation==group if enabled else np.ones(len(xyz),bool)
        if not take.any(): continue
        lab=object_labels(xyz[take],None if normals is None else normals[take],
                          method,eps,min_samples,min_cluster_size,height)
        good=lab>=0
        ids=np.flatnonzero(take)
        labels[ids[good]]=lab[good]+next_id
        axis={'lineal_vertical':2,'lineal_horizontal_x':0,'lineal_horizontal_y':1}.get(group)
        if good.any():
            for label in np.unique(lab[good]): axes[int(label)+next_id]=axis
            next_id=int(labels.max())+1
    labels=merge_collinear_fragments(xyz,labels,infer_instance_axes(xyz,labels),merge_gap,merge_lateral)
    # Primero se rellena otro/ruido con los segmentos ya observados.
    missing=labels<0; known=labels>=0
    reassigned=0
    if missing.any() and known.any() and reassign_distance>0:
        distance,near=cKDTree(xyz[known]).query(xyz[missing],workers=2)
        accepted=distance<reassign_distance
        ids=np.flatnonzero(missing)[accepted]
        labels[ids]=labels[known][near[accepted]];reassigned=len(ids)
    # No perder equipos volumétricos lejos de acero: clustering del otro restante.
    missing=labels<0
    if missing.sum()>=min_samples:
        lab=object_labels(xyz[missing],None if normals is None else normals[missing],
                          method,eps,min_samples,min_cluster_size,height)
        good=lab>=0; ids=np.flatnonzero(missing)
        labels[ids[good]]=lab[good]+max(int(labels.max())+1,0)
    # Misma segunda fusión en ambos brazos de la ablación. Lo único que
    # cambia con enabled es la partición inicial por orientación.
    labels=merge_collinear_fragments(xyz,labels,infer_instance_axes(xyz,labels),
                                     merge_gap,merge_lateral,1200)
    # Renumeración contigua, sin cambiar las relaciones entre puntos.
    unique=np.unique(labels[labels>=0]); good=labels>=0
    labels[good]=np.searchsorted(unique,labels[good])
    return labels,orientation,dict(n_reasignados=reassigned,n_segmentos=len(unique))


def summarize_classes(xyz, labels, planes):
    """Reglas sobre la nube efectivamente etiquetada, no porcentajes voxel."""
    desc=describe_instances(xyz,labels,planes)
    n=max(len(labels),1)
    classes=desc.groupby('clase_heuristica',as_index=False).agg(
        n_segmentos=('id','count'),n_puntos=('n_puntos','sum'))
    classes['porcentaje']=100*classes.n_puntos/n
    return desc,classes


def geometric_comparison(Xgeo, labels_full, common_idx, base_full, class_map=None,
                         min_points=500, matched_k=11):
    """Distancia fija; segmentos grandes, soporte pareado y control K=11.

    El filtro de tamaño NO elimina el sesgo por K. Por eso se incluye además
    la agregación a once grupos: k-means de centroides de segmentos ponderados
    por sus tamaños comunes, manteniendo cada segmento completo. Es una
    coarsening geométrica y no un nuevo descubrimiento de objetos.
    ARI/NMI frente a clase solo miden concordancia de particiones.
    """
    from sklearn.metrics import normalized_mutual_info_score
    ids,counts=np.unique(labels_full,return_counts=True)
    big=ids[(ids>=0)&(counts>=min_points)]
    labels=np.asarray(labels_full)[common_idx]
    keep=np.isin(labels,big)
    base=np.asarray(base_full)[common_idx]
    b_ids,b_counts=np.unique(base_full,return_counts=True)
    shared=keep & np.isin(base,b_ids[(b_ids>=0)&(b_counts>=min_points)])
    sizes=counts[ids>=0]
    result=dict(n_segmentos_500=int(len(big)),cobertura_500_pct=100*float(keep.mean()),
        mayor_segmento_pct=100*float(sizes.max()/len(labels_full)) if len(sizes) else 0.,
        silhouette_geo=silhouette_sample(Xgeo[shared],labels[shared],n=2500,seed=0),
        DB_geo=davies_bouldin(Xgeo[shared],labels[shared]),
        silhouette_base_pareado=silhouette_sample(Xgeo[shared],base[shared],n=2500,seed=0),
        DB_base_pareado=davies_bouldin(Xgeo[shared],base[shared]),n_soporte_pareado=int(shared.sum()))
    u=np.unique(labels[shared]); coarsened=np.full(len(labels),-1,np.int32)
    if len(u)>=matched_k:
        centers=np.array([Xgeo[shared & (labels==label)].mean(axis=0) for label in u])
        weights=np.array([np.sum(shared & (labels==label)) for label in u])
        group=KMeans(n_clusters=matched_k,n_init=10,random_state=0).fit(centers,sample_weight=weights).labels_
        mask=np.isin(labels,u); coarsened[mask]=group[np.searchsorted(u,labels[mask])]
    else:
        coarsened=labels.copy()
    result.update(n_geo_K11=int(len(np.unique(coarsened[shared]))),
        silhouette_geo_K11=silhouette_sample(Xgeo[shared],coarsened[shared],n=2500,seed=0),
        DB_geo_K11=davies_bouldin(Xgeo[shared],coarsened[shared]))
    if class_map is not None:
        classes=np.array([class_map[int(label)] for label in labels])
        result['ARI_clases_base']=float(adjusted_rand_score(base,classes))
        result['NMI_clases_base']=float(normalized_mutual_info_score(base,classes))
    else:
        result['ARI_clases_base']=float('nan');result['NMI_clases_base']=float('nan')
    return result


def density_1d(values, eps, min_samples):
    """DBSCAN 1D sin materializar vecindarios cuadráticos; empate: core menor."""
    values=np.asarray(values,float)
    order=np.argsort(values,kind='stable'); v=values[order]
    labels=np.full(len(v),-1,np.int32)
    count=np.searchsorted(v,v+eps,side='right')-np.searchsorted(v,v-eps,side='left')
    core=np.flatnonzero(count>=min_samples)
    if not len(core): return labels
    cv=v[core]; groups=np.cumsum(np.r_[False,np.diff(cv)>eps]).astype(np.int32)
    j=np.searchsorted(cv,v);right=np.minimum(j,len(cv)-1);left=np.maximum(j-1,0)
    near=np.where(np.abs(v-cv[left])<=np.abs(v-cv[right]),left,right)
    good=np.abs(v-cv[near])<=eps; labels[good]=groups[near[good]]
    result=np.empty_like(labels);result[order]=labels
    return result


def infer_instance_axes(xyz, labels):
    """Ejes globales confirmados por PCA para fusión, con/sin orientación."""
    result={}
    for label in np.unique(labels[labels>=0]):
        q=xyz[labels==label].astype(float)
        if len(q)<6: continue
        eig,vec=np.linalg.eigh(np.cov(q.T))
        if (eig[-1]-eig[-2])/max(eig[-1],1e-12)<.55: continue
        direction=np.abs(vec[:,-1]);axis=int(direction.argmax())
        if direction[axis]>.85: result[int(label)]=axis
    return result
