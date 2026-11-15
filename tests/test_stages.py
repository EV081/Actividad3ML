"""Verificaciones geométricas de las etapas nuevas, sin editar helpers de clase."""
import sys
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcseg.stages import (ransac_plane, extract_planes, assign_planes, plane_distance,
    floor_metrics, local_pca, describe_instances, split_verticals, refine_color, pipeline)


def test_ransac_ignores_more_numerous_incompatible_normals():
    rng=np.random.default_rng(0)
    floor=np.c_[rng.uniform(-5,5,(1000,2)),rng.normal(0,.005,1000)]
    wall=np.c_[np.zeros(1600),rng.uniform(-5,5,(1600,2))]
    xyz=np.r_[floor,wall]
    normals=np.r_[np.tile([0,0,1],(1000,1)),np.tile([1,0,0],(1600,1))]
    coef,mask=ransac_plane(xyz,normals,threshold=.03)
    assert mask[:1000].mean()>.99 and not mask[1000:].any()
    assert abs(coef[2])>.999 and abs(coef[3])<.002
    assert_allclose(coef,ransac_plane(xyz,normals,threshold=.03)[0])


def test_distance_propagation_joins_patches_and_prioritizes_floor():
    planes=[dict(coef=[0,0,1,0],patches=[[0,0,1,0],[0,0,1,-.2]],z=0.,**{'class':'piso'}),
            dict(coef=[1,0,0,0],z=2.,**{'class':'pared'})]
    xyz=np.array([[0,100,.01],[50,100,.21],[0,0,2],[50,100,.5]])
    labels=assign_planes(xyz,planes,.08)
    assert_array_equal(labels,[0,0,1,-1])
    assert floor_metrics(xyz,labels,planes)['piso_en_un_segmento']==100
    assert_allclose(plane_distance(xyz,planes[0])[:2],[.01,.01])


def test_extract_merges_low_surfaces_but_keeps_roof():
    rng=np.random.default_rng(0)
    a=np.c_[rng.uniform(-5,5,(1200,2)),np.zeros(1200)]
    b=np.c_[rng.uniform(6,16,(900,2)),np.full(900,.22)]
    roof=np.c_[rng.uniform(-5,5,(1500,2)),np.full(1500,7.)]
    p=np.r_[a,b,roof]; n=np.tile([0,0,1],(len(p),1))
    planes=extract_planes(p,n,.04)
    assert sum(q['class']=='piso' for q in planes)==1
    assert sum(q['class']=='techo' for q in planes)==1
    labels=assign_planes(p,planes,.04)
    # RANSAC puede dejar unos pocos extremos fuera del umbral, sin partirlos.
    assert np.mean(labels[:2100]==0)>.99
    assert labels[0]!=labels[-1]


def test_local_pca_and_instance_rules_distinguish_axis_orientation():
    z=np.linspace(0,6,80)
    vertical=np.c_[np.zeros(80),np.zeros(80),z]
    horizontal=np.c_[z,np.ones(80),np.ones(80)]
    p=np.r_[vertical,horizontal]
    local=local_pca(vertical,10)
    assert_allclose(local[:,0],1,atol=1e-10)
    desc=describe_instances(p,np.repeat([0,1],80),[],np.r_[local,local])
    assert desc.clase_heuristica.tolist()==['columna','viga_o_tuberia']
    assert (desc.n_puntos==80).all()


def test_vertical_rule_splits_connected_object_into_distinct_fustes():
    rng=np.random.default_rng(0)
    # Dos fustes unidos por una viga: la entrada tiene un único ID de objeto.
    rods=[]
    for x in (0.,3.):
        rods.append(np.c_[x+rng.normal(0,.025,350),rng.normal(0,.025,350),np.linspace(0,5,350)])
    beam=np.c_[np.linspace(0,3,200),np.zeros(200),np.full(200,5.)]
    xyz=np.r_[*rods,beam]
    labels=split_verticals(xyz,np.zeros(len(xyz),dtype=int),[])
    desc=describe_instances(xyz,labels,[])
    assert (desc.clase_heuristica=='columna').sum()>=2
    assert labels[100]!=labels[450]


def test_color_cannot_split_small_compact_object_or_plane():
    rng=np.random.default_rng(0)
    xyz=rng.uniform(0,.5,(300,3)); rgb=np.r_[np.zeros((150,3)),np.full((150,3),255)].astype(np.uint8)
    labels=np.zeros(300,dtype=int)
    actual,diagnostic=refine_color(xyz,rgb,labels,[],True)
    assert_array_equal(actual,labels)
    assert diagnostic.empty


def test_pipeline_preserves_one_assignment_per_query_and_plane_distance():
    rng=np.random.default_rng(0)
    xyz=np.c_[rng.uniform(-5,5,(3000,2)),np.zeros(3000)]
    cloud={'xyz':xyz,'rgb':np.zeros((len(xyz),3),np.uint8),'normals':np.tile([0,0,1],(len(xyz),1))}
    # Query muy lejos de cualquier representante, pero sobre el mismo plano.
    query=np.array([[100.,100.,.01],[100.,100.,1.]])
    result=pipeline(cloud,query,voxel=.15,split_columns=False)
    assert result['labels'].shape==(2,)
    assert result['labels'][0]==0 and result['labels'][1]==-1


def test_without_height_still_uses_xyz_and_keeps_separate_elevations():
    from pcseg.stages import object_labels
    rng=np.random.default_rng(0)
    low=rng.normal(0,.01,(40,3))
    high=low+[0,0,5]
    labels=object_labels(np.r_[low,high],height=False,eps=.1,min_samples=5)
    assert labels[0]>=0 and labels[40]>=0 and labels[0]!=labels[40]


def test_local_pca_features_vertical_line_and_horizontal_line():
    from pcseg.stages import local_pca_features,orientation_classes
    t=np.linspace(-4,4,121)
    vertical=np.c_[t*0,t*0,t]
    horizontal=np.c_[t,t*0,t*0]
    for points,direction,name in ((vertical,2,'lineal_vertical'),(horizontal,0,'lineal_horizontal_x')):
        f=local_pca_features(points,k=20)
        assert_allclose(f['linealidad'],1,atol=1e-12)
        assert_allclose(f['planaridad'],0,atol=1e-12)
        assert_allclose(np.abs(f['e1'][:,direction]),1,atol=1e-12)
        assert (orientation_classes(f)==name).all()
        assert (f['eigenvalues'][:,0]>0).all()
        assert (np.diff(f['eigenvalues'],axis=1)<=1e-12).all()


def test_orientation_planar_normal_is_separate_from_principal_direction():
    from pcseg.stages import local_pca_features,orientation_classes
    a,b=np.meshgrid(np.linspace(-3,3,41),np.linspace(-3,3,41))
    horizontal=np.c_[a.ravel(),b.ravel(),np.zeros(a.size)]
    f=local_pca_features(horizontal,k=49)
    center=(np.max(np.abs(horizontal[:,:2]),axis=1)<2)
    assert np.median(f['planaridad'][center])>.7
    assert np.all(np.abs(f['normal'][center,2])>.99)
    assert np.mean(orientation_classes(f)[center]=='planar_normal_vertical')>.95
    vertical=horizontal[:,[0,2,1]]
    f=local_pca_features(vertical,k=49)
    assert np.mean(orientation_classes(f)[center]=='planar_normal_horizontal')>.95


def test_degenerate_pca_points_are_not_lines_or_planes():
    from pcseg.stages import local_pca_features,orientation_classes
    f=local_pca_features(np.zeros((20,3)),k=8)
    assert (orientation_classes(f)=='otro').all()
    assert (f['linealidad']==0).all()
    assert local_pca_features(np.empty((0,3)))['e1'].shape==(0,3)


def test_occupied_area_rejects_empty_frame_as_wall():
    from pcseg.stages import plane_support_area
    t=np.linspace(0,10,200)
    frame=np.r_[np.c_[t,t*0,t*0],np.c_[t,t*0,t*0+10],np.c_[t*0,t*0,t],np.c_[t*0+10,t*0,t]]
    area=plane_support_area(frame,[0,1,0,0])
    assert area['area_bbox']>=99
    assert area['cobertura_grid']<.2
    a,b=np.meshgrid(np.linspace(0,10,80),np.linspace(0,10,80))
    filled=np.c_[a.ravel(),np.zeros(a.size),b.ravel()]
    assert plane_support_area(filled,[0,1,0,0])['cobertura_grid']>.9


def test_orientation_separates_touching_columns_and_horizontal_beam():
    from pcseg.stages import cluster_orientations,local_pca_features
    t=np.linspace(0,5,301)
    a=np.c_[t*0,t*0,t]; b=a+[3,0,0]
    beam=np.c_[np.linspace(0,3,181),np.zeros(181),np.full(181,5.)]
    points=np.r_[a,b,beam]
    f=local_pca_features(points,k=24)
    labels,orientation,_=cluster_orientations(points,f,eps=.12,min_samples=4,
        reassign_distance=.15,merge_lateral=.2,merge_gap=.4)
    assert labels[50]>=0 and labels[351]>=0
    assert labels[50]!=labels[351]
    assert labels[50]!=labels[len(a)+len(b)+80]
    assert orientation[50]=='lineal_vertical'


def test_merging_does_not_join_different_axes_or_parallel_columns():
    from pcseg.stages import merge_collinear_fragments
    t=np.linspace(0,1,20)
    a=np.c_[t*0,t*0,t]; b=a+[0,0,1.2]; c=a+[2,0,0]; d=a[:,[2,1,0]]
    points=np.r_[a,b,c,d]; labels=np.repeat(np.arange(4),20)
    actual=merge_collinear_fragments(points,labels,{0:2,1:2,2:2,3:0},gap=.3,lateral=.2)
    assert actual[0]==actual[20] and actual[0]!=actual[40] and actual[0]!=actual[60]


def test_orientation_ablation_keeps_the_same_collinear_merging():
    from pcseg.stages import cluster_orientations,local_pca_features
    t=np.linspace(0,1,41)
    a=np.c_[t*0,t*0,t]; b=a+[0,0,1.25]
    points=np.r_[a,b];f=local_pca_features(points,k=8)
    options=dict(eps=.06,min_samples=3,merge_gap=.4,merge_lateral=.1,reassign_distance=.1)
    yes=cluster_orientations(points,f,enabled=True,**options)[0]
    no=cluster_orientations(points,f,enabled=False,**options)[0]
    assert len(np.unique(yes))==1 and len(np.unique(no))==1
    assert_array_equal(yes,no)


def test_fair_comparison_filters_using_full_cloud_counts_and_paired_support():
    from pcseg.stages import geometric_comparison
    full=np.repeat([0,1,2],[600,600,100])
    base=np.repeat([0,1,1],[600,600,100])
    idx=np.r_[np.arange(0,600,20),np.arange(600,1200,20),np.arange(1200,1300,10)]
    X=np.c_[full[idx]*10.,np.arange(len(idx))*.001]
    result=geometric_comparison(X,full,idx,base,{0:'piso',1:'viga',2:'mezcla'},matched_k=2)
    assert result['n_segmentos_500']==2
    assert result['n_soporte_pareado']==60
    assert_allclose(result['cobertura_500_pct'],100*60/70)
    assert_allclose(result['mayor_segmento_pct'],100*600/1300)
    assert result['n_geo_K11']==2 and result['silhouette_geo']>.9
    assert_allclose(result['silhouette_geo'],result['silhouette_base_pareado'])
    assert 0<=result['NMI_clases_base']<=1
