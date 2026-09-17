"""Method-blind two-level self-collision classification; never edit model assets.

Original proxies nominate pairs. Exact compiled CAD triangles confirm surfaces.
Closed, consistently oriented shells support solid containment checks. An open
or degenerate shell is enclosed conservatively, never silently treated as empty.
Overlap involving such an enclosure is UNRESOLVED, not a physical collision.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree


def geometry_libraries():
    """Use installed pure-Python geometry package, without installing/upgrading."""
    try:
        import trimesh
        import rtree
    except ModuleNotFoundError:
        # The repository's Isaac environment already supplies these packages.
        # Load the dependencies here and remove the cross-environment search path.
        location = "/home/jbnu/miniconda3/envs/isaaclab/lib/python3.11/site-packages"
        sys.path.append(location)
        try:
            trimesh = importlib.import_module("trimesh")
            rtree = importlib.import_module("rtree")
            importlib.import_module("networkx")
        finally:
            sys.path.remove(location)
    return trimesh, rtree


trimesh, rtree = geometry_libraries()
BLOCKING = {"HARD_SELF_COLLISION", "UNRESOLVED_GEOMETRY"}
MAX_TRIANGLE_PAIRS = 2_000_000


def _edge_distance(a, b, c, d):
    u, v, w = b-a, d-c, a-c
    aa = np.einsum("ij,ij->i", u, u)
    bb = np.einsum("ij,ij->i", u, v)
    cc = np.einsum("ij,ij->i", v, v)
    dd = np.einsum("ij,ij->i", u, w)
    ee = np.einsum("ij,ij->i", v, w)
    den = aa*cc-bb*bb
    safe = np.where(np.abs(den)>1e-30, den, 1.)
    s, t = (bb*ee-cc*dd)/safe, (aa*ee-bb*dd)/safe
    valid = (np.abs(den)>1e-30)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    best = np.where(valid, np.linalg.norm(w+s[:,None]*u-t[:,None]*v,axis=1), np.inf)
    for p, x, y in ((a,c,d),(b,c,d),(c,a,b),(d,a,b)):
        vec=y-x
        frac=np.clip(np.einsum("ij,ij->i",p-x,vec)/np.maximum(np.einsum("ij,ij->i",vec,vec),1e-30),0,1)
        best=np.minimum(best,np.linalg.norm(p-x-frac[:,None]*vec,axis=1))
    return best


def _segment_hits(tri, start, end):
    direction=end-start
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    h=np.cross(direction,e2); determinant=np.einsum("ij,ij->i",e1,h)
    valid=np.abs(determinant)>1e-25
    inv=np.where(valid,1/np.where(valid,determinant,1.),0.)
    s=start-tri[:,0]; u=inv*np.einsum("ij,ij->i",s,h)
    q=np.cross(s,e1); v=inv*np.einsum("ij,ij->i",direction,q)
    fraction=inv*np.einsum("ij,ij->i",e2,q)
    hit=valid&(u>=0)&(v>=0)&(u+v<=1)&(fraction>=0)&(fraction<=1)
    return hit, start[hit]+fraction[hit,None]*direction[hit]


def surface_distance(first, second):
    """Global triangle-surface distance, including crossing edges and coplanarity.

    A nearest-vertex distance is an upper bound, not an acceptance threshold.
    Expanded triangle AABBs cannot exclude a closer triangle pair.
    """
    if not len(first.faces) or not len(second.faces):
        raise ValueError("empty detailed surface")
    if len(first.faces)>len(second.faces):
        first,second=second,first
    upper=float(np.min(cKDTree(second.vertices).query(first.vertices)[0]))+1e-12
    ta=np.asarray(first.triangles); tb=np.asarray(second.triangles)
    bounds=np.column_stack((ta.min(axis=1)-upper,ta.max(axis=1)+upper))
    ia=[];ib=[]
    tree=second.triangles_tree
    for index,bound in enumerate(bounds):
        ids=list(tree.intersection(bound))
        ia.extend([index]*len(ids));ib.extend(ids)
        if len(ia)>MAX_TRIANGLE_PAIRS:
            raise RuntimeError("finite geometry triangle-pair budget exceeded")
    minimum=upper; intersections=0; samples=[]
    for begin in range(0,len(ia),20000):
        a=ta[np.asarray(ia[begin:begin+20000],dtype=int)]
        b=tb[np.asarray(ib[begin:begin+20000],dtype=int)]
        for k in range(3):
            for x,y in ((a,b),(b,a)):
                closest=trimesh.triangles.closest_point(x,y[:,k])
                minimum=min(minimum,float(np.min(np.linalg.norm(closest-y[:,k],axis=1))))
                hit,points=_segment_hits(x,y[:,k],y[:,(k+1)%3])
                intersections+=int(hit.sum())
                if len(points):
                    minimum=0.
                    if len(samples)<128:samples.extend(points[:128-len(samples)].tolist())
            for j in range(3):
                minimum=min(minimum,float(np.min(_edge_distance(a[:,k],a[:,(k+1)%3],b[:,j],b[:,(j+1)%3]))))
    return {"separation_m":minimum,"edge_surface_intersections":intersections,
            "intersection_samples":samples,"triangle_pairs":len(ia)}


def solid_components(mesh, tolerance):
    """Preserve all triangles; use conservative enclosures for uncertain shells."""
    parts=[]
    # Exact coincident-vertex welding only; no coordinate rounding or hole filling.
    vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True)
    clean=trimesh.Trimesh(vertices=vertices,faces=inverse[mesh.faces],process=False)
    for shell in clean.split(only_watertight=False,repair=False):
        reliable=bool(shell.is_watertight and shell.is_winding_consistent and shell.volume>tolerance**3)
        if reliable:
            certificate=shell
        else:
            try:
                certificate=shell.convex_hull
                if not certificate.is_volume:raise ValueError("degenerate hull")
            except Exception:
                certificate=trimesh.creation.box(extents=np.maximum(shell.extents,tolerance))
                certificate.apply_translation(shell.bounds.mean(axis=0))
        parts.append((certificate,reliable))
    if not parts:raise ValueError("no connected detailed shells")
    return parts


def _contains_checked(mesh,points):
    """Deterministic bidirectional parity; ambiguous rays fail closed."""
    answers=[]
    for direction in ((.4395064455,.617598629942,.652231566745),(.727,-.423,.541)):
        directions=np.tile(np.asarray(direction),(len(points),1))
        _,indices,_=mesh.ray.intersects_location(np.vstack((points,points)),np.vstack((directions,-directions)))
        counts=np.bincount(indices,minlength=2*len(points)).reshape(2,-1)%2
        if np.any(counts[0]!=counts[1]):raise ValueError('ambiguous bidirectional solid-containment rays')
        answers.append(counts[0].astype(bool))
    if np.any(answers[0]!=answers[1]):raise ValueError('ambiguous cross-direction solid-containment rays')
    return answers[0]


def _inside_depth(mesh,points,tolerance,also_inside=None):
    if not len(points):return 0.
    points=np.asarray(points)
    keep=np.all(points>mesh.bounds[0],axis=1)&np.all(points<mesh.bounds[1],axis=1)
    points=points[keep]
    if not len(points):return 0.
    # A strict interior witness suffices: report a penetration lower bound,
    # not the deepest vertex. Batching avoids unbounded ray-query working sets.
    for begin in range(0,len(points),32):
        batch=points[begin:begin+32]
        _,distance,_=trimesh.proximity.closest_point(mesh,batch)
        keep=distance>tolerance
        batch,distance=batch[keep],distance[keep]
        if not len(batch):continue
        if also_inside is not None:
            _,other_distance,_=trimesh.proximity.closest_point(also_inside,batch)
            keep=other_distance>tolerance
            batch,distance=batch[keep],np.minimum(distance[keep],other_distance[keep])
            if not len(batch):continue
        inside=_contains_checked(mesh,batch)
        if also_inside is not None:inside&=_contains_checked(also_inside,batch)
        if np.any(inside):return float(np.max(distance[inside]))
    return 0.


def classify_mesh_pair(first,second,tolerance,parts_first=None,parts_second=None):
    """No robot, side, representation, episode, or task label enters this rule."""
    raw=surface_distance(first,second)
    pa=parts_first if parts_first is not None else solid_components(first,tolerance)
    pb=parts_second if parts_second is not None else solid_components(second,tolerance)
    unresolved=[]; hard_depth=0.; certificate_min=np.inf
    for a,reliable_a in pa:
        for b,reliable_b in pb:
            box_gap=float(np.linalg.norm(np.maximum(np.maximum(a.bounds[0]-b.bounds[1],b.bounds[0]-a.bounds[1]),0)))
            if box_gap>tolerance:
                certificate_min=min(certificate_min,box_gap)
                continue
            detail=surface_distance(a,b)
            gap=detail["separation_m"]
            certificate_min=min(certificate_min,gap)
            # With separated closed surfaces, one point per connected shell
            # resolves full containment. Intersections need interior witnesses.
            if gap>tolerance:
                depth=max(_inside_depth(a,b.vertices[:1],tolerance),_inside_depth(b,a.vertices[:1],tolerance))
                if depth<=tolerance:continue
            else:
                points_a=a.vertices
                points_b=b.vertices
                interior_depth=0.
                if detail["intersection_samples"]:
                    hits=np.asarray(detail["intersection_samples"])
                    mids=(hits[:-1]+hits[1:])/2
                    if len(mids):
                        interior_depth=_inside_depth(a,mids,tolerance,also_inside=b)
                depth=interior_depth
                if depth<=tolerance:depth=_inside_depth(a,points_b,tolerance)
                if depth<=tolerance:depth=_inside_depth(b,points_a,tolerance)
            if reliable_a and reliable_b and depth>tolerance:
                hard_depth=max(hard_depth,depth)
            else:
                unresolved.append({"reason":"UNCERTAIN_SHELL_ENCLOSURE_OVERLAP" if not(reliable_a and reliable_b) else "NEAR_SURFACE_WITHOUT_STRICT_INTERIOR_WITNESS",
                                   "first_solid_reliable":reliable_a,"second_solid_reliable":reliable_b,
                                   "certificate_surface_separation_m":gap})
    classification="HARD_SELF_COLLISION" if hard_depth>tolerance else "UNRESOLVED_GEOMETRY" if unresolved else "PROXY_ONLY_OVERLAP"
    return {"classification":classification,"detailed_collision":True if classification=="HARD_SELF_COLLISION" else False if classification=="PROXY_ONLY_OVERLAP" else None,
            "detailed_separation_mm":1000*raw["separation_m"],"detailed_penetration_lower_bound_mm":1000*hard_depth,
            "detailed_surface_edge_intersections":raw["edge_surface_intersections"],
            "certificate_separation_lower_bound_mm":1000*certificate_min if np.isfinite(certificate_min) else None,
            "uncertain_shells":[sum(not ok for _,ok in pa),sum(not ok for _,ok in pb)],"unresolved_reasons":unresolved}


class GeometryConfirmedCollision:
    """Drop-in collision-record provider for the unchanged position solver."""
    def __init__(self,proxy,tolerance):
        if float(proxy.collision_tolerance)!=float(tolerance):
            raise ValueError("geometry tolerance must equal existing common tolerance")
        self.proxy=proxy;self.g1=proxy.g1;self.tolerance=float(tolerance)
        self.body_meshes={};self.cache=OrderedDict();self.last_records=[]
        self.mesh_inventory={};self.calls=0

    def body_geometry(self,body):
        if body not in self.body_meshes:
            model=self.g1.model;pieces=[];inventory=[]
            for gi in range(model.ngeom):
                if int(model.geom_bodyid[gi])!=body or int(model.geom_group[gi])!=2:continue
                if int(model.geom_type[gi])!=int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise ValueError("detailed non-mesh visual geometry is not supported")
                mi=int(model.geom_dataid[gi]);va=int(model.mesh_vertadr[mi]);nv=int(model.mesh_vertnum[mi]);fa=int(model.mesh_faceadr[mi]);nf=int(model.mesh_facenum[mi])
                vertices=model.mesh_vert[va:va+nv].astype(float)
                rot=np.empty(9);mujoco.mju_quat2Mat(rot,model.geom_quat[gi])
                vertices=vertices@rot.reshape(3,3).T+model.geom_pos[gi]
                faces=model.mesh_face[fa:fa+nf].copy()
                pieces.append(trimesh.Trimesh(vertices=vertices,faces=faces,process=False))
                inventory.append({"geom_id":gi,"mesh_id":mi,"vertices":nv,"triangles":nf,
                                  "compiled_geometry_sha256":hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()})
            if not pieces:raise ValueError("no detailed mesh for proxy body")
            mesh=trimesh.util.concatenate(pieces)
            self.body_meshes[body]=(mesh,solid_components(mesh,self.tolerance))
            self.mesh_inventory[str(body)]={"body":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body),"meshes":inventory,
                "solid_components":len(self.body_meshes[body][1]),"uncertain_components":sum(not ok for _,ok in self.body_meshes[body][1])}
        mesh,parts=self.body_meshes[body]
        transform=np.eye(4);transform[:3,:3]=self.g1.data.xmat[body].reshape(3,3);transform[:3,3]=self.g1.data.xpos[body]
        return mesh,parts,transform

    def inspect(self,q,left_hand,right_hand):
        records=self.proxy._records(q,left_hand,right_hand)
        unique={}
        for record in records:
            pair=tuple(record["geom_pair"])
            if pair not in unique or record["distance_m"]<unique[pair]["distance_m"]:unique[pair]=record
        result=[]
        for pair,record in unique.items():
            bodies=tuple(int(self.g1.model.geom_bodyid[g]) for g in pair)
            # Exact relative transforms, not rounded q, are the cache keys.
            ra=self.g1.data.xmat[bodies[0]].reshape(3,3);rb=self.g1.data.xmat[bodies[1]].reshape(3,3)
            relative=np.eye(4);relative[:3,:3]=ra.T@rb;relative[:3,3]=ra.T@(self.g1.data.xpos[bodies[1]]-self.g1.data.xpos[bodies[0]])
            key=(bodies,relative.tobytes())
            try:
                if key not in self.cache:
                    ma,pa,_=self.body_geometry(bodies[0]);mb,pb,_=self.body_geometry(bodies[1])
                    mb=mb.copy();mb.apply_transform(relative)
                    pb=[(m.copy().apply_transform(relative),ok) for m,ok in pb]
                    self.cache[key]=classify_mesh_pair(ma,mb,self.tolerance,pa,pb)
                    self.calls+=1
                    if len(self.cache)>4096:self.cache.popitem(last=False)
                detailed=self.cache[key]
            except Exception as exc:
                detailed={"classification":"UNRESOLVED_GEOMETRY","detailed_collision":None,"detailed_separation_mm":None,
                          "unresolved_reasons":[{"reason":type(exc).__name__,"detail":str(exc)}]}
            result.append({**record,"proxy_classification":record["classification"],"proxy_collision":True,
                           "proxy_penetration_mm":1000*record["penetration_depth_m"],**detailed})
        self.last_records=result
        return result

    def _records(self,q,left_hand,right_hand):
        return [r for r in self.inspect(q,left_hand,right_hand) if r["classification"] in BLOCKING]
