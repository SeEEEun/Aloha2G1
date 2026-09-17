"""Read-only authored USD hull adapter for the existing MuJoCo checker.

Run export with the installed Isaac Python (pxr); load with offline MuJoCo.
Runtime geometry/filters are never edited. Hulls are solid convex geometry,
not a watertightness claim about the visual triangle surfaces.
"""
from pathlib import Path
from collections import OrderedDict
import copy
import time
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_npz
from .source_phase import COMMON


def export(out):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf
    import ast
    from scipy.spatial import ConvexHull
    cfg=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')
    stage=Usd.Stage.Open(cfg['source_scene']);stage.SetEditTarget(stage.GetSessionLayer())
    engine=ROOT/'tools/run_doll_handoff_graspable_proxy_v2_isaac.py';tree=ast.parse(engine.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_define_beveled_wall_mesh','apply_bin_height'}]
    for node in nodes:
        node.returns=None
        for arg in node.args.args:arg.annotation=None
    namespace=dict(np=np,Usd=Usd,UsdGeom=UsdGeom,UsdPhysics=UsdPhysics,UsdShade=UsdShade,Gf=Gf,
        BIN_PARTS={n:'/World/DollHandoffEnvironment/TrashBin/'+n for n in ('Bottom','FrontWall','BackWall','LeftWall','RightWall')})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(engine),'exec'),namespace)
    bin_geometry=namespace['apply_bin_height'](stage,.150,.003)
    cache=UsdGeom.XformCache(); rows=[];arrays={}
    for p in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:continue
        path=str(p.GetPath())
        if '/World/G1/Asset/' in path:
            link=p.GetParent();name=link.GetName()
            if any(x in name for x in ('camera_base','palm_link')):
                name=name.split('_hand_')[0]+'_wrist_yaw_link'
                link=stage.GetPrimAtPath('/World/G1/Asset/'+name)
            children=[x for x in Usd.PrimRange(p,Usd.TraverseInstanceProxies()) if x.IsA(UsdGeom.Mesh)]
            for meshprim in children:
                points=np.asarray(UsdGeom.Mesh(meshprim).GetPointsAttr().Get(),float)
                transform=np.asarray(cache.GetLocalToWorldTransform(meshprim)*cache.GetLocalToWorldTransform(link).GetInverse())
                points=(np.c_[points,np.ones(len(points))]@transform)[:,:3]
                vertices=points[ConvexHull(points).vertices];key=f'collider_{len(rows)}';arrays[key]=vertices
                rows.append(dict(key=key,body=name,path=str(meshprim.GetPath()),kind='robot_convex',vertex_count=len(vertices),
                    approximation=UsdPhysics.MeshCollisionAPI(meshprim).GetApproximationAttr().Get() or 'convexHull inherited/default',
                    transform_to_body=transform.T))
        elif p.IsA(UsdGeom.Cube) and '/World/DollHandoffEnvironment/' in path:
            size=float(UsdGeom.Cube(p).GetSizeAttr().Get());matrix=np.asarray(cache.GetLocalToWorldTransform(p)).T
            rows.append(dict(key=f'collider_{len(rows)}',path=path,kind='environment_box',size=size,world_matrix=matrix))
        elif p.IsA(UsdGeom.Mesh) and '/World/DollHandoffEnvironment/TrashBin/' in path:
            points=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(),float);matrix=np.asarray(cache.GetLocalToWorldTransform(p))
            points=(np.c_[points,np.ones(len(points))]@matrix)[:,:3];key=f'collider_{len(rows)}';arrays[key]=points
            rows.append(dict(key=key,path=path,kind='environment_convex'))
    destination=out/'target_repair/runtime_bin150'
    atomic_npz(destination/'RUNTIME_HULL_VERTICES.npz',**arrays)
    value=dict(rows=rows,source_scene=record(cfg['source_scene']),robot_usd=record(read(COMMON)['models']['g1_usd']),
        runtime_inventory=record(ROOT/'outputs/contact_coordination_hybrid/20260907T073437Z/common_control/scripted_captured/RUNTIME_COLLISION_MODEL.json'),
        parity='same authored point sets/local transforms; MuJoCo convex query; PhysX cooking/contact margins remain numerical implementation differences',
        self_response='runtime disabled; independent offline rejection of unwanted robot intersections retained',
        bin_geometry=bin_geometry,bin_geometry_function=record(engine),exporter=record(__file__),session_saved=False)
    atomic_json(destination/'RUNTIME_HULLS.json',value);return value


def object_dimensions(out=None):
    cfg=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')
    return np.asarray(next(row for row in cfg['geometry_candidates'] if row['name']=='INTERMEDIATE_PLUSH_PROXY')['dimensions_m'])


def protected_object_separation():
    """Positive separation before contact, from the existing object contract."""
    return float(read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json')['object']['contact_offset_m'])


def make_model(g1,out):
    import mujoco,xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    import ast
    # Reuse existing proxy vertex construction without importing its simulator.
    engine=ROOT/'tools/run_doll_handoff_graspable_proxy_v2_isaac.py'
    tree=ast.parse(engine.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rounded_oval_mesh')
    node.returns=None
    for a in node.args.args:a.annotation=None
    class Gf:
        Vec3f=staticmethod(lambda *x:x)
    import math
    ns={'np':np,'math':math,'Gf':Gf};exec(compile(ast.Module(body=[node],type_ignores=[]),str(engine),'exec'),ns)
    # mj_saveLastXML also uses MuJoCo's last-loaded XML specification. Another
    # Checker in the same process can otherwise donate its already-added hulls.
    # Reload this exact authoritative base model before serializing it.
    base_model=mujoco.MjModel.from_xml_path(str(g1.path))
    tmp=out/'target_repair/runtime_bin150/runtime_check.xml';mujoco.mj_saveLastXML(str(tmp),base_model);xml=ET.parse(tmp);root=xml.getroot()
    root.find('compiler').set('meshdir',str(g1.path.parent/'assets'))
    for geom in root.iter('geom'):
        geom.set('contype','0');geom.set('conaffinity','0')
    assets=root.find('asset');world=root.find('worldbody');bodies={b.get('name'):b for b in root.iter('body')}
    meta=read(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json');v=dict(np.load(out/'target_repair/runtime_bin150/RUNTIME_HULL_VERTICES.npz'));missing=[]
    names=[]
    for row in meta['rows']:
        key=row['key']
        if row['kind']=='robot_convex':
            if row['body'] not in bodies:missing.append(row['body']);continue
            ET.SubElement(assets,'mesh',name=key,vertex=' '.join(map(str,v[key].ravel())))
            ET.SubElement(bodies[row['body']],'geom',name=key,type='mesh',mesh=key,contype='1',conaffinity='1',rgba='0.5 0.5 0.5 0.2',group='4')
            names.append(key)
        elif row['kind']=='environment_convex':
            vertices=g1.world_to_model_position(v[key]);ET.SubElement(assets,'mesh',name=key,vertex=' '.join(map(str,vertices.ravel())))
            ET.SubElement(world,'geom',name=key,type='mesh',mesh=key,contype='1',conaffinity='1',group='4')
        else:
            t=np.asarray(row['world_matrix']);scales=np.linalg.norm(t[:3,:3],axis=0);rot=g1.root_pose[:3,:3].T@(t[:3,:3]/scales)
            pos=g1.world_to_model_position(t[:3,3]);quat=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
            ET.SubElement(world,'geom',name=key,type='box',size=' '.join(map(str,scales*row['size']/2)),pos=' '.join(map(str,pos)),quat=' '.join(map(str,quat)),contype='1',conaffinity='1',group='4')
    # Missing fixed leg detail may be irrelevant to arm motion but is explicit.
    unexpected=[x for x in missing if any(s in x for s in ('hand','wrist','arm','elbow','shoulder','torso'))]
    assert not unexpected,unexpected
    cfg=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json');dim=np.asarray(next(x for x in cfg['geometry_candidates'] if x['name']=='INTERMEDIATE_PLUSH_PROXY')['dimensions_m'])
    if 'object_vertex_key' in meta:
        # Explicit geometry bundle from the installed PhysX cooker. These
        # vertices already include the fixed collider-to-object transform.
        points=np.asarray(v[meta['object_vertex_key']])
    else:
        points,_,_=ns['rounded_oval_mesh'](dim);points=np.asarray(points);points[:,2]+=(dim[2]-cfg['object']['visual_dimensions_m'][2])/2
    ET.SubElement(assets,'mesh',name='hybrid_object',vertex=' '.join(map(str,points.ravel())))
    obj=ET.SubElement(world,'body',name='hybrid_object',pos='0 0 0')
    ET.SubElement(obj,'geom',name='hybrid_object',type='mesh',mesh='hybrid_object',contype='1',conaffinity='1',group='4')
    # Both predicted object and environment are static in this query model.
    # MuJoCo omits automatic contacts within a welded/static assembly; explicit
    # pairs make the carried-object clearance query independent of that filter.
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    for row in meta['rows']:
        if row['kind'].startswith('environment_'):
            ET.SubElement(contact,'pair',geom1='hybrid_object',geom2=row['key'])
    xml.write(tmp)
    model=mujoco.MjModel.from_xml_path(str(tmp))
    atomic_json(out/'target_repair/runtime_bin150/RUNTIME_CHECK_MODEL.json',dict(model=record(tmp),omitted_fixed_bodies=missing,
        robot_colliders=len(names),engine_mesh_function=record(engine),geometry=meta['parity'],runtime_filters_changed=False))
    return model,meta


class Checker:
    def __init__(self,g1,out,names):
        import mujoco
        from .incidental_contact import policy
        self.incidental_policy=policy(out)
        self.mj=mujoco;self.g1=g1;self.model,self.meta=make_model(g1,out);self.data=mujoco.MjData(self.model)
        self.qids=np.asarray([self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names])
        self.obj=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'hybrid_object')
        self.geo=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'hybrid_object')
        self.rows={r['key']:r for r in self.meta['rows']}
        self._geom_names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(self.model.ngeom)]
        self._pair_metadata={}
        self._query_cache=OrderedDict()
        self.query_statistics=dict(calls=0,hits=0,misses=0,narrow_phase_calls=0,seconds=0.)
        self._named_qpos={mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,j):int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)}
        self._clearance_cache=OrderedDict()

    def protected_object_contacts(self,q,x):
        """Reject near-contact too; ordinary penetration tolerance is unsuitable
        before intentional acquisition. This never changes scene geometry.
        """
        separation=protected_object_separation();self.check(q,x,())
        m=self.model;d=self.data;hits=[]
        for index,name in enumerate(self._geom_names):
            row=self.rows.get(name)
            if row is None or row['kind']!='robot_convex':continue
            if np.linalg.norm(d.geom_xpos[index]-d.geom_xpos[self.geo])-m.geom_rbound[index]-m.geom_rbound[self.geo]>separation:continue
            points=np.zeros(6);distance=float(self.mj.mj_geomDistance(m,d,self.geo,index,separation,points))
            if distance>=separation-1e-9:continue
            normal=points[3:]-points[:3];normal/=max(np.linalg.norm(normal),1e-12)
            hits.append(dict(bodies=['object',row['body']],geoms=['hybrid_object',name],
                depth_m=separation-distance,signed_distance_m=distance,required_separation_m=separation,
                point_world=self.g1.model_to_world_position((points[:3]+points[3:])/2),
                normal_world_geom0_to_geom1=self.g1.root_pose[:3,:3]@normal,allowed_contact=False,
                reason='PREGRASP_PROTECTED_OBJECT_CLEARANCE'))
        from .incidental_contact import filter_hits
        return filter_hits(hits,self.incidental_policy,x)

    def clearance(self,q,x,allow_digits=(),all_joint_state=None,object_environment=False,cap=.03):
        """Forbidden-pair signed distance, capped at30mm, for soft path quality.

        Hard acceptance still uses check()/query(), including their contact
        policy. Welded and direct-parent robot bodies have the same anatomical
        exclusions as automatic MuJoCo collision generation.
        """
        extra=tuple(sorted((all_joint_state or {}).items()))
        key=(np.asarray(q).tobytes(),np.asarray(x).tobytes(),tuple(allow_digits),extra,object_environment)
        if key in self._clearance_cache:return self._clearance_cache[key]
        self.check(q,x,allow_digits,all_joint_state,object_environment)
        m=self.model;d=self.data;ids=[i for i,n in enumerate(self._geom_names) if n in self.rows or n=='hybrid_object'];minimum=cap
        for offset,i in enumerate(ids):
            for j in ids[offset+1:]:
                ni,nj=self._geom_names[i],self._geom_names[j]
                ri=self.rows.get(ni,dict(kind='object',body='object'));rj=self.rows.get(nj,dict(kind='object',body='object'))
                robot_i=ri['kind']=='robot_convex';robot_j=rj['kind']=='robot_convex'
                if not(robot_i or robot_j) and not(object_environment and 'hybrid_object' in (ni,nj)):continue
                if robot_i and robot_j:
                    bi=int(m.geom_bodyid[i]);bj=int(m.geom_bodyid[j]);wi=int(m.body_weldid[bi]);wj=int(m.body_weldid[bj])
                    if wi==wj or m.body_weldid[m.body_parentid[wi]]==wj or m.body_weldid[m.body_parentid[wj]]==wi:continue
                    if all(not any(k in r.get('body','') for k in ('hand','wrist','elbow','shoulder')) for r in (ri,rj)):continue
                if 'hybrid_object' in (ni,nj):
                    body=rj.get('body','') if ni=='hybrid_object' else ri.get('body','')
                    if any(body.startswith(side+'_hand_') and any(k in body for k in ('thumb','middle','index')) for side in allow_digits):continue
                if np.linalg.norm(d.geom_xpos[i]-d.geom_xpos[j])-m.geom_rbound[i]-m.geom_rbound[j]>minimum:continue
                distance=self.mj.mj_geomDistance(m,d,i,j,cap,None)
                minimum=min(minimum,float(distance))
        self._clearance_cache[key]=minimum
        if len(self._clearance_cache)>2048:self._clearance_cache.popitem(last=False)
        return minimum

    def query(self,q,x,allow_digits=(),all_joint_state=None,object_environment=False):
        """Pure contact-result query; unlike check(), does not promise data state.

        Exact float64 bytes, no quantization. Cache is local to this immutable
        loaded model/config and includes every contact-mode and full-state input.
        Call check() when the caller subsequently inspects self.data geometry.
        """
        started=time.monotonic();self.query_statistics['calls']+=1
        extra=tuple(sorted((str(k),float(v)) for k,v in (all_joint_state or {}).items()))
        key=(np.asarray(q,dtype=np.float64).tobytes(),np.asarray(x,dtype=np.float64).tobytes(),tuple(allow_digits),extra,bool(object_environment))
        if key in self._query_cache:
            self.query_statistics['hits']+=1;self._query_cache.move_to_end(key)
            result=copy.deepcopy(self._query_cache[key])
        else:
            self.query_statistics['misses']+=1
            result=self.check(q,x,allow_digits,all_joint_state,object_environment)
            self._query_cache[key]=copy.deepcopy(result)
            if len(self._query_cache)>4096:self._query_cache.popitem(last=False)
        self.query_statistics['seconds']+=time.monotonic()-started
        return result

    def check(self,q,x,allow_digits=(),all_joint_state=None,object_environment=False):
        from scipy.spatial.transform import Rotation
        m=self.model;d=self.data;m.body_pos[self.obj]=self.g1.world_to_model_position(x[:3,3]);m.body_quat[self.obj]=Rotation.from_matrix(self.g1.root_pose[:3,:3].T@x[:3,:3]).as_quat()[[3,0,1,2]]
        d.qpos[:]=m.key_qpos[0];d.qpos[self.qids]=q
        if all_joint_state is not None:
            for name,value in all_joint_state.items():
                if str(name) not in self._named_qpos:raise ValueError('Runtime joint missing from offline model: '+str(name))
                d.qpos[self._named_qpos[str(name)]]=value
        # Geometry-only query: explicit static pairs have no dynamic constraint
        # tree and must not enter mj_forward's force/constraint-solver stages.
        self.mj.mj_kinematics(m,d);self.mj.mj_comPos(m,d);self.mj.mj_collision(m,d)
        self.query_statistics['narrow_phase_calls']+=1
        hits=[]
        for c in d.contact[:d.ncon]:
            if c.dist>=-1e-5:continue
            ids=[int(c.geom1),int(c.geom2)]
            metadata_key=(*ids,tuple(allow_digits),bool(object_environment))
            if metadata_key in self._pair_metadata:
                metadata=self._pair_metadata[metadata_key]
                if metadata is None:continue
                names,bodies,allowed=metadata
                normal=self.g1.root_pose[:3,:3]@np.asarray(c.frame[:3])
                hits.append(dict(bodies=bodies,geoms=names,depth_m=float(-c.dist),point_world=self.g1.model_to_world_position(c.pos),normal_world_geom0_to_geom1=normal,allowed_contact=allowed))
                continue
            self._pair_metadata[metadata_key]=None
            names=[self._geom_names[i] for i in ids]
            rows=[self.rows.get(n,dict(kind='object' if n=='hybrid_object' else 'disabled',body='object')) for n in names]
            if any(r['kind']=='disabled' for r in rows):continue
            if all(r['kind']!='robot_convex' for r in rows):
                if not (object_environment and 'hybrid_object' in names):continue
            bodies=[r.get('body',r.get('path','object')) for r in rows]
            if all(r['kind']=='robot_convex' for r in rows):
                # MuJoCo already filters welded/direct-parent anatomical
                # neighbors. Do not suppress nonadjacent same-hand collisions:
                # crossing fingers or a digit through the wrist are invalid.
                if all(not any(k in b for k in ('hand','wrist','elbow','shoulder')) for b in bodies):continue
            allowed=False
            if 'hybrid_object' in names:
                for side in allow_digits:
                    allowed |= any(b.startswith(side+'_hand_') and any(k in b for k in ('thumb','middle','index')) for b in bodies)
            self._pair_metadata[metadata_key]=(names,bodies,allowed)
            normal=self.g1.root_pose[:3,:3]@np.asarray(c.frame[:3])
            hits.append(dict(bodies=bodies,geoms=names,depth_m=float(-c.dist),point_world=self.g1.model_to_world_position(c.pos),normal_world_geom0_to_geom1=normal,allowed_contact=allowed))
        return hits


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();print('Exported',len(export(a.run_dir)['rows']),'authored colliders')
