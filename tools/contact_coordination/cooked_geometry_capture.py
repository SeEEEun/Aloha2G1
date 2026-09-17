"""Export cooked robot instance proxies without executing a control trajectory."""
import argparse
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def capture(stage, folder):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools, Sdf
    from omni.physx import get_physx_cooking_interface
    from tools.contact_coordination.io import atomic_json
    cache=UsdGeom.XformCache();cooking=get_physx_cooking_interface()
    stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt();rows=[];seen=set()
    for collider in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:continue
        for prim in Usd.PrimRange(collider,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()) in seen:continue
            seen.add(str(prim.GetPath()));result={}
            def received(status, convexes):
                result.update(status=str(status),convexes=[dict(
                    vertices=[[float(v.x),float(v.y),float(v.z)] for v in c.vertices],
                    indices=list(c.indices),polygons=[dict(index_base=p.index_base,
                    num_vertices=p.num_vertices) for p in c.polygons]) for c in convexes])
            # This installed API cannot parse inherited collision APIs on
            # instanced Xforms. Cook a separate in-memory mesh replica with
            # the same authored geometry and resolved collision attributes.
            # The running stage and its shapes are never modified.
            replica=Usd.Stage.CreateInMemory()
            replica.GetRootLayer().ImportFromString('''#usda 1.0
def Xform "Export" {}
''')
            geometry=UsdGeom.Mesh.Define(replica,'/Export/Mesh')
            source_mesh=UsdGeom.Mesh(prim)
            geometry.CreatePointsAttr(source_mesh.GetPointsAttr().Get())
            geometry.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get())
            geometry.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get())
            geometry.CreateSubdivisionSchemeAttr(source_mesh.GetSubdivisionSchemeAttr().Get())
            rp=geometry.GetPrim()
            UsdPhysics.CollisionAPI.Apply(rp)
            UsdPhysics.MeshCollisionAPI.Apply(rp)
            attrs={}
            for ancestor in [collider,prim]:
                for attr in ancestor.GetAttributes():
                    if attr.GetName().startswith(('physics:','physx')):
                        value=attr.Get()
                        if value is not None:
                            rp.CreateAttribute(attr.GetName(),attr.GetTypeName()).Set(value)
                            attrs[attr.GetName()]=str(value)
            replica_id=UsdUtils.StageCache.Get().Insert(replica).ToLongInt()
            cooking.request_convex_collision_representation(stage_id=replica_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(rp.GetPath()),
                run_asynchronously=False,on_result=received)
            body=prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
            world=np.asarray(cache.GetLocalToWorldTransform(prim)).T
            bt=np.asarray(cache.GetLocalToWorldTransform(body)).T if body else np.eye(4)
            rows.append(dict(path=str(prim.GetPath()),collision_api_path=str(collider.GetPath()),
                body=str(body.GetPath()) if body else None, T_world_mesh=world,
                T_body_mesh=np.linalg.inv(bt)@world,
                authored_points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                resolved_attributes=attrs,replica_cooking=True,**result))
    atomic_json(Path(folder)/'PHYSX_COOKED_COLLIDERS.json',dict(rows=rows,
        instance_proxies_traversed=True,physical_control_frames=0,read_only=True,
        runtime_colliders_or_filters_changed=False,api='request_convex_collision_representation',
        caveat='Same PhysX cooker on separate meshes with resolved inherited APIs; not direct extraction of attached PxShapes. Runtime force/contact agreement must still be checked.'))
    print('COOKED_EXPORT',len(rows),[(r['path'],r.get('status'),len(r.get('convexes',[]))) for r in rows],flush=True)


def main():
    from tools.contact_coordination import calibration_capture
    from tools import run_direct_physical_execution_isaac as engine
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--direct-freeze-manifest',type=Path,required=True)
    p.add_argument('--qualification-mode',action='store_true')
    a,remaining=p.parse_known_args()
    source,_=calibration_capture.instrument(engine.ENGINE.read_text())
    source=source.replace('from tools.contact_coordination.calibration_capture import capture_cooked',
        'from tools.contact_coordination.cooked_geometry_capture import capture as capture_cooked')
    source=source.replace('    capture_cooked(stage, output_dir)\n',
        '    capture_cooked(stage, output_dir)\n    return 0\n')
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST']=str(a.direct_freeze_manifest.resolve())
    os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE']='1'
    sys.argv=[str(engine.ENGINE),*remaining,'--dex3-hard-limit-contract',str(engine.AUTHORITATIVE_JOINT_CONTRACT),
        '--dex3-hard-limit-inset-rad',engine.DEX3_JOINT_STOP_INSET_RAD]
    ns=dict(__name__='__main__',__file__=str(engine.ENGINE),__package__=None)
    exec(compile(source,str(engine.ENGINE),'exec'),ns,ns)


if __name__=='__main__':main()
