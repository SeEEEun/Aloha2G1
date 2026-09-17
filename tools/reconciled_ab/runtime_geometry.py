"""Read-only USD collision instrumentation; never alters physics geometry."""
import json
from pathlib import Path

def capture(stage,folder):
    from pxr import UsdPhysics,PhysxSchema,UsdGeom
    rows=[];articulations=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col=UsdPhysics.CollisionAPI(prim);mesh=UsdPhysics.MeshCollisionAPI(prim)
            rows.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),enabled=col.GetCollisionEnabledAttr().Get(),
                approximation=mesh.GetApproximationAttr().Get() if mesh else None,
                filtered_pairs=[str(x) for x in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()] if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else [],
                physics_attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:','physx'))}))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api=PhysxSchema.PhysxArticulationAPI(prim)
            articulations.append(dict(path=str(prim.GetPath()),self_collisions_enabled=api.GetEnabledSelfCollisionsAttr().Get(),solver_position=api.GetSolverPositionIterationCountAttr().Get(),solver_velocity=api.GetSolverVelocityIterationCountAttr().Get()))
    layers=[x.realPath for x in stage.GetUsedLayers() if x.realPath]
    out=dict(colliders=rows,articulations=articulations,usd_layers=layers,read_only=True,
        caveat='Runtime collider approximation and filtering are independent of offline detailed modeled-surface classification. Visual open shells are not certified solids. No collider or pair changed by this audit.')
    p=Path(folder)/'RUNTIME_COLLISION_MODEL.json';p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

