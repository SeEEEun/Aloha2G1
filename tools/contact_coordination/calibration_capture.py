"""Read-only full-articulation and cooked-collider capture over common controls."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.contact_coordination import physics_capture
from tools import run_direct_physical_execution_isaac as engine


def capture_cooked(stage, folder):
    from pxr import UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    from tools.contact_coordination.io import atomic_json
    import numpy as np

    cooking = get_physx_cooking_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cache = UsdGeom.XformCache()
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        result = {}
        def received(status, convexes):
            result.update(status=str(status), convexes=[dict(
                vertices=[[float(v.x), float(v.y), float(v.z)] for v in c.vertices],
                indices=list(c.indices),
                polygons=[dict(index_base=p.index_base, num_vertices=p.num_vertices,
                               plane=[float(x) for x in p.plane]) for p in c.polygons])
                for c in convexes])
        cooking.request_convex_collision_representation(
            stage_id=stage_id,
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
            run_asynchronously=False, on_result=received)
        parent = prim
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        world = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        body = np.asarray(cache.GetLocalToWorldTransform(parent)).T if parent else np.eye(4)
        rows.append(dict(path=str(prim.GetPath()), body=str(parent.GetPath()) if parent else None,
                         T_world_mesh=world, T_body_mesh=np.linalg.inv(body) @ world,
                         authored_points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()),
                         **result))
    atomic_json(Path(folder) / 'PHYSX_COOKED_COLLIDERS.json', dict(
        read_only=True, transform_convention='T_AB maps B into A; column vectors',
        api='PhysXCooking.request_convex_collision_representation; synchronous',
        runtime_shapes_or_filters_changed=False, colliders=rows))


def instrument(source):
    result, counts = physics_capture.instrument(source)
    substitutions = [
        ('            "measured_q_rad",\n',
         '            "measured_q_rad",\n            "all_measured_q_rad",\n            "body_position_world_m",\n            "body_quaternion_xyzw",\n'),
        ('                "measured_q_rad": measured,\n',
         '                "measured_q_rad": measured,\n'
         '                "all_measured_q_rad": numpy(robot.data.joint_pos)[0].astype(np.float64),\n'
         '                "body_position_world_m": numpy(robot.data.body_pos_w)[0].astype(np.float64),\n'
         '                "body_quaternion_xyzw": numpy(robot.data.body_quat_w)[0].astype(np.float64),\n'),
        ('    arrays["joint_names"] = np.asarray(names)\n',
         '    arrays["joint_names"] = np.asarray(names)\n'
         '    arrays["all_joint_names"] = np.asarray(isaac_names)\n'
         '    arrays["body_names"] = np.asarray(body_names)\n'),
        ('    body_names = list(robot.data.body_names)\n',
         '    body_names = list(robot.data.body_names)\n'
         '    from tools.contact_coordination.calibration_capture import capture_cooked\n'
         '    capture_cooked(stage, output_dir)\n'),
    ]
    for old, new in substitutions:
        assert result.count(old) == 1, old
        result = result.replace(old, new)
    compile(result, str(engine.ENGINE), 'exec')
    counts.update(full_articulation_read_only=1, cooked_geometry_read_only=1)
    return result, counts


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--direct-freeze-manifest', type=Path, required=True)
    p.add_argument('--qualification-mode', action='store_true')
    p.add_argument('--validate-patch-only', action='store_true')
    args, remaining = p.parse_known_args()
    source, counts = instrument(engine.ENGINE.read_text())
    if args.validate_patch_only:
        print(counts)
        return
    os.environ['DIRECT_EVAL35_FREEZE_MANIFEST'] = str(args.direct_freeze_manifest.resolve())
    if args.qualification_mode:
        os.environ['DIRECT_EXECUTION_QUALIFICATION_MODE'] = '1'
    sys.argv = [str(engine.ENGINE), *remaining, '--dex3-hard-limit-contract',
                str(engine.AUTHORITATIVE_JOINT_CONTRACT), '--dex3-hard-limit-inset-rad',
                engine.DEX3_JOINT_STOP_INSET_RAD]
    ns = dict(__name__='__main__', __file__=str(engine.ENGINE), __package__=None)
    exec(compile(source, str(engine.ENGINE), 'exec'), ns, ns)


if __name__ == '__main__':
    main()
