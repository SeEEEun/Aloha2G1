#!/usr/bin/env python3
"""Create the Doll-Handoff task overlay in a copied Isaac Lab MagSafe scene.

Run through Isaac Lab Python, e.g.:
  /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p build_doll_handoff_overlay.py --apply

The script never edits the original isaaclab_magsafe_fixed_scene.  It is intended
for the copied isaaclab_doll_handoff_scene directory created by install.sh.
"""
from __future__ import annotations
import argparse, json, math, shutil, sys
from pathlib import Path


def _material(stage, path, color):
    from pxr import UsdShade, Sdf
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def _mesh(stage, path, verts, faces, mat=None, collision=False):
    from pxr import UsdGeom, UsdPhysics, Vt, Gf
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    counts = [len(f) for f in faces]
    indices = [i for f in faces for i in f]
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    if mat is not None:
        _bind(mesh.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def _torus_mesh(R, r, n_major=72, n_minor=8, rotate_x=0.0):
    verts=[]; faces=[]
    cx=math.cos(rotate_x); sx=math.sin(rotate_x)
    for i in range(n_major):
        a=2*math.pi*i/n_major
        ca,sa=math.cos(a),math.sin(a)
        for j in range(n_minor):
            b=2*math.pi*j/n_minor
            cb,sb=math.cos(b),math.sin(b)
            x=(R+r*cb)*ca; y=(R+r*cb)*sa; z=r*sb
            # rotate around X
            yy=cx*y-sx*z; zz=sx*y+cx*z
            verts.append((x,yy,zz))
    for i in range(n_major):
        ni=(i+1)%n_major
        for j in range(n_minor):
            nj=(j+1)%n_minor
            a=i*n_minor+j; b=ni*n_minor+j; c=ni*n_minor+nj; d=i*n_minor+nj
            faces += [(a,b,c),(a,c,d)]
    return verts,faces


def _wall_geometry(side, topx,topy,botx,boty,h,t):
    in_topx=topx-2*t; in_topy=topy-2*t; in_botx=botx-2*t; in_boty=boty-2*t
    if side in ("front","back"):
        sy=-1 if side=="front" else 1
        v=[(-botx/2,sy*boty/2,0),(botx/2,sy*boty/2,0),(-topx/2,sy*topy/2,h),(topx/2,sy*topy/2,h),
           (-in_botx/2,sy*in_boty/2,0),(in_botx/2,sy*in_boty/2,0),(-in_topx/2,sy*in_topy/2,h),(in_topx/2,sy*in_topy/2,h)]
    else:
        sx=-1 if side=="left" else 1
        v=[(sx*botx/2,-boty/2,0),(sx*botx/2,boty/2,0),(sx*topx/2,-topy/2,h),(sx*topx/2,topy/2,h),
           (sx*in_botx/2,-in_boty/2,0),(sx*in_botx/2,in_boty/2,0),(sx*in_topx/2,-in_topy/2,h),(sx*in_topx/2,in_topy/2,h)]
    f=[(0,1,3),(0,3,2),(4,7,5),(4,6,7),(2,3,7),(2,7,6),(0,4,5),(0,5,1),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    return v,f


def _deactivate_old_task(stage):
    exact={"phone","accessory","charger","magsafephone","magsafeaccessory","magsafecharger","chargerbase","chargerpad"}
    deactivated=[]
    # Traverse snapshot because SetActive changes traversal.
    prims=list(stage.Traverse())
    # Prefer highest matching parents so children follow automatically.
    matches=[]
    for prim in prims:
        name=prim.GetName().lower()
        if name in exact or name.startswith("phone_") or name.startswith("accessory_") or name.startswith("charger_"):
            p=str(prim.GetPath())
            if p.startswith("/World/"):
                matches.append(prim)
    # Remove descendants when an ancestor is already matched.
    paths=sorted([str(p.GetPath()) for p in matches], key=lambda s:(s.count('/'),len(s)))
    keep=[]
    for p in paths:
        if not any(p.startswith(k + "/") for k in keep):
            keep.append(p)
    for p in keep:
        prim=stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            prim.SetActive(False); deactivated.append(p)
    return deactivated


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--layout", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    args=ap.parse_args()
    scene=args.scene_dir.expanduser().resolve()
    layout_path=(args.layout or (scene/"scene_layout_doll_handoff.json")).resolve()
    cfg=json.loads(layout_path.read_text(encoding="utf-8"))
    base=scene/cfg["source_scene"]["base_usd_relative"]
    if not base.is_file():
        print(f"ERROR missing base USD: {base}", file=sys.stderr); return 2
    if not args.apply:
        print("DRY RUN")
        print(f"base USD: {base}")
        print(f"doll world XY: {cfg['doll']['world_xy']}")
        print(f"bin world XY: {cfg['trash_bin']['world_xy']}")
        print("Re-run with --apply")
        return 0

    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage=Usd.Stage.Open(str(base))
    if stage is None:
        print(f"ERROR cannot open USD: {base}", file=sys.stderr); return 3
    deactivated=_deactivate_old_task(stage)
    root_path="/World/DollHandoffTask"
    old=stage.GetPrimAtPath(root_path)
    if old and old.IsValid():
        stage.RemovePrim(root_path)
    root=UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetCustomDataByKey("task", "doll_handoff_to_trash_bin")
    root.GetPrim().SetCustomDataByKey("layoutFile", str(layout_path))

    green=_material(stage, root_path+"/Looks/Green", (0.55,0.78,0.08))
    white=_material(stage, root_path+"/Looks/White", (0.92,0.92,0.86))
    binmat=_material(stage, root_path+"/Looks/Bin", (0.78,0.80,0.70))

    # Doll/ball
    d=cfg["doll"]; radius=float(d["diameter"])/2; table_z=float(cfg["table"]["surface_height"])
    dx,dy=map(float,d["world_xy"])
    doll_xf=UsdGeom.Xform.Define(stage, root_path+"/Doll")
    doll_xf.AddTranslateOp().Set(Gf.Vec3d(dx,dy,table_z+radius+0.001))
    sph=UsdGeom.Sphere.Define(stage, root_path+"/Doll/Ball")
    sph.CreateRadiusAttr(radius); _bind(sph.GetPrim(), green)
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sph.GetPrim())
    UsdPhysics.MassAPI.Apply(sph.GetPrim()).CreateMassAttr(float(d["mass_kg"]))
    # Decorative seams, no collision.
    for name,rx in (("SeamA",0.0),("SeamB",math.pi/2)):
        v,f=_torus_mesh(radius*0.985, 0.0012, rotate_x=rx)
        _mesh(stage, root_path+f"/Doll/{name}", v,f,white,False)

    # Bin, static open container
    b=cfg["trash_bin"]; bx,by=map(float,b["bottom_outer_size_xy"]); tx,ty=map(float,b["top_outer_size_xy"])
    h=float(b["height"]); t=float(b["wall_thickness"]); bt=float(b["bottom_thickness"]); cx,cy=map(float,b["world_xy"])
    bin_xf=UsdGeom.Xform.Define(stage, root_path+"/TrashBin")
    bin_xf.AddTranslateOp().Set(Gf.Vec3d(cx,cy,table_z))
    for side in ("front","back","left","right"):
        v,f=_wall_geometry(side,tx,ty,bx,by,h,t)
        _mesh(stage, root_path+f"/TrashBin/{side.capitalize()}Wall", v,f,binmat,True)
    bottom=UsdGeom.Cube.Define(stage, root_path+"/TrashBin/Bottom")
    bottom.CreateSizeAttr(1.0)
    bottom.AddScaleOp().Set(Gf.Vec3f(bx,by,bt))
    bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,bt/2))
    _bind(bottom.GetPrim(), binmat); UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())

    named=scene/"generated"/"doll_handoff_scene_v1.usda"
    backup=scene/"generated"/"magsafe_magnetic_scene_v2.before_doll_handoff.usda"
    if not backup.exists(): shutil.copy2(base, backup)
    stage.GetRootLayer().Export(str(named))
    shutil.copy2(named, base)
    report={"base_usd":str(base),"named_usd":str(named),"backup":str(backup),"deactivated_old_task_prims":deactivated,
            "doll_world_xy":d["world_xy"],"bin_world_xy":b["world_xy"],"table_z":table_z}
    (scene/"doll_handoff_build_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("PASS: DOLL_HANDOFF_USD_BUILT")
    print(json.dumps(report, indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
