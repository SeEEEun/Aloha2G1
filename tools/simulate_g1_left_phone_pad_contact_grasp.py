#!/usr/bin/env python3
"""Triangle-patch-aware rerun of the fixed Dex3 horizontal phone grasp."""
from pathlib import Path
import sys
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path.insert(0,str(ROOT/"tools"))
import simulate_g1_left_phone_contact_aware_grasp as impl

def render_calibration():
 import json,mujoco,numpy as np
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 cal=json.loads((impl.PAD_CAL_OUT/"dex3_pad_contact_calibration.json").read_text())
 model=mujoco.MjModel.from_xml_path(str(impl.old.G1_XML))
 labels={"thumb":"left_hand_thumb_2_link","index0":"left_hand_index_0_link","index1":"left_hand_index_1_link"}
 meshes=[]
 for label,body in labels.items():
  gid=impl.ref.collision_geoms(model,body)[-1];mid=int(model.geom_dataid[gid])
  va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid]);fa=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid])
  v=np.asarray(model.mesh_vert[va:va+vn]);f=np.asarray(model.mesh_face[fa:fa+fn]);allowed=set(cal["patches"][label]["allowed_inner_pad_triangle_indices"])
  fig=plt.figure(figsize=(7,7));ax=fig.add_subplot(projection="3d")
  poly=Poly3DCollection(v[f],facecolors=["limegreen" if i in allowed else "lightgray" for i in range(fn)],
   edgecolors="none",alpha=.85);ax.add_collection3d(poly);lo=v.min(0);hi=v.max(0)
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=f"{body}: green=allowed inner-pad triangles")
  fig.tight_layout();fig.savefig(impl.PAD_CAL_OUT/f"{label if label=='thumb' else label+'_inner'}_pad_triangles.png",dpi=180);plt.close(fig)
  meshes.append((label,fn,len(allowed)))
 return meshes

def main():
 impl.PAD_MODE=True
 impl.OUT=ROOT/"evaluation/g1_left_phone_pad_contact_grasp"
 impl.PRIM=ROOT/"converted_runs/g1_left_phone_pad_contact_grasp"
 rc=impl.main()
 renames={
  "contact_aware_grasp_report.json":"pad_contact_grasp_report.json",
  "controller_candidate_comparison.csv":"candidate_comparison.csv",
  "actuator_trajectory.csv":"contact_patch_trajectory.csv",
 }
 for src,dst in renames.items():
  p=impl.OUT/src
  if p.exists():p.replace(impl.OUT/dst)
 render_calibration()
 return rc
if __name__=="__main__":
 if "--render-only" in sys.argv:
  print(render_calibration())
 else:raise SystemExit(main())
