"""Early rejection certificates for trial seeds; never an acceptance shortcut.

A strict interior vertex in a reliable closed detailed solid proves overlap.
Trials with no such witness still require the original full classifier.
The original final geometry rule and geometry tolerance remain unchanged.
"""
import numpy as np
from tools.common_geometry_confirmed_collision import _inside_depth

def sampled_vertices(mesh):
    # This finite subset can only provide a rejection witness. A missing
    # witness must still fall through to the unchanged complete classifier.
    ids=np.linspace(0,len(mesh.vertices)-1,min(64,len(mesh.vertices)),dtype=int)
    return mesh.vertices[ids]

def hard_witness(collision,q,hands):
    records=collision.proxy._records(q,*hands);g=collision.g1
    for r in records:
        pair=tuple(r['geom_pair']);a,b=[int(g.model.geom_bodyid[x]) for x in pair]
        ma,pa,ta=collision.body_geometry(a);mb,pb,tb=collision.body_geometry(b)
        relative=np.linalg.inv(ta)@tb
        second=[(p.copy().apply_transform(relative),reliable) for p,reliable in pb if reliable]
        for x,rx in pa:
            if not rx:continue
            for y,ry in second:
                gap=np.maximum(np.maximum(x.bounds[0]-y.bounds[1],y.bounds[0]-x.bounds[1]),0)
                if np.linalg.norm(gap)>collision.tolerance:continue
                depth=max(_inside_depth(x,sampled_vertices(y),collision.tolerance),_inside_depth(y,sampled_vertices(x),collision.tolerance))
                if depth>collision.tolerance:
                    return dict(classification='HARD_SELF_COLLISION',geom_pair=list(pair),body_pair=r['body_pair'],
                        detailed_penetration_lower_bound_mm=1000*depth,proxy_penetration_mm=1000*r['penetration_depth_m'],
                        detailed_separation_mm=None,separation_status='NOT_NEEDED_FOR_STRICT_INTERIOR_REJECTION_CERTIFICATE',
                        acceptance_shortcut=False,certificate='Strict detailed-solid interior witness; both shells reliable closed solids')
    return None

