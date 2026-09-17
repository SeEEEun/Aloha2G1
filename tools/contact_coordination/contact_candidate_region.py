"""Small object-frame sampling of the existing calibrated contact span.

These are planning candidates, not independently qualified physical grasps.
No source ID, world pose, successful command, or outcome is consumed here.
"""
import itertools
import numpy as np


def translation_bank(calibrated_offset):
    offset=np.asarray(calibrated_offset,float)
    radius=float(np.linalg.norm(offset))
    if offset.shape!=(3,) or not np.isfinite(offset).all() or radius<=0:
        raise ValueError('Finite nonzero calibrated contact extent required')
    values=[dict(offset=offset*f,priority=0,family='calibrated_clearance_direction',fraction=f)
            for f in (0.,.5,1.)]
    # Face normals and face diagonals of the modeled object's coordinate box.
    # The magnitude is unchanged; no world-space reach projection is involved.
    for components in itertools.product((-1,0,1),repeat=3):
        if not 1<=np.count_nonzero(components)<=2:continue
        direction=np.asarray(components,float);direction/=np.linalg.norm(direction)
        proposed=radius*direction
        if any(np.allclose(proposed,v['offset'],atol=1e-12,rtol=0) for v in values):continue
        values.append(dict(offset=proposed,priority=1,family='object_geometry_direction',direction=components))
    return values
