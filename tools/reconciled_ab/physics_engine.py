"""Unchanged source-clock PhysX engine plus read-only collision provenance."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools import final_paper_physics_isaac as source_clock
original=source_clock.engine.instrument
def instrument(source):
    result,counts=original(source)
    anchor='    dt = float(config["timing"]["physics_dt_s"])\n'
    assert result.count(anchor)==1
    result=result.replace(anchor,'    from tools.reconciled_ab.runtime_geometry import capture\n    capture(stage, output_dir)\n'+anchor)
    counts['read_only_runtime_geometry_capture']=1
    return result,counts
source_clock.engine.instrument=instrument
if __name__=='__main__':raise SystemExit(source_clock.engine.main())

