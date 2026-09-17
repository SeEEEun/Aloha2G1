#!/usr/bin/env python3
"""Same contact-constrained engine, source-clock controller adapter only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_direct_physical_execution_isaac as engine
original=engine.instrument
def instrument(source):
    result,counts=original(source)
    old='from tools.direct_physical_execution_isaac_runtime import build_runtime'
    assert result.count(old)==1
    result=result.replace(old,'from tools.final_paper_source_clock_dex3 import build_runtime')
    # This is reference/policy-agnostic execution; provenance is in the outer
    # invocation manifest, not an invented ACT checkpoint in the engine report.
    result=result.replace('"policy_or_checkpoint_used": True,','"policy_or_checkpoint_used": False,')
    counts['source_clock_adapter']=1
    return result,counts
engine.instrument=instrument
if __name__=='__main__':raise SystemExit(engine.main())
