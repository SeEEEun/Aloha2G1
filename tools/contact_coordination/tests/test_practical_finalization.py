import tempfile,unittest
from pathlib import Path
from tools.contact_coordination.io import read,record,atomic_json,atomic_text
from tools.contact_coordination.practical_finalize import final_selection,verify_freeze
from tools.contact_coordination.practical_paper import plan_case


class PracticalFinalizationTests(unittest.TestCase):
    def test_insufficient_report_uses_measured_failure_counts_and_stops(self):
        from .test_practical_study_runner import PracticalRunnerTests
        from tools.contact_coordination.practical_diagnostics import run
        from tools.contact_coordination.practical_planning import rank
        from tools.contact_coordination.practical_parameters import RECIPES
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);PracticalRunnerTests().fixture(out,[(8,8)]*4)
            sweep=read(out/'PLANNING_SWEEP.json');sweep['status']='PASS';atomic_json(out/'PLANNING_SWEEP.json',sweep)
            atomic_json(out/'CALIBRATION_SEARCH_SPACE.json',dict(recipes=RECIPES))
            rank(out);run(out)
            value=read(out/'PLANNING_CALIBRATION_CAUSAL_DIAGNOSTICS.json')
            self.assertEqual(value['aggregate_causes'][0],dict(cause='NO_IK',phase='APPROACH_CLEARANCE',count=256))
            text=(out/'PLANNING_CALIBRATION_INSUFFICIENT.md').read_text()
            self.assertIn('256 of the 320',text);self.assertIn('FINAL PAPER A/B: NOT RUN',text)

    def test_final_selection_uses_all_predeclared_sources_not_execution_only(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);ids=['s0','s1','s2'];names=['FULL_COVERAGE','ONE_EXECUTION']
            atomic_json(out/'PRACTICAL_STUDY.json',dict(physical_source_ids=ids))
            atomic_json(out/'PHYSICAL_SUBSET_PREDECLARATION.json',dict(source_ids=ids))
            ranked=[];rows=[]
            for index,name in enumerate(names):
                area=out/'recipes'/name
                for filename in ('CALIBRATION_PARAMETERS.json','PRACTICAL_PARAMETERS.json'):
                    atomic_json(area/filename,dict(fixture=name))
                ranked.append(dict(recipe=name,score=.5,source_deviation=.01,planning_seconds=10.,complexity=index))
                for method in ('B_INDEPENDENT','C_COUPLED'):
                    for i,sid in enumerate(ids):
                        # 2/3 beats 1/3 despite the latter's perfect conditional rate.
                        success=i<(2 if index==0 else 1)
                        rows.append(dict(recipe=name,source_id=sid,method_key=method,physics_executed=index==0 or success,
                            folder=str(area/('fixture_trace_'+sid+'_'+method)),
                            stages=dict(FULL_TASK=success,HANDOFF=success,RIGHT_OWNERSHIP=success)))
            atomic_json(out/'RECIPE_RANKING.json',dict(top_recipes=names,ranked=ranked))
            atomic_json(out/'TOP_RECIPE_PHYSICS.json',dict(status='PASS',rows=rows))
            final_selection(out)
            result=read(out/'TRAIN40_FINAL_SHARED_CONFIG.json')
            self.assertEqual(result['recipe'],'FULL_COVERAGE')
            self.assertEqual(result['selection'][0]['scheduled'],6)

    def test_freeze_rejects_changed_controller_or_parameter_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);parameter=out/'parameter.json';atomic_json(parameter,dict(value=1))
            atomic_json(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json',dict(files=[record(parameter)]))
            verify_freeze(out);atomic_json(parameter,dict(value=2))
            with self.assertRaisesRegex(RuntimeError,'FROZEN_FINAL_DEPENDENCY_CHANGED'):verify_freeze(out)

    def test_paper_case_cache_cannot_cross_a_freeze_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);manifest=out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'
            artifact=out/'planned_path';atomic_text(artifact,'original geometry')
            atomic_json(manifest,dict(hash='first'))
            row=dict(source_id='s0',method='C_COUPLED',frozen_manifest=record(manifest),artifacts=[record(artifact)])
            atomic_json(out/'FINAL_FROZEN/paper_planning_cases/s0_C_COUPLED.json',row)
            self.assertEqual(plan_case(out,'s0','C_COUPLED'),row)
            atomic_json(manifest,dict(hash='changed'))
            with self.assertRaisesRegex(RuntimeError,'FROZEN_PAPER_CASE_MANIFEST_MISMATCH'):plan_case(out,'s0','C_COUPLED')


if __name__=='__main__':unittest.main()
