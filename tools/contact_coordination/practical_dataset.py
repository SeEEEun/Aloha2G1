"""Package actual source/selection/geometry evidence with converted episodes."""
from pathlib import Path
import shutil
from .io import read,record,atomic_json


def export_evidence(raw,folder,source):
    folder=Path(folder);source=Path(source);plan=raw['planning']['plan'];sid=raw['source_id']
    assert read(source/'PHASE_RECORD.json')['source_id']==sid
    copied=[]
    def copy_file(src,dst,required=True):
        src=Path(src);dst=folder/dst
        if not src.exists():
            if required:raise FileNotFoundError(src)
            return
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        original=record(src);target=record(dst);assert original['sha256']==target['sha256']
        copied.append(dict(original=original,packaged=target))
    for p in sorted(source.iterdir()):
        if p.suffix in ('.json','.npz'):copy_file(p,Path('source')/p.name)
    context=Path(plan['context']);acquisition=context/'prototype'/sid/'morphology_acquisition_v4'
    for name in ('GOALS.json','PLAN_RESULT.json','PHASE_Q.npz'):
        copy_file(acquisition/name,Path('planning/acquisition')/name,required=plan['full_task_plan'])
    if plan.get('receipt'):copy_file(plan['receipt'],'selection/CHAIN_SELECTION.json')
    for item in plan.get('artifacts',[]):
        if Path(item['path']).name=='GRASP_CANDIDATES.json':
            assert record(item['path'])==item
            copy_file(item['path'],'selection/GRASP_CANDIDATES.json')
    for name in ('CALIBRATION_PARAMETERS.json','PRACTICAL_PARAMETERS.json','INPUT_CONTEXT.json'):
        copy_file(context/name,Path('configuration')/name,required=False)
    if plan['full_task_plan']:
        physical_plan=Path(plan['plan']);header=read(physical_plan/'PLAN.json')
        assert not header.get('diagnostic') and header['source_id']==sid
        reference=header['connection'];assert record(reference['path'])==reference
        connection=Path(reference['path']).parent;result=read(reference['path'])
        # This is the exact branch used by the unchanged command exporter.
        chosen=next(r for r in result['results'] if r['all_admissible'])
        sub=connection/chosen.get('subdirectory',f'seed_{chosen["candidate_seed"]}')
        copy_file(reference['path'],'planning/CONNECTION_RESULT.json')
        for name in ('GOALS.json','PHASE_IK.json','PHASE_Q.npz'):
            copy_file(sub/name,Path('planning/transport')/name)
        copy_file(connection/'SELECTED_CONTACTS.json','selection/SELECTED_CONTACTS.json')
    physical=raw['physical']
    if physical['physics_executed']:
        copy_file(Path(physical['folder'])/'input/INCIDENTAL_CONTACT_POLICY.json','configuration/INCIDENTAL_CONTACT_POLICY.json')
    value=dict(source_id=sid,paper_method=raw['paper_method'],internal_method=raw['internal_method'],files=copied,
        complete_plan=bool(plan['full_task_plan']),diagnostic_reconstruction=False,
        geometry_storage='Original phase GOALS.json, endpoint PHASE_Q.npz and PLAN_RESULT/PHASE_IK connecting_q/planner certificates; distinct from retimed COMMANDS.npz',
        no_plan_evidence='Source and available failed candidate/phase records retained; missing phases are not reconstructed')
    path=folder/'PACKAGED_PLANNING_EVIDENCE.json';atomic_json(path,value);return path


def verify_dataset(manifest_path):
    manifest=read(manifest_path);ids=manifest['source_ids'];seen=[];count=0
    for item in manifest['episode_manifests']:
        assert record(item['path'])==item
        episode=read(item['path']);sid=episode['result']['source_id'];seen.append(sid)
        assert episode['result']['paper_method']==manifest['paper_method']
        assert episode['diagnostic_reconstruction'] is False
        for artifact in episode['artifacts']:
            assert record(artifact['path'])==artifact;count+=1
        reference=episode['packaged_planning_evidence'];assert record(reference['path'])==reference
        package=read(reference['path']);assert package['source_id']==sid and not package['diagnostic_reconstruction']
        for row in package['files']:
            assert record(row['packaged']['path'])==row['packaged'];count+=1
    assert seen==ids and len(set(seen))==len(ids)
    return dict(paper_method=manifest['paper_method'],source_count=len(ids),verified_payload_files=count,
        manifest=record(manifest_path),no_diagnostic_reconstruction=True)
