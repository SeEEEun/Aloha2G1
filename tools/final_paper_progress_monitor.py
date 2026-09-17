#!/usr/bin/env python3
"""Persistent major-stage/episode log; never infer missing outcomes."""
from pathlib import Path
import datetime,hashlib,json,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/final_single_variable_ab';D=OUT/'paper_completion_v1';MASTER=OUT/'master_autonomous'
def record(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def atomic(p,value):
    temp=p.with_suffix(p.suffix+'.progress-incomplete');temp.write_text(value);temp.replace(p)
def main():
    seen_path=D/'PROGRESS_MONITOR_INDEX.json';seen=set(json.loads(seen_path.read_text())) if seen_path.exists() else set()
    while not (D/'FINAL_ARTIFACT_VERIFICATION.json').exists():
        paths=list((D/'position').glob('*/RESULT.json'))+list((D/'full6d').glob('*/RESULT.json'))+list((D/'complete_action').glob('*/RESULT.json'))+list((D/'reference_physics').glob('*/RESULT.json'))
        paths.extend(p for p in (D/'dex3/COMMON_DEX3_QUALIFICATION.json',D/'dex3/SOURCE_CLOCK_QUALIFICATION.json',D/'REFERENCE_PHYSICS_COMPLETE.json',D/'ACT_BRANCH_RESULT.json') if p.exists())
        for p in sorted(paths):
            identity=str(p)+':'+record(p)['sha256']
            if identity in seen:continue
            r=json.loads(p.read_text());status=r.get('outcome',r.get('status','COMPLETE'))
            log=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=p.parent.parent.name,status=status,
                authoritative_inputs=[str(OUT/'03_common_execution_freeze/COMMON_RETARGETING_FREEZE_MANIFEST.json')],files_hashes_used=[record(p)],
                observed_problem=None if status in ('POSITION_EXECUTABLE','FULL6D_EXECUTABLE','PASS','PHYSICAL_SUCCESS') else status,
                root_cause_classification=status,action_taken='Persist outcome; no scientific failure rescue. Continue next episode/stage.',retries=0,artifacts_created=[record(p)],next_stage='REMAINING_COHORT_AND_SUPPORTED_PAPER_ARTIFACTS')
            with (D/'MASTER_RUN_LOG.jsonl').open('a') as stream:stream.write(json.dumps(log,ensure_ascii=False)+'\n')
            with (MASTER/'MASTER_RUN_LOG.jsonl').open('a') as stream:stream.write(json.dumps(log,ensure_ascii=False)+'\n')
            seen.add(identity)
        atomic(seen_path,json.dumps(sorted(seen)))
        count=len(list((D/'position').glob('*/RESULT.json')));six=len(list((D/'full6d').glob('*/RESULT.json')));physical=len(list((D/'reference_physics').glob('*/RESULT.json')))
        text=f'# Current paper-completion stage\n\nPosition outcomes persisted: {count}/150. 6D stage outcomes persisted: {six}/150. Reference DEV35 outcomes persisted: {physical}/70.\n\nScientific failures remain data. Missing cases are pending, not counted as failures. Fixed numerical budget unchanged.\n'
        atomic(D/'CURRENT_STAGE.md',text)
        atomic(MASTER/'CURRENT_STAGE.md',text+'\nAuthoritative run: '+str(D)+'\n')
        atomic(MASTER/'CHECKPOINT_STATE.json',json.dumps(dict(run=str(D),position=count,full6d=six,reference_physics=physical,status='RUNNING_FROZEN_PAPER_COMPLETION',scientific_failures_are_results=True)))
        message=f'위치 결과 {count}/150, 6D 단계 결과 {six}/150, 참조 DEV35 결과 {physical}/70를 저장했습니다. 미완료 사례는 실패로 세지 않습니다. 동결 예산은 변경하지 않으며 과학적 실패를 보존하고 다음 단계와 논문 산출물을 계속 처리합니다.'
        atomic(D/'CHATGPT_UPDATE.md',message+'\n');atomic(MASTER/'CHATGPT_UPDATE.md',message+'\n');print(message,flush=True)
        time.sleep(30)

if __name__=='__main__':main()
