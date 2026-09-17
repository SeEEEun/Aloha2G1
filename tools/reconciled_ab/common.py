from pathlib import Path
import csv, io, json, hashlib, datetime, os

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs/final_single_variable_ab'
PRIOR = OLD / 'paper_completion_v1'
RUN = ROOT / 'outputs/ab_physical_evaluation_reconciled/20260907T020915Z'

def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def record(p):
    p=Path(p).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def text(p,value):
    p=Path(p); assert p.is_relative_to(RUN), p
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.incomplete'); tmp.write_text(value); os.replace(tmp,p)
def save(p,value): text(p,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvsave(p,rows):
    buf=io.StringIO(); fields=list(dict.fromkeys(k for row in rows for k in row))
    w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows(rows);text(p,buf.getvalue())
def log(stage,status,action,inputs=(),artifacts=(),next_stage='',korean=''):
    entry=dict(timestamp=now(),stage=stage,status=status,action=action,
               authoritative_inputs=[record(p) for p in inputs],artifacts=[record(p) for p in artifacts],next_stage=next_stage)
    p=RUN/'RUN_LOG.jsonl'; text(p,(p.read_text() if p.exists() else '')+json.dumps(entry)+'\n')
    text(RUN/'CURRENT_STATUS.md',f'# Reconciled A/B evaluation\n\n{stage}: {status}\n\n{action}\n\nNext: {next_stage}\n')
    if korean: text(RUN/'CHATGPT_UPDATE.md',korean+'\n');print(korean,flush=True)

if __name__=='__main__':
    assert not (RUN/'RUN_LOG.jsonl').exists(), 'Do not reset an existing run'
    text(RUN/'DECISIONS.md','# Versioned decisions\n\nHistorical experiments are immutable provenance. This experiment separates fidelity, physical execution validity, and measured task outcomes. No solver or physics process was active at handoff. Raw targets, registration, source timing, natural q0, and the 0.700 s prefix are preserved. No new episode-specific recovery is authorized.\n\nPredeclared reconciliation sample: TRAIN recording indices 0, 10, 24 for both methods; all compatible saved-array checks will then be expanded. Numeric indices are resolved to recording identities through manifests.\n')
    text(RUN/'FINAL_REPORT.md','# Reconciled A/B evaluation — work in progress\n\nREPORT_PACKAGE_STATUS: INCOMPLETE\n\nPHYSICAL_EXPERIMENT_STATUS: NOT_EXECUTED\n\nHistorical reports are not new physical observations. Audit and component checks are in progress.\n')
    log('READ_ONLY_LINEAGE_RECONCILIATION','IN_PROGRESS','Recover actual tensors, compatible contracts, all candidate rejection gates and actual 6D invocations.',
        [OLD/'FINAL_PAPER_READY_SINGLE_VARIABLE_AB_REPORT.md',PRIOR/'CASE_MANIFEST.json'],next_stage='KNOWN_ANSWER_6D_TESTS',
        korean='이전 결과를 보존하고 새 버전에서 계보·판정 기준을 대조합니다. 정확도, 실행 유효성, 실제 물리 결과를 분리하며 아직 새 DEV35 물리 실행은 없습니다.')
