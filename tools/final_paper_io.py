"""Standard-library paper artifact I/O; no simulator or solver dependencies."""
from pathlib import Path
import datetime,hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/final_single_variable_ab'
DEST=OUT/'paper_completion_v1'
def read(path):return json.loads(Path(path).read_text())
def file_record(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size)
def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.incomplete')
    tmp.write_text(text);os.replace(tmp,path)
def atomic_json(path,value):atomic_text(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def cases():return read(DEST/'CASE_MANIFEST.json')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
