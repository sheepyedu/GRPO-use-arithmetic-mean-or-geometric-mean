import os,json,hashlib,datetime,subprocess
from pathlib import Path
root=Path('/mount-data/GRPO-Sequence-Ratio')
run=root/'runs/seed71'
source=root/'source-v4'
out=run/'acceptance-v1'
original=root/'runs/seed43/acceptance-v1/verify.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='8383ff6875fbe90db1a1ebeac25b5f2a781eeffad6c5f0c7cb1d144b94349214'
for env in ['math','mixed']:
 assert json.loads((run/f'formal-{env}-exit.json').read_text())['codes']==[0,0]
assert json.loads((root/'logs/seed71-run-v4-exit.json').read_text())['code']==0
for pid in [3623275,3623277,3623282,3623284,3623283,3623285]:
 assert not Path(f'/proc/{pid}').exists(),pid
out.mkdir()
text=original.read_text().replace('/runs/seed43','/runs/seed71')
with (out/'verify.py').open('x') as f:f.write(text)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(source),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
with (out/'LOG.txt').open('xb') as f:
 p=subprocess.Popen([str(root/'env/bin/python'),'-u',str(out/'verify.py')],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
proc=Path(f'/proc/{p.pid}')
raw=(proc/'cmdline').read_bytes()
launch={'pid':p.pid,'start_ticks':int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),'uid':proc.stat().st_uid,'cmdline_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'original_source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'output':str(out),'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cuda_visible_devices':''}
with (out/'LAUNCH.json').open('x') as f:json.dump(launch,f,indent=2)
print(json.dumps(launch))
