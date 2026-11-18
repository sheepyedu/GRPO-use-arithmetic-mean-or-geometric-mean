import os,json,hashlib,datetime,subprocess
from pathlib import Path
root=Path('/mount-data/GRPO-Sequence-Ratio')
run=root/'runs/seed101'
source=root/'source-v4'
out=run/'acceptance-v1'
original=root/'runs/seed71/acceptance-v1/verify.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='210cc2af4f11b070e1c76f3e1108e7de2007908344cc9277c986fa8bf45eeb6d'
for env in ['math','mixed']:
 assert json.loads((run/f'formal-{env}-exit.json').read_text())['codes']==[0,0]
assert json.loads((root/'logs/seed101-run-v4-exit.json').read_text())['code']==0
for pid in [3924228, 3924230, 3924234, 3924236, 3924235, 3924237]:
 assert not Path(f'/proc/{pid}').exists(),pid
assert os.uname().nodename=='dsw-556082-75966ff79-6qp6m' and os.getuid()==0
assert json.loads((run/'formal-four-complete.json').read_text())['runs']==4
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits','-i','0,1,2,3'],text=True)
gpu_rows=[list(map(str.strip,x.split(','))) for x in gpu.splitlines()]
assert len(gpu_rows)==4 and all(int(x[2])<=3 for x in gpu_rows)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits','-i','0,1,2,3'],text=True)
assert not apps.strip()
terminal={}
for env,rounds in [('math',50),('mixed',30)]:
 for op in ['am','gm']:
  path=run/f'formal-{env}-{op}'
  c=json.loads((path/'COMPLETE.json').read_text())
  stop=json.loads((path/'STOP_DECISION.json').read_text())
  test=json.loads((path/'test-final-metrics.json').read_text())
  assert not (path/'FAILURE.json').exists()
  assert c['formal'] is True and c['rounds']==rounds and c['updates']==16*rounds
  assert stop==c['paired_stop_decision'] and stop['stop'] is True and stop['reason']=='paired_plateau'
  assert test==c['metrics'] and [test[k]['count'] for k in ('gsm8k','arc','boolq')]==[1319,2376,3270]
  terminal[path.name]={name:{'sha256':hashlib.sha256((path/name).read_bytes()).hexdigest(),'data':data} for name,data in [('COMPLETE.json',c),('STOP_DECISION.json',stop),('test-final-metrics.json',test)]}
 assert terminal[f'formal-{env}-am']['STOP_DECISION.json']==terminal[f'formal-{env}-gm']['STOP_DECISION.json']
out.mkdir()
text=original.read_text().replace('/runs/seed71','/runs/seed101')
with (out/'verify.py').open('x') as f:f.write(text)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(source),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
with (out/'LOG.txt').open('xb') as f:
 p=subprocess.Popen([str(root/'env/bin/python'),'-u',str(out/'verify.py')],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
proc=Path(f'/proc/{p.pid}')
raw=(proc/'cmdline').read_bytes()
launch={'pid':p.pid,'start_ticks':int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),'uid':proc.stat().st_uid,'cmdline_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'original_source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'output':str(out),'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cuda_visible_devices':''}
with (out/'LAUNCH.json').open('x') as f:json.dump(launch,f,indent=2)
print(json.dumps({'launch':launch,'terminal':terminal,'gpu':gpu_rows,'compute_apps':apps.splitlines(),'original_pids_gone':[3924228, 3924230, 3924234, 3924236, 3924235, 3924237]}))
