import datetime
import hashlib
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from scripts.status import snapshot

source = Path('/mount-data/GRPO-Sequence-Ratio/source-v4')
root = Path('/mount-data/GRPO-Sequence-Ratio/runs/seed101')
host = {'hostname': socket.gethostname(), 'uid': os.getuid(), 'cpus':len(os.sched_getaffinity(0)), 'memory_limit':Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text().strip(), 'cpfs':subprocess.check_output(['findmnt','-T','/mount-data','-n','-o','SOURCE'],text=True).strip()}
assert host['hostname']=='dsw-556082-75966ff79-6qp6m' and host['uid']==0 and host['cpus']==184
assert host['memory_limit']=='1932735283200' and 'cpfs-03001u70faecthn4bkib9' in host['cpfs']
s = snapshot(root)
s['host_identity'] = host
s['terminal_files'] = {}
for p in [root/'formal-four-complete.json',root/'formal-math-exit.json',root/'formal-mixed-exit.json',root.parent.parent/'logs/seed101-run-v4-exit.json']:
 s['terminal_files'][str(p)] = json.loads(p.read_text()) if p.exists() else None

def finite(value):
 if isinstance(value,float): return math.isfinite(value)
 if isinstance(value,dict): return all(finite(v) for v in value.values())
 if isinstance(value,list): return all(finite(v) for v in value)
 return True

config=json.loads((source/'configs/pai-seed101.json').read_text())
s['training_progress']=[]
for run in s['runs']:
 p=Path(run['directory'])
 rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines() if line.strip()] if (p/'metrics.jsonl').exists() else []
 env,operator=p.name.split('-')[1:]
 control=root/f'formal-{env}-control'
 checkpoints=sorted(p.glob('checkpoint-*/COMMITTED.json'))
 committed=checkpoints[-1] if checkpoints else None
 number=int(committed.parent.name.split('-')[-1]) if committed else 0
 ready_path=control/f'ready-{operator}-{number:03d}.json'
 ready=json.loads(ready_path.read_text()) if ready_path.exists() else None
 decisions=sorted(control.glob('decision-*.json'))
 manifest=json.loads((p/'source_manifest.json').read_text()) if (p/'source_manifest.json').exists() else None
 actual_config=json.loads((p/'config.json').read_text())
 commit_sha=hashlib.sha256(committed.read_bytes()).hexdigest() if committed else None
 item={'directory':str(p),'recorded_rounds':[r['round'] for r in rows],'metrics_all_finite':all(finite(r) for r in rows),'latest_metrics':rows[-1] if rows else None,'validation_metrics':[{'round':int(q.name.split('-')[1]),'metrics':json.loads(q.read_text())} for q in sorted(p.glob('validation-*-metrics.json'))],'committed_sha256':commit_sha,'ready_matches_committed':ready.get('checkpoint_sha256')==commit_sha if ready is not None else None,'latest_paired_decision':json.loads(decisions[-1].read_text()) if decisions else None,'source_matches':all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in manifest.items()) if manifest is not None else None,'config_matches':actual_config=={**config,'environment':env,'operator':operator}}
 s['training_progress'].append(item)
s['controllers']=[]
for p in sorted(root.glob('formal-*-control/controller.json')):
 c=json.loads(p.read_text())
 proc=Path('/proc')/str(c['pid'])
 try:
  raw=(proc/'cmdline').read_bytes()
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  identity={'pid':c['pid'],'uid':proc.stat().st_uid,'start_ticks':int(stat[19]),'cmdline_sha256':hashlib.sha256(raw).hexdigest(),'state':stat[0],'matches_original_start':int(stat[19])==c['start_ticks'],'matches_config':b'configs/pai-seed101.json' in raw,'matches_output':os.fsencode(str(root)) in raw}
 except FileNotFoundError:
  identity={'pid':c['pid'],'gone':True}
 s['controllers'].append({'path':str(p),'controller':c,'identity':identity})
s['observed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(s))
