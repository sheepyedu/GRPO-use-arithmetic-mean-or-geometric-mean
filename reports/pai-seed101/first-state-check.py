import datetime,json,hashlib,torch
from pathlib import Path
from safetensors.torch import load_file
torch.set_num_threads(4)
root=Path('/mount-data/GRPO-Sequence-Ratio/runs/seed101');checks=[]
for env in ['math','mixed']:
 for op in ['am','gm']:
  ck=root/f'formal-{env}-{op}'/'checkpoint-001'
  committed=json.loads((ck/'COMMITTED.json').read_text())
  assert committed['round']==1 and committed['updates']==16
  for name,digest in committed['files_sha256'].items():
   h=hashlib.sha256()
   with (ck/name).open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   assert h.hexdigest()==digest
  state=torch.load(ck/'training_state.pt',map_location='cpu',weights_only=False)
  entries=state['optimizer']['state'];ts=[v for st in entries.values() for v in st.values() if torch.is_tensor(v)]
  assert all(torch.isfinite(t).all().item() for t in ts)
  adapter=load_file(str(ck/'adapter_model.safetensors'),device='cpu')
  assert all(torch.isfinite(t).all().item() for t in adapter.values())
  assert state['round']==1 and state['updates']==16 and state['scheduler']['last_epoch']==16
  assert all(k in state for k in ['python_rng','numpy_rng','torch_rng','cuda_rng','schedule_rng','schedule_orders'])
  checks.append({'checkpoint':str(ck),'round':1,'updates':16,'optimizer_parameters':len(entries),'optimizer_tensors':len(ts),'optimizer_finite':True,'adapter_tensors':len(adapter),'adapter_finite':True,'scheduler_epoch':16,'rng_schedule_present':True,'file_hashes_verified':True})
  del state,entries,ts,adapter
print(json.dumps({'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':checks,'scope':'CPU-only read of four committed first-round checkpoints, no generation or optimizer replay'}))
