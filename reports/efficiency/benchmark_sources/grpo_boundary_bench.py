import os,time,json,gc,traceback
from pathlib import Path
import torch
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM,set_seed
from src.train import token_scores
from src.objectives import sequence_loss
out=Path('/mount-data/GRPO-Sequence-Ratio/efficiency-memory-boundary-v3');out.mkdir(exist_ok=True)
torch.set_num_threads(8);set_seed(29)
try:
 records=json.loads(Path('/mount-data/GRPO-Sequence-Ratio/runs/seed17/formal-math-am/rollout-001.json').read_text())[::8][:8]
 records=[{'prompt_ids':[2]*128,'token_ids':[3]*1408}]
 model=AutoModelForCausalLM.from_pretrained('/mount-data/Models/Qwen3-8B',torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
 model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM',bias='none')).cuda()
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 def scores(batch):
  result=[]
  for r in batch:
   ids=torch.tensor([r['prompt_ids']+r['token_ids']],device='cuda')
   result.append(token_scores(model,ids,len(r['prompt_ids'])))
  return result
 model.eval()
 with torch.no_grad():old=[x[0].cpu() for x in scores(records[:1])]
 rows=[];baselines=None;grads=None
 for size in [1,2]:
  model.grpo_fused_statistics = False
  if size==2:model.gradient_checkpointing_disable()
  else:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
  if size == 2:
   warm=scores(records[:1]);warm[0][0].sum().backward();model.zero_grad()
  model.eval();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();values=[]
  with torch.no_grad():
   for i in range(0,len(records),1):values.extend(x[0].cpu() for x in scores(records[i:i+1]))
  torch.cuda.synchronize();scoring=time.perf_counter()-start
  if baselines is None:baselines=values
  diff=max(float((a-b).abs().max()) for a,b in zip(values,baselines))
  mae=sum(float((a-b).abs().mean()) for a,b in zip(values,baselines))/len(values)
  model.train();model.zero_grad();torch.cuda.synchronize();start=time.perf_counter()
  for i in range(0,len(records),1):
   losses=[]
   for j,(current,_) in enumerate(scores(records[i:i+1])):
    ref=baselines[i+j].cuda();loss,_=sequence_loss(current,ref,ref,torch.ones_like(current,dtype=torch.bool),torch.tensor([1.0 if (i+j)%2 else -1.0],device='cuda'),'am')
    losses.append(loss)
   (torch.stack(losses).sum()/len(records)).backward()
  torch.cuda.synchronize();backward=time.perf_counter()-start
  current=torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.requires_grad and p.grad is not None]);assert torch.isfinite(current).all()
  if grads is None:grads=current
  relative=float((current-grads).norm()/grads.norm())
  rows.append({'mode':'no_checkpoint' if size==2 else 'checkpoint','scoring_seconds':scoring,'forward_backward_seconds':backward,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'logp_max_abs_difference':diff,'logp_mae':mae,'gradient_relative_l2_difference':relative,'sequences':len(records),'tokens':sum(len(r['token_ids']) for r in records)})
  (out/'progress.json').write_text(json.dumps(rows,indent=2))
  print(json.dumps(rows[-1]),flush=True)
  model.zero_grad();gc.collect();torch.cuda.empty_cache()
 (out/'RESULT.json').write_text(json.dumps({'code':0,'rows':rows,'scope':'Engineering benchmark on eight saved sequences, fresh adapter, no optimizer step or scientific results'},indent=2))
except BaseException:
 (out/'FAILURE.txt').write_text(traceback.format_exc());raise
