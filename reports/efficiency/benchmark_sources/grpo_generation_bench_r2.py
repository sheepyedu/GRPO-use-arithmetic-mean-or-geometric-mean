import os,json,time,subprocess,sys,tempfile
from pathlib import Path
def main():
 root=Path('/mount-data/GRPO-Sequence-Ratio/efficiency-generation-v3-r2')
 if len(sys.argv)==1:
  for name in ['eager32','graph64']:
   with (root/(name+'.log')).open('xb') as log:
    p=subprocess.run([sys.executable,'-u',__file__,name],env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8','VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_RPC_BASE_PATH':tempfile.mkdtemp(prefix='grpo-speed-')},stdout=log,stderr=subprocess.STDOUT)
   (root/(name+'-exit.json')).write_text(json.dumps({'code':p.returncode}))
   if p.returncode:break
 else:
  from vllm import LLM,SamplingParams
  import torch
  name=sys.argv[1];old=json.loads(Path('/mount-data/GRPO-Sequence-Ratio/runs/seed17/formal-math-am/rollout-001.json').read_text());prompts=[{'prompt_token_ids':r['prompt_ids']} for r in old[::8][:48]]
  start=time.perf_counter()
  engine=LLM(model='/mount-data/Models/Qwen3-8B',dtype='bfloat16',tensor_parallel_size=1,enable_lora=True,max_lora_rank=16,enable_sleep_mode=True,gpu_memory_utilization=.72,max_model_len=8192,max_num_seqs=32 if name=='eager32' else 64,max_num_batched_tokens=4096,enforce_eager=name=='eager32',seed=29,distributed_executor_backend='uni',disable_log_stats=True)
  init=time.perf_counter()-start
  engine.generate(prompts[:8],SamplingParams(temperature=0,max_tokens=16),use_tqdm=False)
  rows=[]
  for repeat in range(2):
   torch.cuda.synchronize();start=time.perf_counter();answers=engine.generate(prompts,SamplingParams(temperature=0,max_tokens=256),use_tqdm=False);torch.cuda.synchronize();elapsed=time.perf_counter()-start;tokens=sum(len(o.token_ids) for a in answers for o in a.outputs);rows.append({'seconds':elapsed,'output_tokens':tokens,'tokens_per_second':tokens/elapsed})
  from vllm.lora.request import LoRARequest
  adapter=LoRARequest('existing-seed17',1,'/mount-data/GRPO-Sequence-Ratio/runs/seed17/formal-math-am/checkpoint-036')
  first=engine.generate(prompts[:2],SamplingParams(temperature=0,max_tokens=16),lora_request=adapter,use_tqdm=False)
  engine.sleep(level=1);engine.wake_up()
  second=engine.generate(prompts[:2],SamplingParams(temperature=0,max_tokens=16),lora_request=adapter,use_tqdm=False)
  assert [[s.token_ids for s in x.outputs] for x in first]==[[s.token_ids for s in x.outputs] for x in second]
 
  (root/(name+'-RESULT.json')).write_text(json.dumps({'name':name,'initialization_seconds':init,'rows':rows,'sleep_wake_passed':True,'scope':'Engineering timing only, 48 fixed saved prompts, greedy capped256, no scientific results'},indent=2))

if __name__=='__main__':
 main()
