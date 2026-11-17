import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

root = Path('/mount-data/GRPO-Sequence-Ratio')
source = root / 'source-v4'
output = root / 'runs/seed71'
logs = root / 'logs'
claim = logs / 'seed71-launch-v4-claim.json'
config = source / 'configs/pai-seed71.json'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert socket.gethostname() == 'dsw-556082-75966ff79-6qp6m'
assert os.getuid() == 0 and len(os.sched_getaffinity(0)) == 184
assert Path('/sys/fs/cgroup/memory/memory.limit_in_bytes').read_text().strip() == '1932735283200'
assert 'cpfs-03001u70faecthn4bkib9' in subprocess.check_output(['findmnt', '-T', '/mount-data', '-n', '-o', 'SOURCE'], text=True)
assert shutil.disk_usage('/mount-data').free > 200 * 1024**3
acceptance = root / 'runs/seed43/acceptance-v1'
assert sha(acceptance / 'RESULT.json') == '4d45d7daddb16b58f92e72ca1b206c5ae607295556887beb8b02ac62dd86ccd5'
assert json.loads((acceptance / 'EXIT.json').read_text())['code'] == 0
for pid in [3223389, 3223390, 3223394, 3223395, 3223396, 3223397, 3618719]:
    assert not Path(f'/proc/{pid}').exists(), pid
manifest = json.loads((root / 'runs/seed43/formal-math-am/source_manifest.json').read_text())
assert all(sha(source / name) == digest for name, digest in manifest.items())
assert sha(config) == '9c54651ffa92e821d2cbe6f257f46b041fed1ed7172f7a7e5128cf42ad64b63f'
old = json.loads((source / 'configs/pai-seed43.json').read_text())
new = json.loads(config.read_text())
assert new == {**old, 'seed': 71}
archive_sha = sha(root / 'source-v4.tar.gz')
assert archive_sha == '65d086a3127f73afbd3f7073e4258d891ea3cc9d34cc823a59cfb30611dc35c2'
gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
selected = {row.split(',')[1].strip() for row in gpu_rows if int(row.split(',')[0]) in range(4)}
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True).splitlines()
assert len(selected) == 4 and not any(row.split(',')[0].strip() in selected for row in apps)
assert not output.exists() and not claim.exists()
old_wrapper = logs / 'seed43-run-v4-wrapper.py'
assert sha(old_wrapper) == '85c1cab1cac61e72389a9c6e79a4fcb8bbb41c7fb61ae4ca41e60c1c56f00826'
wrapper = logs / 'seed71-run-v4-wrapper.py'
wrapper_text = old_wrapper.read_text().replace('seed43', 'seed71')
with claim.open('x') as handle:
    json.dump({'seed': 71, 'source': str(source), 'output': str(output), 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'state': 'claimed_before_dispatch'}, handle)
with wrapper.open('x') as handle:
    handle.write(wrapper_text)
output.mkdir()
environment = {**os.environ, 'HF_HOME': str(root / 'cache/huggingface'), 'VLLM_CACHE_ROOT': str(root / 'cache/vllm'), 'TRITON_CACHE_DIR': str(root / 'cache/triton'), 'TORCHINDUCTOR_CACHE_DIR': str(root / 'cache/inductor'), 'XDG_CACHE_HOME': str(root / 'cache'), 'PYTHONPATH': str(source), 'OMP_NUM_THREADS': '8', 'TOKENIZERS_PARALLELISM': 'false'}
with (logs / 'seed71-run-v4.log').open('xb') as handle:
    process = subprocess.Popen([str(root / 'env/bin/python'), '-u', str(wrapper)], cwd=source, env=environment, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
proc = Path(f'/proc/{process.pid}')
raw = (proc / 'cmdline').read_bytes()
identity = {'wrapper_pid': process.pid, 'start_ticks': int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]), 'uid': proc.stat().st_uid, 'cmdline_sha256': hashlib.sha256(raw).hexdigest(), 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source': str(source), 'config_sha256': sha(config), 'archive_sha256': archive_sha, 'output': str(output), 'gpu_scope': [0, 1, 2, 3], 'state': 'spawned', 'wrapper_source_sha256': sha(wrapper), 'claim_path': str(claim), 'preflight_source_files': len(manifest)}
with (logs / 'seed71-launch-v4.json').open('x') as handle:
    json.dump(identity, handle, indent=2)
print(json.dumps(identity))
