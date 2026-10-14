# Run Gemma 4 LoRA on AWS g6e.2xlarge

This guide targets **one NVIDIA L40S (48 GB VRAM), 64 GiB host RAM** on Linux.
The notebook is not modified. Python defaults now target single-GPU Gemma 4:
bfloat16, microbatch 1, 1,024-token maximum, LoRA rank 16/alpha 32,
gradient accumulation 4, non-reentrant gradient checkpointing, and `cuda:0`.
No multi-GPU launcher, CPU offloading, or QLoRA is needed for the initial run.
Memory fit and speed still require measurement on your actual checkpoint.

## 1. Launch carefully with your $50 budget

- Check the **current regional price**, GPU quota and Spot availability before launch.
- Choose a free Ubuntu NVIDIA/PyTorch Deep Learning AMI with a compatible driver.
  Avoid paid Marketplace images. Use Python 3.11 or 3.12, not Kaggle's Python 3.13.
- Use persistent encrypted EBS for weights/data/checkpoints; 80–100 GB is a starting
  allocation, not a guarantee. EBS, public IPv4, snapshots and transfer can incur
  charges beyond instance hours. Spot can terminate with little warning.
- Set billing alerts (they are **not** a hard spending cap). Reserve at least $10
  for setup, storage and retries. Credits eligibility depends on your credit terms.
- Stop/terminate idle compute. EBS still bills when an instance is stopped; verify
  volume deletion settings and back up needed checkpoints before termination.

## 2. Install in a clean environment

Transfer this project to the instance and work from its root directory. The commands
below assume the project is at `/home/ubuntu/Gemma4-Kaggle-Agent`.

```bash
cd /home/ubuntu/Gemma4-Kaggle-Agent
nvidia-smi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Explicit CUDA 12.8 Torch wheel, matching pyproject.toml. Do not use the CPU wheel.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# Base dependencies suffice for the custom LoRA SFT loop; no vLLM/TRL install needed.
python -m pip install -e .
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -m hackrl.run_kaggle --help
```

Use `python3.11` instead if 3.12 is unavailable. Confirm `nvidia-smi` reports an L40S
and a driver compatible with CUDA 12.8. Do not change the NVIDIA driver blindly during
an active run. Packaging dependencies are declared in `pyproject.toml`, not the empty
`requirements.txt`. The pre-existing `uv.lock` was not changed by this implementation.

### Offline installation/execution

The installation above is an **Internet-enabled preparation step on AWS**, not the
competition notebook. For a fully offline AWS environment, prepare a complete Linux
x86-64 wheelhouse for the exact Python version, including the CUDA Torch wheel and
project/build dependencies, before disconnecting:

```bash
python -m pip install --no-index --find-links /home/ubuntu/wheels -e .
```

An old Transformers 4.56.2 wheelhouse cannot load Gemma 4. Use this project's
Transformers 5.5.x / PEFT 0.21.x requirements and validate dependencies before paying
for a long GPU run. AWS paths are not `/kaggle/input` paths.

## 3. Stage the checkpoint and data

Place a complete licensed checkpoint at `/home/ubuntu/models/gemma-4-e4b-it`.
Transfer from your authorized local copy, or accept the model's access conditions
and download during preparation using Hugging Face authentication:

```bash
hf auth login
hf download google/gemma-4-e4b-it --local-dir /home/ubuntu/models/gemma-4-e4b-it
```

Do not publish your token. For offline execution after staging:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
```

Provide `/home/ubuntu/data/problems.jsonl` in normalized `ProblemSample` format.
For example, one line:

```json
{"task_id":"example/add","prompt":"def add(a, b):\n    \"\"\"Add integers.\"\"\"\n","canonical_solution":"    return a + b\n","test_clean":"assert add(2, 3) == 5","entry_point":"add"}
```

Raw EvalPlus input arrays are not runnable test source. SFT never executes solution
or test code. Use a held-out split for evaluation; a pilot is not benchmark evidence.

## 4. Run the preflight

```bash
python -m hackrl.run_kaggle --mode preflight \
  --model-id /home/ubuntu/models/gemma-4-e4b-it \
  --local-files-only --output-dir /home/ubuntu/runs/preflight
```

This validates CUDA/bfloat16, native causal-LM configuration support and package
versions without allocating the model weights. It writes `preflight.json`.

## 5. Test four updates with explicit pilot data

```bash
python -m hackrl.run_kaggle --mode train-sft --pilot --alpha 0.0 \
  --model-id /home/ubuntu/models/gemma-4-e4b-it --local-files-only \
  --steps 4 --seq-length 512 --batch-size 1 --grad-accum 1 \
  --save-every 2 --output-dir /home/ubuntu/runs/pilot-01
```

No silent four-task fallback exists anymore: `--pilot` is required. Every run must
use a new output directory. The final adapter is under
`/home/ubuntu/runs/pilot-01/sft_alpha_0.0/checkpoint_step_4`.

## 6. Measure 20 real optimizer updates

```bash
python -m hackrl.run_kaggle --mode train-sft --alpha 0.0 \
  --model-id /home/ubuntu/models/gemma-4-e4b-it --local-files-only \
  --data-file /home/ubuntu/data/problems.jsonl \
  --steps 20 --seq-length 1024 --batch-size 1 --grad-accum 4 \
  --save-every 10 --output-dir /home/ubuntu/runs/benchmark-01
```

`--steps` now counts **optimizer updates**, not microbatches. The old code counted
microbatches, so historical step labels/timing estimates are not directly comparable.
Each update above processes four microbatches. SFT masks prompt/BOS and padding
tokens and supervises only the solution plus EOS. Full function solutions are not
duplicated by appending the prompt. Boundary-spanning tokenizer tokens are supervised.
This is raw-code completion SFT, not chat-template SFT.

Inspect `training_metrics.json` for loss, seconds/update, input-token throughput and
peak allocated GPU memory. CUDA timings exclude model loading and checkpoint saves.
Discard initial warm-up steps and measure total wall time separately.

```bash
python - <<'PY'
import json
import statistics
from pathlib import Path
path = Path('/home/ubuntu/runs/benchmark-01/sft_alpha_0.0/training_metrics.json')
metrics = json.loads(path.read_text())
seconds = statistics.median(row['seconds'] for row in metrics['loss_history'][5:])
print('Median seconds/update:', seconds)
print('Estimated compute hours for 500 updates:', seconds * 500 / 3600)
print('Peak allocated VRAM GiB:', metrics['peak_vram_gib'])
PY
```

Multiply hours by your actual hourly rate and add setup/checkpoint time/storage costs.
Earlier chat training-time numbers were unmeasured estimates, not benchmarks.

## 7. Run the baseline only after checking cost and memory

```bash
python -m hackrl.run_kaggle --mode train-sft --alpha 0.0 \
  --model-id /home/ubuntu/models/gemma-4-e4b-it --local-files-only \
  --data-file /home/ubuntu/data/problems.jsonl \
  --steps 500 --seq-length 1024 --batch-size 1 --grad-accum 4 \
  --save-every 25 --seed 42 --output-dir /home/ubuntu/runs/sft-01
```

Keep rank 16/alpha 32 consistent across comparisons. Increase sequence length or
batch size only after memory measurements. On OOM, reduce sequence length first,
then batch size; changing these also changes the experiment, so log the change.
Sample order is deterministic/cyclic, not shuffled. Validation and full distributed
training are not implemented in this loop. No scheduler is used (fixed learning rate).

## 8. Resume after a Spot interruption

Checkpoints contain LoRA weights, tokenizer, optimizer state, sample cursor, RNG states,
data fingerprint and `COMPLETE`. No full base-model weight copies are written.
Ignore any `.incomplete` directory. Resume only **your own trusted checkpoint**:
`training_state.pt` contains pickle-serialized RNG/optimizer state.

For an interruption after update 100 of the baseline:

```bash
python -m hackrl.run_kaggle --mode train-sft --alpha 0.0 \
  --model-id /home/ubuntu/models/gemma-4-e4b-it --local-files-only \
  --data-file /home/ubuntu/data/problems.jsonl \
  --steps 500 --seq-length 1024 --batch-size 1 --grad-accum 4 --seed 42 \
  --resume-from /home/ubuntu/runs/sft-01/sft_alpha_0.0/checkpoint_step_100 \
  --output-dir /home/ubuntu/runs/sft-resumed-01
```

`--steps 500` is the **total target**, not 500 more updates. Keep the model path, data
and ordering, seed, rank, batch, accumulation and other update settings identical.
Resume into a fresh directory. Keep model files immutable and the same library/runtime
versions; exact GPU bitwise reproducibility is not guaranteed. Preserve checkpoints
on retained EBS or back them up securely before termination. Local NVMe is not durable.

## 9. Limits and safety

- GRPO remains a custom, unvalidated prototype. Installing TRL does not change it.
- `train-grpo`, `evaluate` and `all` are blocked by default because they execute
  generated code in an ordinary subprocess, not a secure sandbox. A programmatic
  caller is guarded too. `--allow-unsafe-execution` is a prototype acknowledgement,
  **not isolation**; do not use it on a host with credentials or sensitive data.
- A proper isolated reward backend and validated GRPO objective are prerequisites
  for reportable RL experiments. More GPU memory does not solve that problem.
- A base-only refusal probe comparing a model to itself does not establish drift.
- The notebook's old markdown descriptions of SFT remain unchanged as requested;
  this guide documents the new Python implementation instead.

## 10. Stop paying when finished

Download/back up adapters and logs, verify the copy, then stop or terminate the
instance in the EC2 console. Review retained EBS volumes/snapshots/public IPs and
billing. Do not assume a $50 credit automatically stops AWS at $50.



python - <<'PY'
import json, statistics
from pathlib import Path

path = Path('output/runs/benchmark-01/sft_alpha_0.0/training_metrics.json')
metrics = json.loads(path.read_text())
history = metrics['loss_history']
seconds = statistics.median(r['seconds'] for r in history[5:])

print(f"Initial loss: {history[0]['loss']:.4f} -> Final step loss: {history[-1]['loss']:.4f}")
print(f"Median seconds/update: {seconds:.2f}s")
print(f"Estimated time for 500 updates: {seconds * 500 / 3600:.2f} hours")
print(f"Peak VRAM: {metrics['peak_vram_gib']:.2f} GiB")
PY
