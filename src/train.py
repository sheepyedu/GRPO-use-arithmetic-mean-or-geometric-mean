"""Single-GPU BF16 LoRA trainer with colocated vLLM sleep/offload.

Each process owns one visible GPU. The two objectives share this entry point.
Rollouts, frozen behavior/reference scores and optimizer commits are retained.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.data import PromptSchedule, content_hash, write_json
from src.objectives import group_advantages, sequence_loss
from src.rewards import parse_answer, score, user_prompt


def event(directory, phase, **fields):
    value = {"time": time.time(), "phase": phase, "pid": os.getpid(), **fields}
    write_json(directory / "status.json", value)
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def token_scores(model, ids, prompt_length):
    # Only materialize completion logits, reducing peak vocabulary activation.
    output = model(
        input_ids=ids, use_cache=False, logits_to_keep=ids.shape[1] - prompt_length + 1
    )
    logits = output.logits[:, :-1, :].float()
    targets = ids[:, prompt_length:]
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_probabilities = selected - torch.logsumexp(logits, dim=-1)
    with torch.no_grad():
        probabilities = logits.softmax(-1)
        entropy = (
            torch.logsumexp(logits, -1) - (probabilities * logits).sum(-1)
        ).mean()
    return log_probabilities, entropy


def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class Experiment:
    def __init__(self, config, environment, operator, directory):
        from vllm import LLM

        if config["prompt_batch"] != 64 or config["group_size"] != 8:
            raise ValueError("This protocol requires 64 prompts and 8 answers")
        if config["microbatch_sequences"] != 1:
            raise ValueError("The current trainer supports sequence microbatch 1")
        if config["minibatch_sequences"] < 1 or 512 % config["minibatch_sequences"]:
            raise ValueError("The optimizer minibatch must divide 512 sequences")
        self.config, self.environment, self.operator = config, environment, operator
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        write_json(
            self.directory / "config.json",
            {**config, "environment": environment, "operator": operator},
        )
        event(
            self.directory, "initializing", environment=environment, operator=operator
        )
        self.start = time.time()
        set_seed(config["seed"])
        torch.set_num_threads(8)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"], local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["lora_targets"],
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.config.use_cache = False
        self.model.eval()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.optimizer, factor=1.0, total_iters=1
        )
        self.engine = LLM(
            model=config["model"],
            dtype="bfloat16",
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=config["lora_rank"],
            max_loras=1,
            max_cpu_loras=2,
            enable_sleep_mode=True,
            gpu_memory_utilization=config["vllm_memory_fraction"],
            max_model_len=config["max_model_len"],
            max_num_seqs=config["vllm_max_sequences"],
            max_num_batched_tokens=4096,
            enable_chunked_prefill=True,
            enforce_eager=True,
            seed=config["seed"],
            distributed_executor_backend="uni",
            disable_log_stats=True,
        )
        self.adapter_id = 0
        self.engine_asleep = False
        self.rows = json.loads(
            (Path(config["data"]) / f"{environment}.json").read_text()
        )
        self.validation = json.loads(
            (Path(config["data"]) / "validation.json").read_text()
        )
        self.test = json.loads((Path(config["data"]) / "test.json").read_text())
        data_manifest = json.loads((Path(config["data"]) / "manifest.json").read_text())
        for name, rows in (
            (environment, self.rows),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if content_hash(rows) != data_manifest["split_hashes"][name]:
                raise ValueError(f"Data content hash mismatch for {name}")
        self.schedule = PromptSchedule(self.rows, environment, config["seed"])
        self.updates = 0
        source_root = Path(__file__).resolve().parent.parent
        source_files = {
            str(path.relative_to(source_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for folder in ("src", "scripts", "configs")
            for path in sorted((source_root / folder).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
        write_json(self.directory / "source_manifest.json", source_files)
        write_json(
            self.directory / "environment.json",
            {
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ["torch", "transformers", "peft", "vllm", "datasets"]
                },
                "gpu": torch.cuda.get_device_name(),
                "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "data_manifest_hash": content_hash(
                    json.loads((Path(config["data"]) / "manifest.json").read_text())
                ),
                "trainable_parameters": sum(p.numel() for p in self.parameters),
                "thinking": config["thinking"],
                "template_sha": content_hash(self.tokenizer.chat_template),
            },
        )

    def prompts(self, rows):
        result = []
        for row in rows:
            tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt(row)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.config["thinking"],
            )
            if (
                len(tokens) + self.config["max_new_tokens"]
                > self.config["max_model_len"]
            ):
                raise ValueError(f"Prompt exceeds context limit: {row['id']}")
            result.append({"prompt_token_ids": tokens})
        return result

    def adapter(self, path):
        from vllm.lora.request import LoRARequest

        if path is None:
            return None
        if self.adapter_id:
            self.engine.llm_engine.remove_lora(self.adapter_id)
        self.adapter_id += 1
        return LoRARequest(
            f"round-{self.adapter_id}", self.adapter_id, str(Path(path).resolve())
        )

    def generate(self, rows, adapter, n, seed, greedy=False):
        from vllm import SamplingParams

        parameters = SamplingParams(
            n=n,
            temperature=0 if greedy else self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_new_tokens"],
            seed=seed,
            logprobs=1 if not greedy else None,
        )
        outputs = self.engine.generate(
            self.prompts(rows), parameters, lora_request=adapter, use_tqdm=False
        )
        records = []
        for row, output in zip(rows, outputs):
            for sample in output.outputs:
                truncated = sample.finish_reason == "length"
                completed_thinking = (
                    not self.config["thinking"] or "</think>" in sample.text
                )
                record = {
                    "row": row,
                    "prompt_ids": list(output.prompt_token_ids),
                    "token_ids": list(sample.token_ids),
                    "text": sample.text,
                    "finish_reason": sample.finish_reason,
                    "reward": score(sample.text, row, truncated)
                    if completed_thinking
                    else 0,
                    "valid": completed_thinking
                    and parse_answer(
                        sample.text, row["task"], row.get("labels"), truncated
                    )
                    is not None,
                }
                if not greedy:
                    record["sampling_log_probs"] = [
                        float(values[token].logprob)
                        for token, values in zip(sample.token_ids, sample.logprobs)
                    ]
                records.append(record)
        return records

    def evaluate(self, stage, adapter, rows):
        event(
            self.directory,
            "evaluation",
            stage=stage,
            examples=len(rows),
            updates=self.updates,
        )
        records = self.generate(rows, adapter, 1, self.config["seed"], greedy=True)
        write_json(self.directory / f"{stage}-predictions.json", records)
        metrics = {}
        for task in ("gsm8k", "arc", "boolq"):
            selected = [r for r in records if r["row"]["task"] == task]
            metrics[task] = {
                "count": len(selected),
                "accuracy": sum(r["reward"] for r in selected) / len(selected),
                "mean_tokens": sum(len(r["token_ids"]) for r in selected)
                / len(selected),
                "truncation": sum(r["finish_reason"] == "length" for r in selected)
                / len(selected),
                "valid": sum(r["valid"] for r in selected) / len(selected),
            }
        metrics["macro"] = (
            sum(metrics[t]["accuracy"] for t in ("gsm8k", "arc", "boolq")) / 3
        )
        metrics["mixed_weighted"] = (
            0.5 * metrics["gsm8k"]["accuracy"]
            + 0.25 * metrics["arc"]["accuracy"]
            + 0.25 * metrics["boolq"]["accuracy"]
        )
        write_json(self.directory / f"{stage}-metrics.json", metrics)
        return metrics

    def train_round(self, number, records):
        config = self.config
        self.engine.sleep(level=1)
        self.engine_asleep = True
        self.model.to("cuda")
        move_optimizer(self.optimizer, "cuda")
        rewards = torch.tensor([r["reward"] for r in records])
        advantages, std = group_advantages(
            rewards, config["group_size"], config["advantage_delta"]
        )
        frozen = []
        event(
            self.directory,
            "frozen_scores",
            round=number,
            sequences=len(records),
            updates=self.updates,
        )
        with torch.no_grad():
            for record in records:
                ids = torch.tensor(
                    [record["prompt_ids"] + record["token_ids"]], device="cuda"
                )
                old, _ = token_scores(self.model, ids, len(record["prompt_ids"]))
                with self.model.disable_adapter():
                    reference, _ = token_scores(
                        self.model, ids, len(record["prompt_ids"])
                    )
                frozen.append((old.cpu(), reference.cpu()))
        torch.save(
            {"old_reference": frozen, "advantages": advantages, "std": std},
            self.directory / f"frozen-{number:03d}.pt",
        )
        # Behavior policy is fixed for both optimization epochs. HF recomputes
        # scores before updates, and backend differences are measured explicitly.
        backend_error = sum(
            (old[0] - torch.tensor(r["sampling_log_probs"])).abs().mean().item()
            for r, (old, _) in zip(records, frozen)
        ) / len(records)
        summary = {
            "reward": rewards.float().mean().item(),
            "group_std": std.mean().item(),
            "zero_advantage_fraction": (std == 0).float().mean().item(),
            "sampling_scoring_logp_mae": backend_error,
            "tokens": sum(len(r["token_ids"]) for r in records),
            "truncation": sum(r["finish_reason"] == "length" for r in records)
            / len(records),
            "valid": sum(r["valid"] for r in records) / len(records),
        }
        summary["tasks"] = {
            task: {
                "count": len(selected),
                "reward": sum(r["reward"] for r in selected) / len(selected),
                "mean_tokens": sum(len(r["token_ids"]) for r in selected)
                / len(selected),
                "truncation": sum(r["finish_reason"] == "length" for r in selected)
                / len(selected),
            }
            for task in ("gsm8k", "arc", "boolq")
            if (selected := [r for r in records if r["row"]["task"] == task])
        }
        diagnostics = []
        self.model.train()
        for epoch in range(config["optimization_epochs"]):
            order = list(range(len(records)))
            random.Random(config["seed"] + number * 100 + epoch).shuffle(order)
            for offset in range(0, len(order), config["minibatch_sequences"]):
                minibatch = order[offset : offset + config["minibatch_sequences"]]
                self.optimizer.zero_grad(set_to_none=True)
                for index in minibatch:
                    record = records[index]
                    ids = torch.tensor(
                        [record["prompt_ids"] + record["token_ids"]], device="cuda"
                    )
                    current, entropy = token_scores(
                        self.model, ids, len(record["prompt_ids"])
                    )
                    old, reference = (value.to("cuda") for value in frozen[index])
                    loss, stats = sequence_loss(
                        current,
                        old,
                        reference,
                        torch.ones_like(current, dtype=torch.bool),
                        advantages[index : index + 1].to("cuda"),
                        self.operator,
                        config["epsilon"],
                        config["beta"],
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    (loss / len(minibatch)).backward()
                    diagnostics.append(
                        {
                            **{k: float(v.mean()) for k, v in stats.items()},
                            "entropy": float(entropy),
                            "loss": float(loss.detach()),
                        }
                    )
                grad = torch.nn.utils.clip_grad_norm_(
                    self.parameters, config["gradient_clip"], error_if_nonfinite=True
                )
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
                event(
                    self.directory,
                    "optimizer",
                    round=number,
                    epoch=epoch + 1,
                    updates=self.updates,
                    grad_norm=float(grad),
                    minibatch_offset=offset,
                )
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        move_optimizer(self.optimizer, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint = self.directory / f"checkpoint-{number:03d}"
        checkpoint.mkdir()
        self.model.save_pretrained(checkpoint)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "schedule_rng": self.schedule.random.getstate(),
                "schedule_orders": self.schedule.orders,
                "updates": self.updates,
                "round": number,
            },
            checkpoint / "training_state.pt",
        )
        write_json(
            checkpoint / "COMMITTED.json",
            {
                "round": number,
                "updates": self.updates,
                "adapter_sha": content_hash(
                    (checkpoint / "adapter_config.json").read_text()
                ),
            },
        )
        for key in diagnostics[0]:
            summary[key] = sum(d[key] for d in diagnostics) / len(diagnostics)
        summary["ratio_quantiles"] = {
            str(q): float(torch.tensor([d["ratio"] for d in diagnostics]).quantile(q))
            for q in (0.0, 0.01, 0.5, 0.99, 1.0)
        }
        summary.update(
            round=number, updates=self.updates, elapsed=time.time() - self.start
        )
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        self.engine.wake_up()
        self.engine_asleep = False
        return self.adapter(checkpoint)

    def run(self, calibration=False):
        adapter = None
        if calibration:
            # Fixed validation-only subset, selected by task and content order.
            rows = [
                row
                for task in ("gsm8k", "arc", "boolq")
                for row in [r for r in self.validation if r["task"] == task][:16]
            ]
            records = self.generate(rows, None, 8, self.config["seed"])
            write_json(self.directory / "calibration.json", records)
            _, std = group_advantages(torch.tensor([r["reward"] for r in records]))
            rates = {
                task: float((std[i * 16 : (i + 1) * 16] == 0).float().mean())
                for i, task in enumerate(("gsm8k", "arc", "boolq"))
            }
            write_json(
                self.directory / "calibration-summary.json",
                {
                    "zero_variance": rates,
                    "truncation": sum(r["finish_reason"] == "length" for r in records)
                    / len(records),
                    "valid": sum(r["valid"] for r in records) / len(records),
                },
            )
            if any(rate > 0.95 for rate in rates.values()):
                raise RuntimeError(
                    "Calibration degenerate; inspect parser/template/length before formal training"
                )
        else:
            self.evaluate("validation-000", adapter, self.validation)
        for number in range(1, self.config["rounds"] + 1):
            rows = self.schedule.batch()
            event(
                self.directory,
                "rollout",
                round=number,
                updates=self.updates,
                prompt_ids=[row["id"] for row in rows],
            )
            records = self.generate(
                rows, adapter, self.config["group_size"], self.config["seed"] + number
            )
            write_json(self.directory / f"rollout-{number:03d}.json", records)
            adapter = self.train_round(number, records)
            if not calibration and number % self.config["validation_every"] == 0:
                self.evaluate(f"validation-{number:03d}", adapter, self.validation)
        if not calibration:
            metrics = self.evaluate("test-final", adapter, self.test)
        else:
            metrics = {"calibration": True}
        write_json(
            self.directory / "COMPLETE.json",
            {
                "rounds": self.config["rounds"],
                "updates": self.updates,
                "elapsed_seconds": time.time() - self.start,
                "metrics": metrics,
                "formal": not calibration,
            },
        )
        event(
            self.directory,
            "complete",
            rounds=self.config["rounds"],
            updates=self.updates,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment", choices=["math", "mixed"], required=True)
    parser.add_argument("--operator", choices=["am", "gm"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output)
    try:
        experiment = Experiment(
            json.loads(Path(args.config).read_text()),
            args.environment,
            args.operator,
            directory,
        )
        experiment.run(args.calibration)
    except BaseException:
        if directory.exists():
            write_json(
                directory / "FAILURE.json",
                {"time": time.time(), "traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
