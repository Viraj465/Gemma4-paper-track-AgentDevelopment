"""Evaluation harness for agentic software engineering tasks (SWE-bench-lite)."""

import asyncio
from typing import List, Dict, Any
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from hackrl.config import ExperimentConfig
from hackrl.data.split_generator import extract_python_code
from hackrl.sandbox.verifier import SafeSandboxVerifier


class SWEEvalResult(BaseModel):
    """Results for a single SWE-bench-lite task."""

    task_id: str
    plan_correct: bool = False
    patch_applied: bool = False
    tests_passed: bool = False
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""


class SWEBenchLiteHarness:
    """Evaluates agents on simplified software engineering tasks."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.verifier = SafeSandboxVerifier(
            timeout_sec=config.timeout_sec,
            max_memory_mb=config.max_memory_mb,
        )

    async def _run_agent_turn(
        self,
        model,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Simulates a single turn of the agent generating a patch/plan."""
        device = model.get_input_embeddings().weight.device
        user_msg = (
            f"{prompt.rstrip()}\n\n"
            "Write a complete Python implementation for this task. "
            "Return only executable Python code: no explanation, reasoning, or Markdown fences."
        )
        if getattr(tokenizer, "chat_template", None):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = user_msg

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        raw_completion = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
        return extract_python_code(raw_completion)

    async def evaluate_task(
        self,
        model,
        tokenizer: AutoTokenizer,
        task_data: Dict[str, Any],
    ) -> SWEEvalResult:
        """Evaluates a single task: prompt -> patch -> verify."""
        task_id = task_data["instance_id"]
        prompt = task_data["problem_description"]
        golden_test = task_data["test_code"]

        patch = await self._run_agent_turn(model, tokenizer, prompt)
        res = self.verifier.execute(
            candidate_code=patch,
            test_code=golden_test,
            allow_early_exit=False,
            reject_exploits=True,
        )

        return SWEEvalResult(
            task_id=task_id,
            patch_applied=bool(patch.strip()),
            tests_passed=res.passed,
            stdout=res.stdout,
            stderr=res.stderr,
        )

    def evaluate_dataset(
        self,
        model,
        tokenizer: AutoTokenizer,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Runs evaluation over a set of tasks."""
        results = []
        for task in dataset:
            res = asyncio.run(self.evaluate_task(model, tokenizer, task))
            results.append(res.model_dump())

        pass_rate = sum(1 for r in results if r["tests_passed"]) / max(1, len(results))
        return {
            "pass_rate": pass_rate,
            "total_tasks": len(results),
            "results": results,
        }
