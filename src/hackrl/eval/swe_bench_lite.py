"""Evaluation harness for agentic software engineering tasks (SWE-bench-lite).

This module implements a simplified version of the SWE-bench evaluation pipeline,
allowing the model to plan and apply patches to a codebase, then verify them
using a sandbox verifier.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from hackrl.config import ExperimentConfig
from hackrl.runtime import load_model_and_tokenizer, resolve_device
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
        tokenizer, 
        prompt: str, 
        max_tokens: int = 512
    ) -> str:
        """Simulates a single turn of the agent generating a patch/plan."""
        device = model.device if hasattr(model, "device") else torch.device("cuda")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Extract only the generated part
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)

    async def evaluate_task(
        self, 
        model, 
        tokenizer, 
        task_data: Dict[str, Any]
    ) -> SWEEvalResult:
        """Evaluates a single task: prompt -> patch -> verify."""
        task_id = task_data["instance_id"]
        prompt = task_data["problem_description"]
        golden_test = task_data["test_code"]
        
        # 1. Generate Patch
        patch = await self._run_agent_turn(model, tokenizer, prompt)
        
        # 2. Apply and Verify
        # Note: In a full agent, we'd apply the patch to a repo. 
        # Here we use the verifier's execute() which combines candidate + test.
        # We assume the agent generates the fixed function directly.
        res = self.verifier.execute(
            candidate_code=patch,
            test_code=golden_test,
            allow_early_exit=False,
            reject_exploits=True,
        )
        
        return SWEEvalResult(
            task_id=task_id,
            patch_applied=True if patch else False,
            tests_passed=res.passed,
            stdout=res.stdout,
            stderr=res.stderr,
        )

    def evaluate_dataset(
        self, 
        model, 
        tokenizer, 
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs evaluation over a set of tasks."""
        results = []
        for task in dataset:
            # Use asyncio to run the async evaluate_task
            res = asyncio.run(self.evaluate_task(model, tokenizer, task))
            results.append(res.model_dump())
            
        pass_rate = sum(1 for r in results if r["tests_passed"]) / len(results)
        
        return {
            "pass_rate": pass_rate,
            "total_tasks": len(results),
            "results": results
        }
