"""agent.py
主循环调度模块：
检索长期经验 -> 生成答案 -> 三层评估 -> 失败则进化提示词 -> 经验回放
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from evaluator import EvaluationReport, Evaluator
from evolution import PromptEvolver, ToolBootstrapper
from memory import LongTermMemory, ShortTermMemory


class OpenAIOrMockLLM:
    """优先走OpenAI API；无Key或依赖时自动回退到本地可演示Mock。"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI  # type: ignore

                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate(self, prompt: str, system_prompt: str) -> str:
        if self.client is not None:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()
        return self._mock_generate(prompt, system_prompt)

    def _mock_generate(self, prompt: str, system_prompt: str) -> str:
        # 逻辑自检分支：故意能指出“5”是错误
        if "严苛审稿人" in prompt and "2+2" in prompt:
            if re.search(r"答案:\s*5", prompt):
                return "错误：2+2不应等于5，存在明显矛盾。"
            return "通过。"

        # 任务回答分支：未进化前故意犯错；进化后修正
        if "2+2" in prompt:
            if "必须先计算并复核结果" in system_prompt or "先计算并复核" in system_prompt:
                return "4"
            return "5"
        return "这是一个基础回答。"


@dataclass
class AgentResult:
    answer: str
    report: EvaluationReport
    attempts: List[Dict]


class AGIAgent:
    def __init__(
        self,
        memory: LongTermMemory,
        evaluator: Evaluator,
        evolver: PromptEvolver,
        tool_bootstrapper: ToolBootstrapper,
        llm: OpenAIOrMockLLM,
        max_attempts: int = 2,
    ):
        self.memory = memory
        self.evaluator = evaluator
        self.evolver = evolver
        self.tool_bootstrapper = tool_bootstrapper
        self.llm = llm
        self.short_memory = ShortTermMemory(max_turns=8)
        self.max_attempts = max(1, max_attempts)

    def solve(self, task: str, task_type: str = "general") -> AgentResult:
        solve_started = datetime.now(timezone.utc).isoformat()
        # 1) 先检索长期记忆经验
        related = self.memory.search_related(task, top_k=3)
        memory_hints = self._build_memory_hints(related)
        attempts: List[Dict] = []

        final_answer = ""
        final_report: EvaluationReport

        for _ in range(self.max_attempts):
            system_prompt = self.evolver.get_system_prompt()
            user_prompt = f"历史经验:\n{memory_hints}\n\n当前任务:\n{task}\n请给出最终答案。"
            answer = self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)
            report = self.evaluator.evaluate(task=task, answer=answer, task_type=task_type)
            attempts.append({"answer": answer, "report": report})

            self.short_memory.add("user", task)
            self.short_memory.add("assistant", answer)

            final_answer, final_report = answer, report
            if report.passed:
                break

            # 2) 失败后触发自我进化：动态更新system prompt
            self.evolver.evolve_from_failure(task=task, answer=answer, reasons=report.reasons)

        # 3) 任务结束后做经验回放沉淀
        attempt_texts = [f"第{i+1}次: {a['answer']} | 评估: {a['report'].reasons}" for i, a in enumerate(attempts)]
        rules = ["失败经验已触发元提示词更新。"] if not final_report.passed else ["该解法可复用。"]
        self.memory.replay_experience(
            question=task,
            attempts=attempt_texts,
            final_result=final_answer,
            rules=rules,
            timestamp=solve_started,
        )
        return AgentResult(answer=final_answer, report=final_report, attempts=attempts)

    @staticmethod
    def _build_memory_hints(related: List[Dict]) -> str:
        if not related:
            return "暂无历史经验。"
        lines = []
        for item in related:
            meta = item.get("metadata", {})
            rules = meta.get("rules", [])
            lines.append(f"- 相似度={item.get('score', 0):.3f}; 规则={rules}")
        return "\n".join(lines)
