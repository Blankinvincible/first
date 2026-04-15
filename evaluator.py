"""evaluator.py
对错分辨与置信度评分模块：
1) 逻辑自洽检查（自我反驳）
2) 事实性核查（优先记忆库交叉验证）
3) 代码沙盒执行（编程任务）
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from memory import LongTermMemory


@dataclass
class EvaluationReport:
    logic_ok: bool
    fact_ok: bool
    sandbox_ok: Optional[bool]
    confidence: float
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.sandbox_ok is None:
            return self.logic_ok and self.fact_ok
        return self.logic_ok and self.fact_ok and self.sandbox_ok


class Evaluator:
    def __init__(self, llm, memory: LongTermMemory):
        self.llm = llm
        self.memory = memory

    def evaluate(self, task: str, answer: str, task_type: str = "general") -> EvaluationReport:
        reasons: List[str] = []
        logic_ok = self._logic_self_consistency(task, answer, reasons)
        fact_ok = self._fact_check(task, answer, reasons)
        sandbox_ok = None
        if task_type == "code":
            sandbox_ok = self._sandbox_check(answer, reasons)
        confidence = self._calc_confidence(logic_ok, fact_ok, sandbox_ok)
        return EvaluationReport(logic_ok, fact_ok, sandbox_ok, confidence, reasons)

    def _logic_self_consistency(self, task: str, answer: str, reasons: List[str]) -> bool:
        """通过“让模型自我反驳”实现逻辑检查。"""
        prompt = (
            "你是严苛审稿人。请检查下述任务与答案是否逻辑自洽，"
            "若有问题必须明确指出“矛盾/错误”，否则写“通过”。\n"
            f"任务: {task}\n答案: {answer}"
        )
        critique = self.llm.generate(prompt=prompt, system_prompt="你负责逻辑自检。")
        bad = any(k in critique for k in ["矛盾", "错误", "不一致"])
        if bad:
            reasons.append(f"逻辑检查未通过: {critique}")
        return not bad

    def _fact_check(self, task: str, answer: str, reasons: List[str]) -> bool:
        """事实核查：优先做可计算事实检查；其次与记忆库经验规则交叉。"""
        # 先处理算术类可验证事实
        arith = re.search(r"(\d+)\s*([\+\-\*/])\s*(\d+)", task)
        if arith:
            a, op, b = int(arith.group(1)), arith.group(2), int(arith.group(3))
            if op == "/" and b == 0:
                reasons.append("事实核查失败: 任务包含除零，无法得到有效数值真值。")
                return False
            truth = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
            got_num = self._extract_first_number(answer)
            if got_num is None or abs(got_num - float(truth)) > 1e-9:
                reasons.append(f"事实核查失败: 该算式正确结果应为 {truth}，但答案为 {answer}")
                return False
            return True

        # 无外部搜索工具时，通过历史经验规则做交叉比对
        related = self.memory.search_related(task, top_k=3)
        for item in related:
            rules = item.get("metadata", {}).get("rules", [])
            for rule in rules:
                if "先计算再作答" in rule and not re.search(r"\d", answer):
                    reasons.append(f"事实核查警告: 历史规则要求数值核算，但当前答案缺少可验证数字。")
                    return False
        return True

    @staticmethod
    def _extract_first_number(text: str) -> Optional[float]:
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None

    def _sandbox_check(self, answer: str, reasons: List[str]) -> bool:
        """编程任务沙盒检查：提取Python代码并在受限参数下运行。"""
        code = self._extract_python_code(answer)
        if not code:
            reasons.append("沙盒检查失败: 未检测到可执行Python代码块。")
            return False
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=True) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    [sys.executable, "-I", f.name],  # -I: 隔离模式，降低环境污染
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
            if result.returncode != 0:
                reasons.append(f"沙盒执行报错: {result.stderr.strip() or result.stdout.strip()}")
                return False
            return True
        except subprocess.TimeoutExpired:
            reasons.append("沙盒检查失败: 代码执行超时。")
            return False
        except Exception as e:
            reasons.append(f"沙盒检查异常: {e}")
            return False

    @staticmethod
    def _extract_python_code(answer: str) -> str:
        m = re.search(r"```python\s*(.*?)```", answer, re.S | re.I)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _calc_confidence(logic_ok: bool, fact_ok: bool, sandbox_ok: Optional[bool]) -> float:
        # 简洁可解释打分：逻辑0.4 + 事实0.4 + 沙盒0.2(仅代码任务)
        score = 0.0
        score += 0.4 if logic_ok else 0.0
        score += 0.4 if fact_ok else 0.0
        if sandbox_ok is None:
            score += 0.2
        else:
            score += 0.2 if sandbox_ok else 0.0
        return round(score, 3)
