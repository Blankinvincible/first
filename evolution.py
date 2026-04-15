"""evolution.py
实现“自我进化”机制：
1) 元提示词动态更新（根据失败经验修正System Prompt）
2) 工具自举（动态注册新的Python函数工具）
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Callable, Dict, List


class PromptEvolver:
    """负责维护可进化System Prompt，并持久化到本地JSON。"""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        default = {
            "system_prompt": "你是谨慎的智能体。回答前进行自检，但当前规则较少。",
            "lessons": [],
        }
        self._save(default)
        return default

    def _save(self, state: Dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_system_prompt(self) -> str:
        return self.state["system_prompt"]

    def evolve_from_failure(self, task: str, answer: str, reasons: List[str]) -> None:
        """失败后自动提炼经验并强化系统提示词，减少重复犯错。"""
        lesson = f"任务[{task}]中答案[{answer}]失败原因: {' | '.join(reasons)}"
        if lesson not in self.state["lessons"]:
            self.state["lessons"].append(lesson)

        add_rules = []
        reason_text = " ".join(reasons)
        if "算式" in reason_text or "先计算" in reason_text or "结果应为" in reason_text:
            add_rules.append("遇到算术题，必须先计算并复核结果后再回答。")
        if "逻辑检查未通过" in reason_text:
            add_rules.append("每次回答前先进行自我反驳，确保逻辑自洽。")
        if "沙盒" in reason_text:
            add_rules.append("涉及代码时，必须先执行沙盒验证后再输出最终结论。")
        if not add_rules:
            add_rules.append("回答前必须执行逻辑与事实双重校验。")

        # 只追加缺失规则，保证提示词持续进化但不重复冗余
        prompt = self.state["system_prompt"]
        for r in add_rules:
            if r not in prompt:
                prompt += f"\n- {r}"
        self.state["system_prompt"] = prompt
        self._save(self.state)


class ToolBootstrapper:
    """工具自举：让智能体把新函数注册到工具库，下次可直接调用。"""

    def __init__(self, registry_file: str):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.tools: Dict[str, Callable[..., Any]] = {}
        self._tool_code: Dict[str, str] = self._load_code_registry()
        self._rehydrate_tools()

    def _load_code_registry(self) -> Dict[str, str]:
        if self.registry_file.exists():
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        self.registry_file.write_text("{}", encoding="utf-8")
        return {}

    def _save_code_registry(self) -> None:
        self.registry_file.write_text(json.dumps(self._tool_code, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rehydrate_tools(self) -> None:
        for _, code in self._tool_code.items():
            self._load_single_tool(code)

    def register_tool_from_code(self, function_code: str) -> str:
        """注册新工具函数；只允许单个函数定义，阻断危险语法。"""
        func_name = self._validate_tool_code(function_code)
        self._load_single_tool(function_code)
        self._tool_code[func_name] = function_code
        self._save_code_registry()
        return func_name

    def _validate_tool_code(self, code: str) -> str:
        tree = ast.parse(code)
        body = tree.body
        if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
            raise ValueError("工具代码必须且只能包含一个函数定义。")
        fn = body[0]
        forbidden = (
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.ClassDef,
            ast.With,
            ast.Try,
            ast.While,
            ast.For,
            ast.AsyncFor,
            ast.Lambda,
            ast.Attribute,
            ast.Raise,
            ast.Delete,
            ast.Yield,
            ast.YieldFrom,
        )
        allowed_calls = {
            "abs",
            "min",
            "max",
            "sum",
            "len",
            "round",
            "sorted",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "range",
        }
        for node in ast.walk(fn):
            if isinstance(node, forbidden):
                raise ValueError("工具代码包含被禁止语法（import/class/try/with等）。")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("工具代码调用方式不安全（仅允许白名单内置函数）。")
                # 阻断exec/eval/__import__及其他非白名单调用，也阻断递归导致资源耗尽
                if node.func.id == fn.name:
                    raise ValueError("工具代码不允许递归调用，避免资源耗尽。")
                if node.func.id not in allowed_calls:
                    raise ValueError(f"工具代码调用了不在白名单中的函数: {node.func.id}")
        return fn.name

    def _load_single_tool(self, code: str) -> None:
        local_ns: Dict[str, Any] = {}
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "round": round,
            "sorted": sorted,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
        }
        exec(code, {"__builtins__": safe_builtins}, local_ns)
        for k, v in local_ns.items():
            if callable(v):
                self.tools[k] = v

    def call_tool(self, name: str, *args, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"工具不存在: {name}")
        return self.tools[name](*args, **kwargs)
