"""main.py
演示流程：
1) 故意犯错（2+2回答为5）
2) 评估引擎发现错误并给出低置信度
3) 触发提示词进化写入JSON
4) 再次回答同题，得到正确结果4
"""

from __future__ import annotations

from pathlib import Path

from agent import AGIAgent, OpenAIOrMockLLM
from evaluator import Evaluator
from evolution import PromptEvolver, ToolBootstrapper
from memory import LongTermMemory


def run_demo() -> None:
    base_dir = Path(__file__).resolve().parent
    state_dir = base_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    memory = LongTermMemory(persist_dir=str(state_dir / "memory"))
    llm = OpenAIOrMockLLM()
    evolver = PromptEvolver(state_file=str(state_dir / "prompt_state.json"))
    tools = ToolBootstrapper(registry_file=str(state_dir / "tools_registry.json"))
    evaluator = Evaluator(llm=llm, memory=memory)
    agent = AGIAgent(memory=memory, evaluator=evaluator, evolver=evolver, tool_bootstrapper=tools, llm=llm, max_attempts=2)

    task = "请直接回答：2+2等于几？只输出数字。"

    # 第一轮：会先犯错（在Mock模式下），随后通过评估发现问题并触发进化
    first = agent.solve(task)
    print("=== 第一轮结果 ===")
    print("答案:", first.answer)
    print("置信度:", first.report.confidence)
    print("评估原因:", first.report.reasons)
    print("当前System Prompt:\n", evolver.get_system_prompt())

    # 第二轮：同题复测，预期通过进化后的提示词修正答案
    second = agent.solve(task)
    print("\n=== 第二轮结果（进化后） ===")
    print("答案:", second.answer)
    print("置信度:", second.report.confidence)
    print("评估原因:", second.report.reasons)
    print("当前System Prompt:\n", evolver.get_system_prompt())

    # 展示经验检索结果，证明长期记忆已写入并可回放
    related = memory.search_related("2+2", top_k=2)
    print("\n=== 长期记忆检索样例 ===")
    for idx, item in enumerate(related, start=1):
        print(f"[{idx}] 相似度={item.get('score', 0):.3f}; 规则={item.get('metadata', {}).get('rules', [])}")


if __name__ == "__main__":
    run_demo()

