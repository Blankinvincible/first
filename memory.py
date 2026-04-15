"""memory.py
实现智能体的动态记忆系统：
1) 短期记忆：保存当前会话上下文
2) 长期记忆：向量化经验存储与检索
3) 经验回放：把问题、尝试过程、结果、规则写入长期记忆
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional


@dataclass
class Experience:
    """一条可回放经验。"""

    question: str
    attempts: List[str]
    final_result: str
    rules: List[str]
    timestamp: str


class ShortTermMemory:
    """短期记忆：保存最近N轮上下文。"""

    def __init__(self, max_turns: int = 8):
        self._messages: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def dump(self) -> List[Dict[str, str]]:
        return list(self._messages)


class SimpleEmbedder:
    """纯Python嵌入器：将文本哈希到固定向量，避免额外依赖。"""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0 for _ in range(self.dim)]
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class LongTermMemory:
    """长期记忆：优先使用 ChromaDB；不可用时回退到本地JSON向量库。"""

    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SimpleEmbedder()
        self._json_db_path = self.persist_dir / "experience_store.json"
        self._json_db = self._load_json_db()
        self._chroma = None
        self._collection = None
        self._try_init_chroma()

    def _try_init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            self._chroma = chromadb.PersistentClient(path=str(self.persist_dir / "chroma"))
            self._collection = self._chroma.get_or_create_collection("agi_experiences")
        except Exception:
            self._chroma = None
            self._collection = None

    def _load_json_db(self) -> List[Dict]:
        if not self._json_db_path.exists():
            return []
        with self._json_db_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json_db(self) -> None:
        with self._json_db_path.open("w", encoding="utf-8") as f:
            json.dump(self._json_db, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def add_experience(self, exp: Experience) -> str:
        """写入长期记忆。"""
        exp_id = str(uuid.uuid4())
        doc = (
            f"问题: {exp.question}\n"
            f"尝试: {' | '.join(exp.attempts)}\n"
            f"结果: {exp.final_result}\n"
            f"规则: {'; '.join(exp.rules)}\n"
            f"时间: {exp.timestamp}"
        )
        embedding = self.embedder.embed(doc)
        metadata = asdict(exp)

        if self._collection is not None:
            self._collection.add(
                ids=[exp_id],
                documents=[doc],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        else:
            self._json_db.append(
                {"id": exp_id, "document": doc, "embedding": embedding, "metadata": metadata}
            )
            self._save_json_db()
        return exp_id

    def search_related(self, query: str, top_k: int = 3) -> List[Dict]:
        """在新问题到来时先检索历史经验。"""
        if top_k <= 0:
            return []
        q_emb = self.embedder.embed(query)

        if self._collection is not None:
            result = self._collection.query(
                query_embeddings=[q_emb], n_results=top_k, include=["metadatas", "documents", "distances"]
            )
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]
            out = []
            for doc, meta, dist in zip(docs, metas, dists):
                out.append({"document": doc, "metadata": meta, "score": 1 - float(dist)})
            return out

        scored = []
        for item in self._json_db:
            score = self._cosine(q_emb, item["embedding"])
            scored.append({"document": item["document"], "metadata": item["metadata"], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def replay_experience(
        self, question: str, attempts: List[str], final_result: str, rules: Optional[List[str]] = None
    ) -> str:
        """经验回放入口：解决任务后沉淀经验。"""
        derived_rules = rules or self._derive_rules(attempts, final_result)
        exp = Experience(
            question=question,
            attempts=attempts,
            final_result=final_result,
            rules=derived_rules,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return self.add_experience(exp)

    @staticmethod
    def _derive_rules(attempts: List[str], final_result: str) -> List[str]:
        """从过程自动提炼规则。"""
        rules = []
        joined = " ".join(attempts + [final_result])
        if re.search(r"\d+\s*[\+\-\*/]\s*\d+", joined):
            rules.append("遇到算术问题时，必须先计算再作答。")
        if "报错" in joined or "error" in joined.lower():
            rules.append("编程任务先在沙盒执行验证再输出最终答案。")
        if not rules:
            rules.append("回答前进行逻辑自检与事实核对。")
        return rules

