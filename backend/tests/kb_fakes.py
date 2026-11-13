"""知识库测试公用的假实现：不调外部 API，向量和打分都可预测。"""

import asyncio
import hashlib
import uuid

import chromadb

from app.kb import KnowledgeBase, open_collection

_client = chromadb.EphemeralClient()


class FakeEmbedder:
    """默认按文本 hash 生成固定向量；vectors 里指定了的文本用指定向量。记下被要求算了哪些文本。"""
    name = "fake"

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.calls: list[str] = []
        self.vectors = vectors or {}
        self.fail = False  # 置 True 模拟 API 故障（超时 / 429）
        self.gate: asyncio.Event | None = None  # 设了就卡在这里，直到测试 set()：让「处理中」状态可控

    async def embed(self, texts):
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise TimeoutError("fake embedding timeout")
        self.calls.extend(texts)
        return [self.vectors.get(t) or [b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]]
                for t in texts]


class FakeReranker:
    """文本里含某个关键词就给对应分数，否则 0。记下每次调用的 (query, docs)。"""
    name = "fake-rerank"

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = False

    async def rerank(self, query, docs):
        if self.fail:
            raise TimeoutError("fake rerank timeout")
        self.calls.append((query, docs))
        return [max((s for kw, s in self.scores.items() if kw in d), default=0.0) for d in docs]


def make_kb(embedder=None):
    emb = embedder or FakeEmbedder()
    return KnowledgeBase(open_collection(_client, f"t-{uuid.uuid4().hex}"), emb), emb
