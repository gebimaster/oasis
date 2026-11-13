"""知识库（M6）存储层：embedding 客户端 + Chroma 入库 + 两路召回。融合与重排在 kb_search.py。

embedding / rerank 走 SiliconFlow（用户选 C）。各包一层自己的接口，测试注入假实现，
以后涉密场景换本地 ONNX 也只换实现。接口细节读的是 docs.siliconflow.com/cn 的
embeddings / rerank 两页：input 数组每次最多 32 条，bge-m3 单条最多 8192 token。
chromadb 1.5.9 是同步库，这里统一用 asyncio.to_thread 包一层，不阻塞事件循环。
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
import httpx
from rank_bm25 import BM25L

from app.kb_chunk import Chunk, chunk_id, embed_text, split_chunks, tokenize

EMBED_MODEL = "BAAI/bge-m3"
EMBED_BATCH = 32
COLLECTION = "kb_chunks"


class Embedder(Protocol):
    name: str
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SiliconFlowEmbedder:
    name = EMBED_MODEL

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 30):
        self.base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.api_key = api_key or os.environ["SILICONFLOW_API_KEY"]
        self.timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), EMBED_BATCH):
                r = await client.post(f"{self.base_url}/embeddings",
                                      headers={"Authorization": f"Bearer {self.api_key}"},
                                      json={"model": self.name, "input": texts[i:i + EMBED_BATCH]})
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])  # 按 index 对回输入顺序
                out.extend(d["embedding"] for d in data)
        return out


def open_collection(client=None, name: str = COLLECTION):
    """embedding_function=None：向量全由我们传入，Chroma 不会去下载它自带的英文模型（实测
    没有 ~/.cache/chroma 目录产生；不传向量的 add 会直接报错）。距离用 cosine，distance = 1 - cos。"""
    if client is None:
        path = os.getenv("KB_DIR", str(Path(__file__).parent.parent / "data" / "chroma"))
        client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name, embedding_function=None,
                                           configuration={"hnsw": {"space": "cosine"}})


@dataclass
class Hit:
    id: str
    text: str
    doc_id: str
    title: str
    chunk_index: int
    score: float  # 这一路自己的分数（cos 或 BM25），两路量纲不同，不能直接比
    section: str = ""    # O5：所在小节标题
    para_index: int = 0  # O5：原文第几段（引用「第N段」用它）


def _hit(id_, text, meta, score) -> Hit:
    # O5 之前入库的块没有 section_title / para_index，退回旧含义（chunk_index），重灌一次就补上
    return Hit(id_, text, meta["doc_id"], meta["title"], meta["chunk_index"], score,
               meta.get("section_title", ""), meta.get("para_index", meta["chunk_index"]))


class KnowledgeBase:
    def __init__(self, collection, embedder: Embedder):
        self.col, self.embedder = collection, embedder

    async def ingest(self, user_id: str, doc_id: str, text: str, title: str | None = None) -> dict:
        """幂等入库：同一 (user_id, doc_id) 重复上传只算差异。返回 {added, deleted, kept}。"""
        title = title or doc_id
        new: dict[str, tuple[int, Chunk]] = {}
        for i, c in enumerate(split_chunks(text)):
            # id 按「实际拿去 embedding 的文本」算：标题也进了向量和词表，改标题必须重算（接 PUT 时发现的坑）。
            # O5 起小节标题也在里面：改小节标题只重算这一节的块
            new.setdefault(chunk_id(user_id, doc_id, embed_text(title, c.section, c.text)), (i, c))  # 重复块只留一份
        where = {"$and": [{"user_id": user_id}, {"doc_id": doc_id}]}
        old = set((await asyncio.to_thread(self.col.get, where=where, include=[]))["ids"])

        to_add = [k for k in new if k not in old]
        to_del = [k for k in old if k not in new]
        kept = [k for k in new if k in old]
        meta = lambda k: {"user_id": user_id, "doc_id": doc_id, "title": title, "chunk_index": new[k][0],
                          "section_title": new[k][1].section, "para_index": new[k][1].para_index}
        ctx = lambda k: embed_text(title, new[k][1].section, new[k][1].text)
        if to_add:
            # contextual embedding：向量用「标题 + 小节 + 块」算，存的原文仍是块本身
            vecs = await self.embedder.embed([ctx(k) for k in to_add])
            # 用户选 C：分词在入库时做一次存进元数据（实测分词占现场建 BM25 耗时的九成），
            # 词表跟块同生同灭，没有缓存失效问题
            metas = [{**meta(k), "tokens": " ".join(tokenize(ctx(k)))} for k in to_add]
            await asyncio.to_thread(self.col.add, ids=to_add, embeddings=vecs,
                                    documents=[new[k][1].text for k in to_add], metadatas=metas)
        if kept:  # 内容没变但位置可能变了：只更新元数据（Chroma 的 update 是合并，tokens 保留），不重算向量
            await asyncio.to_thread(self.col.update, ids=kept, metadatas=[meta(k) for k in kept])
        if to_del:  # 先加后删：中途崩溃最多短暂多出旧块，重跑一次即可收敛，不会出现「新旧都没有」
            await asyncio.to_thread(self.col.delete, ids=to_del)
        return {"added": len(to_add), "deleted": len(to_del), "kept": len(kept)}

    async def delete_document(self, user_id: str, doc_id: str) -> None:
        await asyncio.to_thread(self.col.delete,
                                where={"$and": [{"user_id": user_id}, {"doc_id": doc_id}]})

    # ---- 第 2 步：两路召回。两路都先按 user_id 过滤，再算分 ----

    async def vector_search(self, user_id: str, query: str, k: int) -> list[Hit]:
        """向量一路：Chroma 在 where 过滤后的集合里找最近邻。score = 1 - distance = cos。"""
        [qv] = await self.embedder.embed([query])
        r = await asyncio.to_thread(self.col.query, query_embeddings=[qv], n_results=k,
                                    where={"user_id": user_id})
        return [_hit(i, d, m, 1 - dist) for i, d, m, dist in
                zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    async def keyword_search(self, user_id: str, query: str, k: int) -> list[Hit]:
        """关键词一路：只拿这个用户的块建 BM25（用户答②：先过滤）。如果全库建，B 的文档会改变
        idf，A 的分数和排序就会随 B 的库变化，等于从分数里漏出别人库里有什么词。"""
        r = await asyncio.to_thread(self.col.get, where={"user_id": user_id},
                                    include=["documents", "metadatas"])
        q = tokenize(query)
        if not r["ids"] or not q:
            return []
        # BM25L 而不是 BM25Okapi：Okapi 的 idf=log(N-n+0.5)-log(n+0.5)，n=N/2 时为 0 且不补下限
        # （rank_bm25.py:96-105），小语料里命中的词可能一分不得；BM25L 的 idf=log(N+1)-log(n+0.5) 恒正
        bm = BM25L([m["tokens"].split() for m in r["metadatas"]])
        scores = bm.get_scores(q)
        order = sorted((i for i in range(len(scores)) if scores[i] > 0),  # 一个词都没命中的块不进排名
                       key=lambda i: -scores[i])[:k]
        return [_hit(r["ids"][i], r["documents"][i], r["metadatas"][i], float(scores[i])) for i in order]
