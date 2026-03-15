import os
from dataclasses import dataclass, field
from typing import (
    TypedDict,
    Union,
    Literal,
    Generic,
    TypeVar,
    Optional,
    Dict,
    Any,
    List,
)
from enum import Enum

import numpy as np

from .utils import EmbeddingFunc


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")

#QueryParam 类包含了一系列与查询操作相关的参数，这些参数可以控制查询的模式、响应类型、结果数量、上下文长度等。
@dataclass
class QueryParam:
    #指定查询的模式
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    # Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode.
    #整数类型，指定要检索的前 k 个项目的数量。它从环境变量 "TOP_K" 中获取值，如果环境变量未设置，则使用默认值 60。
    #在 “局部” 模式下对应于实体，在 “全局” 模式下对应于关系。
    top_k: int = int(os.getenv("TOP_K", "60"))
    # Number of document chunks to retrieve.
    # top_n: int = 10
    # Number of tokens for the original chunks.
    #整数类型，指定全局上下文的最大令牌数，默认值为 4000。
    #原始分块的标记数量
    max_token_for_text_unit: int = 4000
    #关系描述的标记数量。
    # Number of tokens for the relationship descriptions
    max_token_for_global_context: int = 4000
    #实体描述的标记数量。
    # Number of tokens for the entity descriptions
    max_token_for_local_context: int = 4000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)
    # Conversation history support
    conversation_history: list[dict] = field(
        default_factory=list
    )  # Format: [{"role": "user/assistant", "content": "message"}]
    history_turns: int = (
        3  # Number of complete conversation turns (user-assistant pairs) to consider
    )

#存储命名空间名称和全局配置信息
@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    #用于在索引操作完成后执行存储相关的提交操作
    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    #用于在查询操作完成后执行存储相关的提交操作
    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass

#BaseVectorStorage 类主要用于表示向量存储的基础结构，提供了一些与向量存储操作相关的属性和方法。
@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

#
@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    #该方法用于获取存储中的所有键，返回一个字符串列表。
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    #批量根据键列表 ids 获取对应的值，还可以指定要获取的字段 fields。
    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    #批量根据键列表 ids 获取对应的值，还可以指定要获取的字段 fields
    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError


    #过滤出给定键列表 data 中在存储里不存在的键，返回一个字符串集合。
    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    #插入或更新数据，接受一个字典 data，字典的键是字符串，值是泛型 T 类型的数据。
    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    #删除整个存储，具体实现需在子类中完成。
    async def drop(self):
        raise NotImplementedError


#抽象基类，用于表示图存储的基本结构和操作，提供了一系列用于操作图数据的异步方法，如检查节点和边是否存在、获取节点和边的信息、插入或更新节点和边、删除节点、对节点进行嵌入处理以及获取知识图谱等。
@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    #检查指定 node_id 的节点是否存在于图中，返回一个布尔值
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    #用于检查从 source_node_id 到 target_node_id 的边是否存在于图中，返回一个布尔值。
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    #用于计算指定 node_id 的节点的度（即与该节点相连的边的数量），返回一个整数。
    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    #用于计算从 src_id 到 tgt_id 的边的度（这里的度可能有特定的定义，需要根据具体情况实现），返回一个整数。
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError


    #用于获取指定 node_id 的节点的信息，返回一个字典，如果节点不存在则返回 None。
    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    #用于获取从 source_node_id 到 target_node_id 的边的信息，返回一个字典，如果边不存在则返回 None。
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    #用于获取与指定 source_node_id 节点相连的所有边的信息，返回一个包含元组的列表，每个元组表示一条边的源节点和目标节点，如果没有边则返回 None。
    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    #用于对图中的节点进行嵌入处理，接受一个嵌入算法名称 algorithm，返回一个包含嵌入向量数组和节点 ID 列表的元组。
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in rag.")

    #用于获取图中所有节点的标签，返回一个字符串列表。
    async def get_all_labels(self) -> List[str]:
        raise NotImplementedError

    #用于获取以指定 node_label 为起点，最大深度为 max_depth 的知识图谱，返回一个字典，字典的键是节点 ID，值是包含节点信息的字典列表。
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> Dict[str, List[Dict]]:
        raise NotImplementedError


#用于表示文档处理的不同状态。枚举类是一种特殊的类，它的实例是一组固定的常量，每个常量都有一个唯一的名称和对应的值。在这个例子中，DocStatus 枚举类包含了四种文档处理状态：PENDING（待处理）、PROCESSING（处理中）、PROCESSED（已处理）和 FAILED（处理失败）。
class DocStatus(str, Enum):
    """Document processing status enum"""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    #字符串类型，存储文档内容的前 100 个字符作为摘要。
    content_summary: str  # First 100 chars of document content
    #整数类型，记录文档的总长度。
    content_length: int  # Total length of document
    status: DocStatus  # Current processing status
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    #表示文档分块后的数量
    chunks_count: Optional[int] = None  # Number of chunks after splitting
    error: Optional[str] = None  # Error message if failed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class DocStatusStorage(BaseKVStorage):
    """Base class for document status storage"""

    async def get_status_counts(self) -> Dict[str, int]:
        """Get counts of documents in each status"""
        raise NotImplementedError

    async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        raise NotImplementedError

    async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        raise NotImplementedError
