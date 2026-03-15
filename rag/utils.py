import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List, Optional
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

from rag.prompt_test3 import PROMPTS


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


ENCODER = None

statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}

logger = logging.getLogger("rag")

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


#配置一个日志记录器，使其将日志信息以指定的格式写入到指定的文件中。
def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

#EmbeddingFunc 类封装了嵌入操作的相关配置和功能，通过信号量控制并发执行的数量，使得可以安全地并发调用嵌入函数。
@dataclass
class EmbeddingFunc:
    #嵌入向量的维度
    embedding_dim: int
    #最大的令牌（token）数量
    max_token_size: int
    #实际执行嵌入操作的函数
    func: callable
    #同时可以执行的嵌入操作的最大数量
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            return await self.func(*args, **kwargs)

#该函数的主要目的是从给定的字符串中尝试提取并处理可能的 JSON 字符串，如果提取成功则返回处理后的字符串，否则返回 None。
def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            maybe_json_str = maybe_json_str.replace("'", '"')
            # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
            return maybe_json_str
    except Exception:
        pass
        # try:
        #     content = (
        #         content.replace(kw_prompt[:-1], "")
        #         .replace("user", "")
        #         .replace("model", "")
        #         .strip()
        #     )
        #     maybe_json_str = "{" + content.split("{")[1].split("}")[0] + "}"
        #     json.loads(maybe_json_str)

        return None

#从一个字符串响应中提取 JSON 字符串并将其解析为 Python 字典
def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None

#根据传入的参数计算一个哈希值。该哈希值通常用于缓存机制，以唯一标识一组特定的参数组合，方便后续检查是否已经处理过相同的参数。
def compute_args_hash(*args, cache_type: str = None) -> str:
    """Compute a hash for the given arguments.
    Args:
        *args: Arguments to hash
        cache_type: Type of cache (e.g., 'keywords', 'query')
    Returns:
        str: Hash string
    """
    import hashlib

    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])
    if cache_type:
        args_str = f"{cache_type}:{args_str}"

    # Compute MD5 hash
    return hashlib.md5(args_str.encode()).hexdigest()


#根据输入的内容计算 MD5 哈希值，并可以为这个哈希值添加一个前缀，最终返回一个带有可选前缀的 MD5 哈希字符串
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


#为异步函数添加一个并发调用限制。也就是说，它可以确保在同一时间内，被装饰的异步函数的最大并发调用次数不超过指定的数量 max_size。
def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


#是给一个函数包装额外的属性，最终返回一个新的 EmbeddingFunc 类型的对象。
def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


#尝试从指定的 JSON 文件中读取数据，并将其解析为 Python 对象。如果指定的文件不存在，函数将返回 None。
def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)

#将一个 Python 对象（通常是字典或列表）以 JSON 格式写入到指定的文件中。
def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


#使用 tiktoken 库将输入的字符串 content 编码为一系列的标记（tokens）。tiktoken 是 OpenAI 开发的一个快速 BPE（字节对编码）分词器，常用于处理与大语言模型（如 GPT 系列）相关的文本编码任务。该函数会根据指定的模型名称 model_name 来获取对应的编码器，并对输入的字符串进行编码。
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

#使用 tiktoken 库将一系列的标记（tokens）解码为原始的文本字符串。tiktoken 是 OpenAI 开发的用于处理文本标记化的库，常用于大语言模型（如 GPT 系列）相关的文本处理。
def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


#是将一系列的文本内容按照交替的 user 和 assistant 角色封装成符合 OpenAI API 消息格式的列表。
def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]

#将一个字符串 content 按照多个分隔标记 markers 进行分割。它会处理多个不同的分隔符，最终返回分割后的非空且去除首尾空格的字符串列表。
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


#对输入的字符串进行清理操作，去除其中的 HTML 转义字符、控制字符以及其他不需要的字符。如果输入不是字符串类型，则直接返回输入本身。
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


#使用正则表达式来判断一个输入的值是否为有效的浮点数表示。如果是有效的浮点数表示，函数返回 True；否则返回 False。
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


#根据最大标记（token）数量对输入的列表进行截断处理。它会遍历列表中的每个元素，通过指定的 key 函数获取元素的特定部分，使用 encode_string_by_tiktoken 函数将其编码为标记，统计标记数量，当标记总数超过 max_token_size 时，返回截断后的列表。
def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

#将一个嵌套列表（即列表的列表）形式的数据转换为 CSV（逗号分隔值）格式的字符串。CSV 是一种常见的文本格式，用于存储表格数据，数据的每一行由逗号分隔的字段组成，每行数据用换行符分隔。
def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(
        output,
        quoting=csv.QUOTE_ALL,  # Quote all fields
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
        lineterminator="\n",  # Explicit line terminator
    )
    writer.writerows(data)
    return output.getvalue()


#将一个 CSV 格式的字符串解析为一个嵌套列表，其中每个内部列表代表 CSV 文件中的一行，列表中的元素则对应每行中的各个字段。
def csv_string_to_list(csv_string: str) -> List[List[str]]:
    # Clean the string by removing NUL characters
    cleaned_string = csv_string.replace("\0", "")

    output = io.StringIO(cleaned_string)
    reader = csv.reader(
        output,
        quoting=csv.QUOTE_ALL,  # Match the writer configuration
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
    )

    try:
        return [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV string: {str(e)}")
    finally:
        output.close()


#将 Python 对象（通常是字典、列表等可 JSON 序列化的数据）保存到指定的文件中，文件以 JSON 格式存储。JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，易于人类阅读和编写，同时也易于机器解析和生成。
def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


#将 XML 文件解析为一个特定结构的 Python 字典，该字典包含 nodes 和 edges 两个列表，分别存储 XML 文件中的节点和边信息，最后返回这个字典。
def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#处理两个 CSV 格式的字符串 hl 和 ll，将它们解析、合并，并生成一个新的 CSV 格式字符串。具体步骤包括提取表头、去除重复的数据行、重新编号并生成新的 CSV 内容。
def process_combine_contexts(hl, ll):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result


#异步函数，其主要功能是从缓存中查找与当前嵌入向量（current_embedding）最相似的缓存条目，并根据相似度阈值和可选的大语言模型（LLM）检查来确定是否返回缓存的响应。如果找到合适的缓存条目，函数会记录相关日志信息并返回缓存的响应；否则返回 None。
async def get_best_cached_response(
        hashing_kv,
        current_embedding,
        #相似度阈值，默认为 0.95
        similarity_threshold=0.95,
        mode="default",
        use_llm_check=False,
        llm_func=None,
        original_prompt=None,
        cache_type=None,
) -> Union[str, None]:
    mode_cache = await hashing_kv.get_by_id(mode)
    if not mode_cache:
        return None

    best_similarity = -1
    best_response = None
    best_prompt = None
    best_cache_id = None

    # Only iterate through cache entries for this mode
    for cache_id, cache_data in mode_cache.items():
        # Skip if cache_type doesn't match
        if cache_type and cache_data.get("cache_type") != cache_type:
            continue

        if cache_data["embedding"] is None:
            continue

        # Convert cached embedding list to ndarray
        cached_quantized = np.frombuffer(
            bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
        ).reshape(cache_data["embedding_shape"])
        cached_embedding = dequantize_embedding(
            cached_quantized,
            cache_data["embedding_min"],
            cache_data["embedding_max"],
        )

        #计算当前嵌入向量与缓存嵌入向量的余弦相似度
        #如果当前相似度大于最佳相似度，更新最佳相似度、最佳响应、最佳提示和最佳缓存 ID。
        similarity = cosine_similarity(current_embedding, cached_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_response = cache_data["return"]
            best_prompt = cache_data["original_prompt"]
            best_cache_id = cache_id



    #根据相似度和 LLM 检查返回结果
    if best_similarity > similarity_threshold:
        # If LLM check is enabled and all required parameters are provided
        if use_llm_check and llm_func and original_prompt and best_prompt:
            compare_prompt = PROMPTS["similarity_check"].format(
                original_prompt=original_prompt, cached_prompt=best_prompt
            )

            try:
                llm_result = await llm_func(compare_prompt)
                llm_result = llm_result.strip()
                llm_similarity = float(llm_result)

                # Replace vector similarity with LLM similarity score
                best_similarity = llm_similarity
                if best_similarity < similarity_threshold:
                    log_data = {
                        "event": "llm_check_cache_rejected",
                        "original_question": original_prompt[:100] + "..."
                        if len(original_prompt) > 100
                        else original_prompt,
                        "cached_question": best_prompt[:100] + "..."
                        if len(best_prompt) > 100
                        else best_prompt,
                        "similarity_score": round(best_similarity, 4),
                        "threshold": similarity_threshold,
                    }
                    logger.info(json.dumps(log_data, ensure_ascii=False))
                    return None
            except Exception as e:  # Catch all possible exceptions
                logger.warning(f"LLM similarity check failed: {e}")
                return None  # Return None directly when LLM check fails

        prompt_display = (
            best_prompt[:50] + "..." if len(best_prompt) > 50 else best_prompt
        )
        log_data = {
            "event": "cache_hit",
            "mode": mode,
            "similarity": round(best_similarity, 4),
            "cache_id": best_cache_id,
            "original_prompt": prompt_display,
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))
        return best_response
    return None

#计算两个向量之间的余弦相似度。余弦相似度是一种常用的度量方法，用于衡量两个向量在方向上的相似程度，其取值范围在 -1 到 1 之间。值越接近 1 表示两个向量的方向越相似，值越接近 -1 表示两个向量的方向越相反，值为 0 表示两个向量相互垂直。
def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


#将输入的嵌入向量（通常是高维的浮点型向量）进行量化处理，将其从浮点数表示转换为指定比特数（默认为 8 比特）的整数表示。量化可以减少数据的存储空间和计算量，同时记录量化所需的关键信息（最小值、最大值），以便后续能够进行反量化操作恢复原始的嵌入向量。
def quantize_embedding(embedding: np.ndarray, bits=8) -> tuple:
    """Quantize embedding to specified bits"""
    # Calculate min/max values for reconstruction
    min_val = embedding.min()
    max_val = embedding.max()

    # Quantize to 0-255 range
    scale = (2 ** bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


#是对经过量化处理的嵌入向量进行反量化操作，将量化后的整数向量恢复为原始的浮点型嵌入向量。
def dequantize_embedding(
        quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """Restore quantized embedding"""
    scale = (max_val - min_val) / (2 ** bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


#异步函数，用于处理缓存相关的操作。它会根据不同的条件，尝试从缓存中获取与输入提示对应的响应。支持简单的缓存匹配和基于嵌入向量的缓存匹配，并且可以配置是否使用大语言模型（LLM）进行相似度检查。如果缓存命中，则返回缓存的响应；如果未命中，则返回 None 以及可能的量化嵌入向量信息。
async def handle_cache(hashing_kv, args_hash, prompt, mode="default", cache_type=None):
    """Generic cache handling function"""
    if hashing_kv is None or not hashing_kv.global_config.get("enable_llm_cache"):
        return None, None, None, None

    # For default mode, only use simple cache matching
    if mode == "default":
        if exists_func(hashing_kv, "get_by_mode_and_id"):
            mode_cache = await hashing_kv.get_by_mode_and_id(mode, args_hash) or {}
        else:
            mode_cache = await hashing_kv.get_by_id(mode) or {}
        if args_hash in mode_cache:
            return mode_cache[args_hash]["return"], None, None, None
        return None, None, None, None

    # Get embedding cache configuration
    embedding_cache_config = hashing_kv.global_config.get(
        "embedding_cache_config",
        {"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False},
    )
    is_embedding_cache_enabled = embedding_cache_config["enabled"]
    use_llm_check = embedding_cache_config.get("use_llm_check", False)

    quantized = min_val = max_val = None
    if is_embedding_cache_enabled:
        # Use embedding cache
        embedding_model_func = hashing_kv.global_config["embedding_func"]["func"]
        llm_model_func = hashing_kv.global_config.get("llm_model_func")

        current_embedding = await embedding_model_func([prompt])
        quantized, min_val, max_val = quantize_embedding(current_embedding[0])
        best_cached_response = await get_best_cached_response(
            hashing_kv,
            current_embedding[0],
            similarity_threshold=embedding_cache_config["similarity_threshold"],
            mode=mode,
            use_llm_check=use_llm_check,
            llm_func=llm_model_func if use_llm_check else None,
            original_prompt=prompt if use_llm_check else None,
            cache_type=cache_type,
        )
        if best_cached_response is not None:
            return best_cached_response, None, None, None
    else:
        # Use regular cache
        if exists_func(hashing_kv, "get_by_mode_and_id"):
            mode_cache = await hashing_kv.get_by_mode_and_id(mode, args_hash) or {}
        else:
            mode_cache = await hashing_kv.get_by_id(mode) or {}
        if args_hash in mode_cache:
            return mode_cache[args_hash]["return"], None, None, None

    return None, quantized, min_val, max_val


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    #表示量化后的嵌入向量，是一个 numpy 数组类型
    quantized: Optional[np.ndarray] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mode: str = "default"
    cache_type: str = "query"


#异步函数，用于将 CacheData 类的实例保存到指定的异步键值存储（hashing_kv）中。该函数会根据存储对象的能力，获取相应模式下的缓存数据，将新的缓存数据添加或更新到其中，最后将更新后的缓存数据写回到存储中。
async def save_to_cache(hashing_kv, cache_data: CacheData):
    if hashing_kv is None or hasattr(cache_data.content, "__aiter__"):
        return

    if exists_func(hashing_kv, "get_by_mode_and_id"):
        mode_cache = (
                await hashing_kv.get_by_mode_and_id(cache_data.mode, cache_data.args_hash)
                or {}
        )
    else:
        mode_cache = await hashing_kv.get_by_id(cache_data.mode) or {}

    mode_cache[cache_data.args_hash] = {
        "return": cache_data.content,
        "embedding": cache_data.quantized.tobytes().hex()
        if cache_data.quantized is not None
        else None,
        "embedding_shape": cache_data.quantized.shape
        if cache_data.quantized is not None
        else None,
        "embedding_min": cache_data.min_val,
        "embedding_max": cache_data.max_val,
        "original_prompt": cache_data.prompt,
    }

    await hashing_kv.upsert({cache_data.mode: mode_cache})

#是对包含 Unicode 转义序列（如 \uXXXX 形式）的字节字符串进行解码，将其中的 Unicode 转义序列替换为实际的 Unicode 字符，最终返回一个解码后的字符串。
def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


#检查一个对象是否包含指定名称的可调用方法（函数）。它接收两个参数：一个对象 obj 和一个字符串类型的函数名 func_name，返回一个布尔值，表示该对象是否存在指定名称的可调用方法。
def exists_func(obj, func_name: str) -> bool:
    """Check if a function exists in an object or not.
    :param obj:
    :param func_name:
    :return: True / False
    """
    if callable(getattr(obj, func_name, None)):
        return True
    else:
        return False

#从给定的对话历史记录中提取指定数量的完整对话轮次，并将这些轮次格式化为字符串返回。在处理过程中，会过滤掉特定的关键词提取消息，并且确保每一轮对话中用户消息在前，助手消息在后。
def get_conversation_turns(conversation_history: list[dict], num_turns: int) -> str:
    """
    Process conversation history to get the specified number of complete turns.

    Args:
        conversation_history: List of conversation messages in chronological order
        num_turns: Number of complete turns to include

    Returns:
        Formatted string of the conversation history
    """
    # Group messages into turns
    turns = []
    messages = []

    # First, filter out keyword extraction messages
    for msg in conversation_history:
        if msg["role"] == "assistant" and (
                msg["content"].startswith('{ "high_level_keywords"')
                or msg["content"].startswith("{'high_level_keywords'")
        ):
            continue
        messages.append(msg)

    # Then process messages in chronological order
    i = 0
    while i < len(messages) - 1:
        msg1 = messages[i]
        msg2 = messages[i + 1]

        # Check if we have a user-assistant or assistant-user pair
        if (msg1["role"] == "user" and msg2["role"] == "assistant") or (
                msg1["role"] == "assistant" and msg2["role"] == "user"
        ):
            # Always put user message first in the turn
            if msg1["role"] == "assistant":
                turn = [msg2, msg1]  # user, assistant
            else:
                turn = [msg1, msg2]  # user, assistant
            turns.append(turn)
        i += 2

    # Keep only the most recent num_turns
    if len(turns) > num_turns:
        turns = turns[-num_turns:]

    # Format the turns into a string
    formatted_turns = []
    for turn in turns:
        formatted_turns.extend(
            [f"user: {turn[0]['content']}", f"assistant: {turn[1]['content']}"]
        )

    return "\n".join(formatted_turns)
