import asyncio
import json
import os
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
    get_conversation_turns,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt_test import GRAPH_FIELD_SEP, PROMPTS
import time


#将输入的文本内容按照指定的规则进行分块处理。它可以根据字符进行分割，也可以直接按照令牌（token）数量进行分割，并且支持重叠部分的设置，最终返回一个包含分块信息的列表。
def chunking_by_token_size(
    content: str,
    split_by_character=None,
    split_by_character_only=False,
    #重叠部分的令牌数量，默认值为 128
    overlap_token_size=128,
    max_token_size=1024,
    #用于编码和解码的 tiktoken 模型名称，默认值为 "gpt-4o"。
    tiktoken_model="gpt-4o",
    **kwargs,
):  #调用 encode_string_by_tiktoken 函数将输入的文本内容编码为令牌列表。初始化一个空列表 results 用于存储分块结果。

    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results

#异步函数，用于处理实体或关系的描述摘要。对于给定的实体或关系名称及其描述，函数会根据描述的令牌数量决定是否需要使用大语言模型（LLM）进行摘要处理。如果描述的令牌数量少于设定的最大摘要令牌数，则直接返回原描述；否则，构造一个提示信息，调用大语言模型生成摘要并返回。
async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    #tiktoken 模型的名称，用于对文本进行编码和解码
    tiktoken_model_name = global_config["tiktoken_model_name"]
    #摘要的最大令牌数。
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    #调用 encode_string_by_tiktoken 函数将描述内容编码为令牌列表。如果令牌数量少于摘要最大令牌数，则不需要进行摘要处理，直接返回原描述。
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    #从 PROMPTS 字典中获取摘要提示模板
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    #构造一个上下文字典 context_base，包含实体名称、描述列表和语言信息。
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    #使用上下文字典填充提示模板，得到最终的提示信息 use_prompt。
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

#异步函数，其主要功能是从给定的记录属性列表中提取单个实体的相关信息。它会对输入的记录属性进行有效性检查，如果满足特定条件，则提取实体的名称、类型、描述和源 ID 等信息，并将这些信息以字典的形式返回；如果不满足条件，则返回 None。
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

#用于从给定的记录属性列表中提取单个关系的相关信息。它会对输入的记录属性进行有效性检查，如果满足特定条件，则提取关系的源节点、目标节点、描述、关键词、源 ID 和权重等信息，并将这些信息以字典的形式返回；如果不满足条件，则返回 None。
async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )

#异步函数，其主要功能是根据实体名称从知识图谱中获取已存在的节点信息，若存在则将其与新的节点数据进行合并；若不存在，则直接使用新的节点数据。合并后的数据会更新到知识图谱中，最后返回合并后的节点数据。
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    #使用 Counter 统计新节点数据和已有节点数据中所有实体类型的出现次数。对统计结果按出现次数降序排序，取出现次数最多的实体类型作为合并后的实体类型。

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    #将新节点数据和已有节点数据中的所有描述信息合并到一个集合中，去除重复项。对集合进行排序，然后使用 GRAPH_FIELD_SEP 连接成一个字符串。
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    #将新节点数据和已有节点数据中的所有源 ID 合并到一个集合中，去除重复项。使用 GRAPH_FIELD_SEP 连接成一个字符串。
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    #调用 _handle_entity_relation_summary 异步函数对合并后的描述信息进行摘要处理，以确保描述信息的长度符合要求。
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    #构造一个包含合并后实体类型、描述和源 ID 的字典 node_data。调用 knowledge_graph_inst 的 upsert_node 方法将更新后的节点数据插入或更新到知识图谱中
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data

#异步函数，其主要功能是将新的边数据与知识图谱中已存在的对应边的数据进行合并，然后将合并后的边数据更新到知识图谱中。若源节点或目标节点不存在于知识图谱中，会先创建这些节点。最后返回合并后的边数据。
async def _merge_edges_then_upsert(
    #边的源节点 ID
    src_id: str,
    #边的目标节点 ID
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    #将新边数据和已有边数据的权重相加，得到合并后的权重
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    #将新边数据和已有边数据的描述合并到一个集合中，去除重复项，排序后使用 GRAPH_FIELD_SEP 连接成一个字符串
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    #将新边数据和已有边数据的关键词合并到一个集合中，去除重复项，排序后使用 GRAPH_FIELD_SEP 连接成一个字符串。
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data

#从给定的文本块中提取实体和实体之间的关系，并将这些信息存储到知识图谱和向量数据库中。函数会利用大语言模型（LLM）进行实体和关系的提取，支持缓存机制以提高性能，同时会处理文本块的分批次处理和结果的合并。
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    llm_response_cache: BaseKVStorage = None,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    #该函数用于调用 LLM 并处理缓存。如果启用了缓存且缓存实例存在，会先检查缓存中是否有对应的结果，如果有则直接返回，否则调用 LLM 并将结果存入缓存
    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] = None
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            need_to_restore = False
            if (
                global_config["embedding_cache_config"]
                and global_config["embedding_cache_config"]["enabled"]
            ):
                new_config = global_config.copy()
                new_config["embedding_cache_config"] = None
                new_config["enable_llm_cache"] = True
                llm_response_cache.global_config = new_config
                need_to_restore = True
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache, arg_hash, _prompt, "default", cache_type="default"
            )
            if need_to_restore:
                llm_response_cache.global_config = global_config
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return
            statistic_data["llm_call"] += 1
            if history_messages:
                res: str = await use_llm_func(
                    input_text, history_messages=history_messages
                )
            else:
                res: str = await use_llm_func(input_text)
            await save_to_cache(
                llm_response_cache,
                CacheData(args_hash=arg_hash, content=res, prompt=_prompt),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, history_messages=history_messages)
        else:
            return await use_llm_func(input_text)

    #该函数用于处理单个文本块。首先构造提示信息并调用 LLM 进行实体和关系的提取，可能会进行多次提取尝试
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """ "Prpocess a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunck-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        """
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await _user_llm_func_with_cache(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, history_messages=history
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await _user_llm_func_with_cache(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.debug(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Level 2 - Extracting entities and relationships",
        unit="chunk",
        position=1,
        leave=False,
    ):
        results.append(await result)

    #合并所有文本块的实体和关系提取结果
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    logger.debug("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Level 3 - Inserting entities",
        unit="entity",
        position=2,
        leave=False,
    ):
        all_entities_data.append(await result)

    logger.debug("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Level 3 - Inserting relationships",
        unit="relationship",
        position=3,
        leave=False,
    ):
        all_relationships_data.append(await result)

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    #如果 entity_vdb 不为 None，将实体信息插入到实体向量数据库中。
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    #如果 relationships_vdb 不为 None，将关系信息插入到关系向量数据库中。
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

#异步函数，其主要功能是根据用户的查询请求，从知识图谱和向量数据库中检索相关信息，构建查询上下文，然后调用大语言模型（LLM）生成响应。同时，该函数支持缓存机制，会先检查缓存中是否有对应的查询结果，如果有则直接返回，否则进行查询处理并将结果存入缓存。
async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
    prompt: str = "",
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    #if cached_response is not None:
        #return cached_response

    #调用 extract_keywords_only 函数从查询请求中提取高级关键词 hl_keywords 和低级关键词 ll_keywords。
    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    print("高级关键词：" + ", ".join(hl_keywords))
    logger.debug(f"Low-level  keywords: {ll_keywords}")
    print("低级关键词：" + ", ".join(ll_keywords))

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords = ", ".join(hl_keywords) if hl_keywords else ""

    logger.info("Using %s mode for query processing", query_param.mode)

    #调用 _build_query_context 函数根据关键词、知识图谱、向量数据库和查询参数构建查询上下文
    # Build context
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    # 打印生成的上下文
    #print("生成的上下文内容如下：")
    #print(context)

    # 保存上下文到文件
    #if context is not None:
        # 创建 "实验" 文件夹
        #experiment_folder = os.path.join(os.getcwd(), "实验")
        #if not os.path.exists(experiment_folder):
            #os.makedirs(experiment_folder)

        # 生成文件名，体现查询模式
        #file_name = f"{query_param.mode}_context.txt"
        #file_path = os.path.join(experiment_folder, file_name)

        # 保存上下文到文件
        #with open(file_path, "w", encoding="utf-8") as f:
            #f.write(context)
        #print(f"上下文已保存到 {file_path}")

    if query_param.only_need_context:
        print("处理只需要上下文的情况，生成的上下文：")
        print(context)
        return context
    if context is None:
        print("fail_response")
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    #构建系统提示信息 sys_prompt，如果提供了自定义提示信息 prompt 则使用，否则使用 PROMPTS["rag_response"]
    sys_prompt_temp = prompt if prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    #调用 use_model_func 函数（即 LLM 函数），传入查询请求、系统提示信息和是否流式响应的标志，生成响应。对响应进行清理，去除系统提示信息、查询内容等多余部分。
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    #保存结果到缓存并返回
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response

#使用大语言模型（LLM）从给定的文本中提取高级和低级关键词。该函数支持缓存机制，会先检查缓存中是否有对应的关键词提取结果，如果有则直接返回，否则进行关键词提取并将结果存入缓存。
async def extract_keywords_only(
    #需要提取关键词的文本
    text: str,
    param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = PROMPTS["keywords_extraction"].format(
        query=text, examples=examples, language=language, history=history_context
    )

    #调用 LLM 函数，传入关键词提取提示信息和 keyword_extraction=True 标志，进行关键词提取。
    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the result.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    print(hl_keywords)
    ll_keywords = keywords_data.get("low_level_keywords", [])
    print(ll_keywords)

    # 7. Cache only the processed keywords with cache type
    cache_data = {"high_level_keywords": hl_keywords, "low_level_keywords": ll_keywords}
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=json.dumps(cache_data),
            prompt=text,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=param.mode,
            cache_type="keywords",
        ),
    )
    return hl_keywords, ll_keywords

#实现了一种混合检索机制，结合了知识图谱和向量搜索来处理用户的查询请求。它会先检查缓存中是否有对应的查询结果，如果有则直接返回；否则，会并行执行知识图谱查询和向量搜索，将两者的结果合并构建混合提示信息，再调用大语言模型生成响应，并将最终结果存入缓存。
async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    """
    Hybrid retrieval implementation combining knowledge graph and vector search.

    This function performs a hybrid search by:
    1. Extracting semantic information from knowledge graph
    2. Retrieving relevant text chunks through vector similarity
    3. Combining both results for comprehensive answer generation
    """
    # 1. Cache handling
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    #并行执行知识图谱和向量搜索
    #从给定的查询中提取关键词，根据关键词的情况设置查询模式，然后利用这些关键词和查询模式从知识图谱中构建相关的上下文信息。
    # 2. Execute knowledge graph and vector searches in parallel
    async def get_kg_context():
        try:
            # Extract keywords using extract_keywords_only function which already supports conversation history
            hl_keywords, ll_keywords = await extract_keywords_only(
                query, query_param, global_config, hashing_kv
            )

            if not hl_keywords and not ll_keywords:
                logger.warning("Both high-level and low-level keywords are empty")
                return None

            # Convert keyword lists to strings
            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            # Set query mode based on available keywords
            if not ll_keywords_str and not hl_keywords_str:
                return None
            elif not ll_keywords_str:
                query_param.mode = "global"
            elif not hl_keywords_str:
                query_param.mode = "local"
            else:
                query_param.mode = "hybrid"

            # Build knowledge graph context
            #构建知识图谱上下文
            context = await _build_query_context(
                [ll_keywords_str, hl_keywords_str],
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            print("查出来的知识图谱数据：", context)
            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context: {str(e)}")
            return None

    #异步函数，其主要目的是在向量数据库中执行查询操作，考虑对话历史信息，获取与查询相关的文本块，并对这些文本块进行处理和格式化，最终返回一个包含相关文本块信息的字符串作为向量上下文。
    async def get_vector_context():
        # Consider conversation history in vector search
        augmented_query = query
        if history_context:
            augmented_query = f"{history_context}\n{query}"

        try:
            # Reduce top_k for vector search in hybrid mode since we have structured information from KG
            mix_topk = min(10, query_param.top_k)
            results = await chunks_vdb.query(augmented_query, top_k=mix_topk)
            print("向量搜索结果：", results)
            if not results:
                return None

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)
            print("对应的文本块数据：", chunks)

            valid_chunks = []
            for chunk, result in zip(chunks, results):
                if chunk is not None and "content" in chunk:
                    # Merge chunk content and time metadata
                    chunk_with_time = {
                        "content": chunk["content"],
                        "created_at": result.get("created_at", None),
                    }
                    valid_chunks.append(chunk_with_time)

            if not valid_chunks:
                return None

            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
            )

            if not maybe_trun_chunks:
                return None

            # Include time information in content
            formatted_chunks = []
            for c in maybe_trun_chunks:
                chunk_text = c["content"]
                if c["created_at"]:
                    chunk_text = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['created_at']))}]\n{chunk_text}"
                formatted_chunks.append(chunk_text)

            logger.info(f"Truncate {len(chunks)} to {len(formatted_chunks)} chunks")
            return "\n--New Chunk--\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error in get_vector_context: {e}")
            return None

    # 3. Execute both retrievals in parallel
    #并行获取知识图谱上下文（kg_context）和向量搜索上下文（vector_context），然后合并这两个上下文信息，构建混合提示信息，调用大语言模型生成响应，并对响应进行清理和缓存处理。
    print("开始执行协程获取知识图谱和向量数据")
    kg_context, vector_context = await asyncio.gather(
        get_kg_context(), get_vector_context()
    )

    # 4. Merge contexts
    if kg_context is None and vector_context is None:
        print("fail_response_Merge contexts")
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = PROMPTS["mix_rag_response"].format(
        kg_context=kg_context
        if kg_context
        else "No relevant knowledge graph information found",
        vector_context=vector_context
        if vector_context
        else "No relevant text information found",
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # 清理响应内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - 只有在收集完整响应后才缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode="mix",
                cache_type="query",
            ),
        )

    return response


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # 提取低级和高级关键词
    ll_keywords, hl_keywords = query[0], query[1]

    # 局部模式（local）：调用 _get_node_data 函数，根据 ll_keywords 从知识图谱和向量数据库中获取相关的实体、关系和文本单元上下文信息。
    if query_param.mode == "local":
        #entities_context, relations_context, text_units_context = await _get_node_data(
        text_units_context=await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    # 全局模式（global）：调用 _get_edge_data 函数，根据 hl_keywords 从知识图谱和向量数据库中获取相关的上下文信息
    elif query_param.mode == "global":
        #entities_context, relations_context, text_units_context = await _get_edge_data(
        text_units_context=await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    # 混合模式（hybrid）：使用 asyncio.gather 并发地调用 _get_node_data 和 _get_edge_data 函数，分别获取低级和高级的上下文信息。然后调用 combine_contexts 函数将这些信息进行合并。
    else:  # hybrid mode
        ll_data, hl_data = await asyncio.gather(
            _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        #(
            #ll_entities_context,
            #ll_relations_context,
            #ll_text_units_context,
        #) = ll_data

        #(
            #hl_entities_context,
            #hl_relations_context,
            #hl_text_units_context,
        #) = hl_data

        ll_text_units_context = ll_data
        hl_text_units_context = hl_data

        #entities_context, relations_context, text_units_context = combine_contexts(
        text_units_context=combine_contexts(
            #[hl_entities_context, ll_entities_context],
            #[hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )

    # 根据模式确定模式字符串和文件名
    if query_param.mode == "local":
        mode_str = "局部模式"
        file_name = "local_mode_context.txt"
    elif query_param.mode == "global":
        mode_str = "全局模式"
        file_name = "global_mode_context.txt"
    else:  # hybrid mode
        mode_str = "混合模式"
        file_name = "hybrid_mode_context.txt"

    # 打印各上下文信息
    #print(f"{mode_str}下获取的实体上下文：")
    #print(entities_context)
    #print(f"{mode_str}下获取的关系上下文：")
    #print(relations_context)
    #print(f"{mode_str}下获取的文本单元上下文：")
    #print(text_units_context)

        # 准备保存上下文到文件
    experiment_folder = os.path.join(os.getcwd(), "实验")
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    file_path = os.path.join(experiment_folder, file_name)

        # 保存有效文本块内容到文件，使用追加模式并添加分隔标识
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("参考上下文\n")
        #f.write(f"-----Entities-----\n```csv\n{entities_context}\n```\n")
        #f.write(f"-----Relationships-----\n```csv\n{relations_context}\n```\n")
        f.write(f"-----Text_units-----\n```csv\n{text_units_context}\n```\n")
        f.write("参考上下文\n")
    print(f"上下文已保存到 {file_path}")

    # not necessary to use LLM to generate a response
    #if not entities_context.strip() and not relations_context.strip():
        #return None

    return f"""
        -----Text_units-----
        ```csv
        {text_units_context}
    """

#异步函数，主要用于根据给定的查询词从知识图谱和相关数据库中获取与实体相关的数据，包括实体信息、实体之间的关系以及相关的文本块信息，并将这些信息整理成 CSV 格式的上下文数据返回。这些上下文数据可用于后续的查询处理，例如为大语言模型提供更丰富的信息
async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    # get similar entities
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""
    # get entity information
    #使用 asyncio.gather 并发地执行两个任务：
    #从知识图谱中获取每个查询结果对应的实体数据。
    #获取每个实体的度（即与该实体相连的边的数量）。
    #检查是否有实体数据缺失，如果有则记录警告信息。
    #过滤掉缺失的实体数据，并将实体名称和度信息添加到实体数据中
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    #"rank": d：将实体的度（d）作为 rank 信息添加到新字典中。这样，每个实体的数据字典中就多了一个表示其度的字段，方便后续使用。
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units, use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # build prompt
    #构建实体上下文
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    #entities_context = list_of_list_to_csv(entites_section_list)

    #构建关系上下文
    #如果关系数据中没有 created_at 字段，则使用 "UNKNOWN" 作为默认值；如果 created_at 是时间戳类型，则将其转换为可读的日期时间格式。
    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    #relations_context = list_of_list_to_csv(relations_section_list)

    #构建文本单元上下文
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    #return entities_context, relations_context, text_units_context
    return text_units_context

#异步函数，其主要目的是从给定的实体数据中找出与之最相关的文本单元。函数通过获取实体的文本单元信息、实体的相邻节点信息，进而获取相邻节点的文本单元信息，然后对所有文本单元进行筛选、排序和截断处理，最终返回最相关的文本单元列表。
async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    #获取实体的相邻节点信息
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    #构建相邻节点的文本单元查找表
    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }
    #构建所有文本单元的查找表
    all_text_units_lookup = {}
    tasks = []
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                tasks.append((c_id, index, this_edges))

    results = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in tasks]
    )

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1
    #筛选有效文本单元
    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    #对文本单元进行排序和截断处理
    #对 all_text_units 列表进行排序，先按实体索引排序，再按关系计数降序排序。
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


#异步函数，其主要功能是从给定的实体数据中找出与之最相关的边（关系）信息。函数会先获取所有相关实体的边，去除重复的边，然后获取这些边的详细信息和度数，对边信息进行排序，最后根据最大令牌数进行截断处理，最终返回最相关的边信息列表。
async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        ),
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    #使用 sorted 函数对 all_edges_data 列表进行排序。排序的键是一个元组 (x["rank"], x["weight"])，表示先按照边的度数（rank）排序，再按照边的权重（weight）排序。reverse=True 表示降序排序，即度数和权重较大的边排在前面。
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data

#根据给定的关键词，从关系向量数据库和知识图谱中获取与之相关的边（关系）数据、实体数据以及文本单元数据，并将这些数据整理成 CSV 格式的上下文信息返回，这些上下文信息包含实体、关系和文本单元的详细信息，可用于后续的查询处理，比如为大语言模型提供数据支持。
async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    #调用 relationships_vdb 的 query 方法，根据关键词 keywords 查找相似的关系，返回前 query_param.top_k 个结果
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        #调用 knowledge_graph_inst.get_edge 函数，传入每条边的两个端点，获取边的详细信息，存储在 all_edges_pack 中
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        #调用 knowledge_graph_inst.edge_degree 函数，传入每条边的两个端点，获取边的度数，存储在 all_edges_degree 中
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    #使用列表推导式遍历 all_edges、all_edges_pack 和 all_edges_degree，将边的端点信息存储在 src_tgt 字段，边的度数存储在 rank 字段，边的详细信息存储在字典中。过滤掉详细信息为 None 的边
    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    #使用 sorted 函数对 all_edges_data 列表进行排序。排序的键是一个元组 (x["rank"], x["weight"])，表示先按照边的度数（rank）排序，再按照边的权重（weight）排序。reverse=True 表示降序排序，即度数和权重较大的边排在前面。
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities, use_text_units = await asyncio.gather(
        #该函数的作用是从给定的关系数据（edge_datas）中找出最相关的实体。
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        #从给定的关系数据（edge_datas）中找出相关的文本单元
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    #relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    #entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    #return entities_context, relations_context, text_units_context
    return text_units_context


#异步函数，其主要功能是从给定的边数据（关系数据）中找出与之最相关的实体信息。该函数会先从边数据中提取出所有涉及的实体名称，然后并发地获取这些实体的详细信息和节点度数，最后根据最大令牌数对实体信息进行截断处理，返回处理后的实体信息列表。
async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas

#异步函数，其主要目的是从给定的边数据（关系数据）中找出与之相关的文本单元信息。函数会先从边数据中提取文本单元 ID，然后并发地从文本块数据库中获取这些文本单元的数据，接着对获取到的文本单元进行排序、过滤和截断处理，最终返回处理后的文本单元列表。
async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units

#主要功能是将高级（hl_ 前缀）和低级（ll_ 前缀）的实体、关系以及来源上下文信息分别进行合并和去重处理。该函数接收三个列表作为参数，分别代表实体、关系和来源的上下文信息，最终返回合并和去重后的实体、关系和来源上下文信息。
#def combine_contexts(entities, relationships, sources):
def combine_contexts(sources):
    # Function to extract entities, relationships, and sources from context strings
    #hl_entities, ll_entities = entities[0], entities[1]
    #hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    #combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    #combined_relationships = process_combine_contexts(
        #hl_relationships, ll_relationships
    #)

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    #return combined_entities, combined_relationships, combined_sources
    return combined_sources


#异步函数，用于执行简单的查询操作。它的主要流程包括处理缓存、从向量数据库查询相关文本块、过滤和截断文本块、构建系统提示信息、调用大语言模型生成响应，最后对响应进行处理并保存到缓存
async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
):
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "default", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # 调用 chunks_vdb 的 query 方法，根据查询内容 query 查找前 query_param.top_k 个相关的文本块。
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    #print(f"naive: {chunks}")

    # 打印每个有效文本块的内容，一行一条
    #print("参考评论：")
    #for chunk in valid_chunks:
        #content = chunk["content"]
        #print(content)

    # 创建 "实验" 文件夹
    experiment_folder = os.path.join(os.getcwd(), "实验")
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # 生成文件名，体现 naive 模式
    file_name = f"naive_mode_context.txt"
    file_path = os.path.join(experiment_folder, file_name)

    # 保存有效文本块内容到文件，使用追加模式并添加分隔标识
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("参考上下文\n")
        for chunk in valid_chunks:
            content = chunk["content"]
            f.write(content + "\n")
        f.write("参考上下文\n")
    print(f"上下文已保存到 {file_path}")

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )

    return response

#异步函数，用于基于知识图谱进行带关键词的查询。它会先检查查询结果是否有缓存，如果有则直接返回缓存结果；接着从 query_param 中提取高低级关键词，并根据关键词情况调整查询模式；然后构建查询上下文，若上下文构建失败则返回失败提示；之后根据是否只需要上下文或提示信息进行相应返回；最后构建系统提示信息并调用大语言模型生成响应，对响应进行处理后保存到缓存并返回。
async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """

    # ---------------------------
    # 1) Handle potential cache for query results
    # ---------------------------
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # ---------------------------
    # 2) RETRIEVE KEYWORDS FROM query_param
    # ---------------------------

    # If these fields don't exist, default to empty lists/strings.
    hl_keywords = getattr(query_param, "hl_keywords", []) or []
    ll_keywords = getattr(query_param, "ll_keywords", []) or []

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    # Flatten low-level and high-level keywords if needed
    ll_keywords_flat = (
        [item for sublist in ll_keywords for item in sublist]
        if any(isinstance(i, list) for i in ll_keywords)
        else ll_keywords
    )
    hl_keywords_flat = (
        [item for sublist in hl_keywords for item in sublist]
        if any(isinstance(i, list) for i in hl_keywords)
        else hl_keywords
    )

    # Join the flattened lists
    ll_keywords_str = ", ".join(ll_keywords_flat) if ll_keywords_flat else ""
    hl_keywords_str = ", ".join(hl_keywords_flat) if hl_keywords_flat else ""

    keywords = [ll_keywords_str, hl_keywords_str]


    logger.info("Using %s mode for query processing", query_param.mode)

    # ---------------------------
    # 3) BUILD CONTEXT
    # ---------------------------
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )
    if not context:
        return PROMPTS["fail_response"]

    # If only context is needed, return it
    if query_param.only_need_context:
        return context

    # ---------------------------
    # 4) BUILD THE SYSTEM PROMPT + CALL LLM
    # ---------------------------

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response
