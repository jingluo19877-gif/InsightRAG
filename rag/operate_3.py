import asyncio
import json
import os
import re

import numpy as np
import requests
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
from .prompt_test3 import GRAPH_FIELD_SEP, PROMPTS
import time


def chunking_by_token_size(
    content: str,
    split_by_character=None,
    split_by_character_only=False,
    overlap_token_size=128,
    max_token_size=1024,
    tiktoken_model="gpt-4o",
    **kwargs,
):
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
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary

#这个函数的主要作用是从输入的记录属性中提取实体信息，并进行必要的清理和验证，最后返回一个结构化的实体信息字典。如果输入不符合要求，则返回 None。
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    #entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[2])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        #entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


#这个函数的主要作用是从输入的记录属性中提取关系信息，包括源节点、目标节点、描述、关键词等，并计算关系权重，最后返回一个结构化的关系信息字典。如果输入不符合要求，则返回 None。
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
    #weight = (
        #float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    #)
    return dict(
        src_id=source,
        tgt_id=target,
        #weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )




#关键点总结：
#这是一个节点合并和更新操作，处理可能存在的多个数据源对同一实体的描述
#采用"多数表决"方式确定实体类型（出现频率最高的类型）
#描述和来源ID采用合并去重策略
#最终执行upsert操作（存在则更新，不存在则插入）
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    #already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        #already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    #entity_type = sorted(
        #Counter(
            #[dp["entity_type"] for dp in nodes_data] + already_entity_types
        #).items(),
        #key=lambda x: x[1],
        #reverse=True,
    #)[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        #entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data




async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    #already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        #already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    #weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
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
                    #"entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            #weight=weight,
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
    #entity_types = global_config["addon_params"].get(
        #"entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    #)
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
        #entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        #entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

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

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

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

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

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

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = prompt if prompt else PROMPTS["rag_response"]
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


async def extract_keywords_only(
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
    ll_keywords = keywords_data.get("low_level_keywords", [])

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


def gradient_based_chunk_selection(chunks, scores, min_k, g):
    sorted_data = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    sorted_chunks = [x[0] for x in sorted_data]
    sorted_scores = [x[1] for x in sorted_data]

    # 选择前 min_k 个文本块作为初始选择集合
    selected_chunks = sorted_chunks[:min_k]
    selected_scores = sorted_scores[:min_k]

    for i in range(min_k, len(sorted_scores)):
        prev_score = sorted_scores[i - 1]
        current_score = sorted_scores[i]
        threshold = prev_score / g

        #print(f"当前索引: {i}, 前一个分数: {prev_score}, 当前分数: {current_score}, 阈值: {threshold}")

        if current_score > threshold:
            selected_chunks.append(sorted_chunks[i])
            selected_scores.append(sorted_scores[i])
        else:
            break

    return selected_chunks, selected_scores


def perform_reranking(file_path, api_key, api_url):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 提取查询内容
        query_match = re.search(r'查询内容: (.*?)\n', content)
        if query_match:
            query = query_match.group(1)
        else:
            print("未找到查询内容，无法进行重排序。")
            return

        paragraphs = re.findall(r'"(\d+)"\s*(.*?)(?=\n\n|$)', content, re.MULTILINE)
        # 只取文本内容部分
        paragraphs = [chunk[1] for chunk in paragraphs]
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件。")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    paragraph_scores = []

    for i, paragraph in enumerate(paragraphs):
        data = {
            "model": "gte-rerank",
            "query": query,
            "top_n": 1,
            "return_documents": True,
            "documents": [paragraph]
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            score = result['data']['results'][0]['relevance_score']
            paragraph_scores.append((paragraph, score))
            print(f"段落 {i + 1} 的分数: {score}")
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP 错误发生: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"请求错误发生: {req_err}")

    chunks = [p[0] for p in paragraph_scores]
    scores = [p[1] for p in paragraph_scores]

    min_k = 30
    # 计算得分的标准差
    #std_dev = np.std(scores)
    # 可调整的倍数
    #std_multiplier = 10
    # 将梯度阈值设置为标准差的一定倍数
    # g = std_multiplier * std_dev
    g = 1.1

    #print(f"标准差: {std_dev}, 倍数: {std_multiplier}, 梯度阈值: {g}")

    selected_chunks, selected_scores = gradient_based_chunk_selection(chunks, scores, min_k, g)

    print("\n选择的段落及其分数：")
    for i, score in enumerate(selected_scores, start=1):
        print(f"段落 {i} 的分数: {score}")

    sorted_data = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    sorted_chunks = [x[0] for x in sorted_data]
    sorted_scores = [x[1] for x in sorted_data]

    print("\n重排后的段落及其分数：")
    for i, score in enumerate(sorted_scores, start=1):
        print(f"段落 {i} 的分数: {score}")

    # 保存重排序结果
    experiment_folder = os.path.dirname(file_path)
    reranked_file = os.path.join(experiment_folder, "globally_reranked_context.txt")
    with open(reranked_file, 'a', encoding='utf-8') as f:
        f.write("\n\n===== 全局重排序结果 =====\n")
        f.write(f"查询: {query}\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"原始结果数: {len(sorted_chunks)} | 筛选后: {len(selected_chunks)}\n\n")

        for i, (chunk, score) in enumerate(zip(selected_chunks, selected_scores), 1):
            f.write(f"【排名 {i} | 相关度: {score:.4f}】\n")
            f.write(f"{chunk}\n\n")


#实现了知识图谱（KG）和向量搜索的混合检索模式，用于生成更全面的查询响应。
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
    #if cached_response is not None:
        #return cached_response

    # Process conversation history
    #history_context = ""
    #if query_param.conversation_history:
        #history_context = get_conversation_turns(
            #query_param.conversation_history, query_param.history_turns
        #)

    # 2. Execute knowledge graph and vector searches in parallel
    #知识图谱上下文获取
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

            #仅高层次关键词（ll_keywords_str为空）：设置为 "global" 模式，优先使用宽泛的高层次概念搜索。
            elif not ll_keywords_str:
                query_param.mode = "global"
            #仅低层次关键词（hl_keywords_str为空）：设置为"local"模式，聚焦具体术语的精确匹配。
            elif not hl_keywords_str:
                query_param.mode = "local"
            #两者均非空：设置为 "mix" 模式，结合两者优势
            else:
                #query_param.mode = "hybrid"
                query_param.mode = "hybrid"

            print(f"ll_keywords_str: {ll_keywords_str}")
            print(f"hl_keywords_str: {hl_keywords_str}")
            print("查询模式",query_param.mode)

            # Build knowledge graph context
            context = await _build_query_context(
                [ll_keywords_str, hl_keywords_str],
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            print("图谱上下文：",context)

            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context: {str(e)}")
            return None

    #向量搜索上下文获取
    async def get_vector_context():
        # Consider conversation history in vector search
        augmented_query = query
        #if history_context:
            #augmented_query = f"{history_context}\n{query}"

        try:
            # Reduce top_k for vector search in hybrid mode since we have structured information from KG
            mix_topk = min(30, query_param.top_k)
            results = await chunks_vdb.query(augmented_query, top_k=mix_topk)
            if not results:
                return None

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)

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
    #并行执行与上下文合并
    kg_context, vector_context = await asyncio.gather(
        get_kg_context(), get_vector_context()
    )

    print(f"知识图谱上下文: {kg_context}")
    print(f"向量搜索上下文: {vector_context}")

    '''
    experiment_folder = os.path.join(os.getcwd(), "实验")
    file_name = "mix_mode_context.txt"
    file_path = os.path.join(experiment_folder, file_name)

    # 创建实验文件夹（如果不存在）
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # 保存上下文信息到文件
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"知识图谱上下文: {kg_context}\n")
        file.write(f"向量搜索上下文: {vector_context}\n")
    '''

    # 保存上下文信息到文件
    experiment_folder = os.path.join(os.getcwd(), "实验")
    file_name = "mix_mode_context.txt"
    file_path = os.path.join(experiment_folder, file_name)

    # 创建实验文件夹（如果不存在）
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # 保存上下文信息（完全修正版）
    with open(file_path, 'a', encoding='utf-8') as file:
        # 写入查询标识
        file.write("\n\n===== 新的查询 =====\n")
        file.write(f"查询时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"查询内容: {query}\n")

        # 知识图谱上下文
        file.write("\n----- 知识图谱上下文 -----\n")
        file.write(kg_context if kg_context else "(无知识图谱信息)\n")

        # 向量搜索结果
        file.write("\n----- 向量搜索结果 -----\n")
        if vector_context:
            file.write("```csv\n")
            file.write('"id","content"\n')  # CSV头部

            # 处理每个chunk（修复了f-string中的转义问题）
            chunks = [c.strip() for c in vector_context.split("--New Chunk--") if c.strip()]
            for i, chunk in enumerate(chunks, 1):
                # 先转义再放入f-string
                escaped_chunk = chunk.replace('"', '""')
                file.write(f'"{i}","{escaped_chunk}"\n')
            file.write("```\n")
        else:
            file.write("(无向量搜索结果)\n")

        # 结束标记
        file.write("\n" + "=" * 40 + "\n")

        # 调用重排序方法
    api_key = "sk-7f18abe89eb445648a5a20b9077c926d"
    api_url = "https://api.bochaai.com/v1/rerank"
    perform_reranking(file_path, api_key, api_url)

    # 处理 globally_reranked_context.txt 文件
    reranked_file = os.path.join(experiment_folder, "globally_reranked_context.txt")
    new_file_path = os.path.join(experiment_folder, "processed_context.txt")

    try:
        with open(reranked_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取每个排名下的文本块
        pattern = r'【排名 \d+ \| 相关度: [\d.]+】\n(.*?)\n\n'
        text_blocks = re.findall(pattern, content, re.DOTALL)

        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            for i, block in enumerate(text_blocks, start=1):
                # 去除前后的引号和逗号
                block = block.strip('",')
                new_file.write(f"{i}:\n{block}\n\n")

        print(f"处理后的文件已保存至 {new_file_path}")
    except FileNotFoundError:
        print(f"未找到 {reranked_file} 文件。")

        # 读取 processed_context.txt 文件内容
    try:
        with open(new_file_path, 'r', encoding='utf-8') as f:
            processed_content = f.read()
    except FileNotFoundError:
        print(f"未找到 {new_file_path} 文件。")
        processed_content = ""

    # 4. Merge contexts
    if kg_context is None and vector_context is None:
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = PROMPTS["mix_rag_response"].format(
        #kg_context=kg_context
        #if kg_context
        #else "No relevant knowledge graph information found",
        #vector_context=vector_context
        #if vector_context
        #else "No relevant text information found",
        processed_context=processed_content,
        response_type=query_param.response_type,
        #history=history_context,
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
    # ll_entities_context, ll_relations_context, ll_text_units_context = "", "", ""
    # hl_entities_context, hl_relations_context, hl_text_units_context = "", "", ""

    ll_keywords, hl_keywords = query[0], query[1]
    print("进入图谱上下文构建")

    if query_param.mode == "local":
        #entities_context, relations_context, text_units_context = await _get_node_data(
        print("进入local")
        entities_context =await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            #text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        #entities_context, relations_context, text_units_context = await _get_edge_data(
        print("进入global")
        relations_context= await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            #text_chunks_db,
            query_param,
        )
    else:  # hybrid mode
        print("进入hybrid")
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

        ll_entities_context= ll_data
        print(ll_entities_context)

        #(
            #hl_entities_context,
            #hl_relations_context,
            #hl_text_units_context,
        #) = hl_data

        hl_relations_context= hl_data
        print(hl_relations_context)

        print("已获取实体与关系上下文信息")

        #entities_context, relations_context, text_units_context = combine_contexts(
            #[hl_entities_context, ll_entities_context],
            #[hl_relations_context, ll_relations_context],
            #[hl_text_units_context, ll_text_units_context],
        #)

        entities_context, relations_context = combine_contexts(
            [ll_entities_context],
            [hl_relations_context],
            #[hl_text_units_context, ll_text_units_context],
        )

        print(f"entities_context2: {entities_context}")
        print(f"relations_context2: {relations_context}")

        # 根据模式确定模式字符串和文件名
    #if query_param.mode == "local":
        #mode_str = "局部模式"
        #file_name = "local_mode_context.txt"
    #elif query_param.mode == "global":
        #mode_str = "全局模式"
        #file_name = "global_mode_context.txt"
    #else:  # hybrid mode
        #mode_str = "混合模式"
        #file_name = "hybrid_mode_context.txt"


        # 准备保存上下文到文件
    #experiment_folder = os.path.join(os.getcwd(), "实验")
    #if not os.path.exists(experiment_folder):
        #os.makedirs(experiment_folder)
    #file_path = os.path.join(experiment_folder, file_name)

        # 保存有效文本块内容到文件，使用追加模式并添加分隔标识
    #with open(file_path, "a", encoding="utf-8") as f:
        #f.write("参考上下文\n")
        #f.write(f"-----Entities-----\n```csv\n{entities_context}\n```\n")
        #f.write(f"-----Relationships-----\n```csv\n{relations_context}\n```\n")
        #f.write(f"-----Text_units-----\n```csv\n{text_units_context}\n```\n")
        #f.write("参考上下文\n")
    #print(f"上下文已保存到 {file_path}")

    # not necessary to use LLM to generate a response
    if not entities_context.strip() and not relations_context.strip():
        return None

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
"""


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
        print("查询结果为空")
        return "", "", ""
    # get entity information
    #node_datas, node_degrees = await asyncio.gather(
    '''
    node_datas = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        #asyncio.gather(
            #*[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        #),
    )
    '''

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )

    print("node_datas1 structure:", type(node_datas), len(node_datas))
    print("node_datas1 content:", node_datas)

    if not all([n is not None for n in node_datas]):
        print("Some nodes are missing, maybe the storage is damaged")
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        #{**n, "entity_name": k["entity_name"], "rank": d}
        {**n, "entity_name": k["entity_name"]}
        for k, n in zip(results, node_datas)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    print("node_datas2:",node_datas)

    '''
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
    '''

    # build prompt
    #entites_section_list = [["id", "entity", "type", "description", "rank"]]
    entites_section_list = [["id", "entity", "description"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                #n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                #n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    '''
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
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    '''
    #return entities_context, relations_context, text_units_context
    print(f"entities_context为: {entities_context}")
    return entities_context


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

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

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

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

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
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    #edge_datas, edge_degree = await asyncio.gather(
    '''
    edge_datas = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        #asyncio.gather(
            #*[
                #knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                #for r in results
            #]
        #),
    )
    '''

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            #"rank": d,
            "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
            **v,
        }
        #for k, v, d in zip(results, edge_datas, edge_degree)
        for k, v in zip(results, edge_datas)
        if v is not None
    ]
    #edge_datas = sorted(
        #edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    #)

    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    print("edge_datas",edge_datas)

    '''
    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    '''

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            #"weight",
            #"rank",
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
                #e["weight"],
                #e["rank"],
                created_at,
            ]
        )

    relations_context = list_of_list_to_csv(relations_section_list)

    '''
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
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    '''


    #return entities_context, relations_context, text_units_context
    print(f"relations_context 的值为: {relations_context}")
    return relations_context


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


#def combine_contexts(entities, relationships, sources):
def combine_contexts(entities, relationships):
    # Function to extract entities, relationships, and sources from context strings
    #hl_entities, ll_entities = entities[0], entities[1]
    #hl_relationships, ll_relationships = relationships[0], relationships[1]
    ll_entities = entities[0]
    print("ll_entities",ll_entities)
    hl_relationships = relationships[0]
    print("hl_relationships", hl_relationships)
    #hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    #combined_entities = process_combine_contexts(hl_entities, ll_entities)
    #combined_entities = process_combine_contexts(ll_entities)
    combined_entities = ll_entities


    # Combine and deduplicate the relationships
    #combined_relationships = process_combine_contexts(
        #hl_relationships, ll_relationships
    #)

    #combined_relationships = process_combine_contexts(hl_relationships)
    combined_relationships = hl_relationships


    # Combine and deduplicate the sources
    #combined_sources = process_combine_contexts(hl_sources, ll_sources)

    #return combined_entities, combined_relationships, combined_sources
    return combined_entities, combined_relationships





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

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

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
            response[len(sys_prompt) :]
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
