import os
import json
import math
import random
from collections import defaultdict, Counter

# =========================
# 路径配置
# =========================
DATA_DIR = "/home/wenbin.guo/DKGE4R/data/FB15k-237"
SUBGRAPH_DIR = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test"
OUTPUT_DIR = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v2"

ENTITY_PATH = os.path.join(DATA_DIR, "entity.json")
RELATION_PATH = os.path.join(DATA_DIR, "relation_new.json")

CANDIDATE_LIMIT = 20
NEGATIVE_SIZE = CANDIDATE_LIMIT - 1
RANDOM_SEED = 42


# =========================
# 基础读写
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# 建立映射
# =========================
def build_entity_map(entities):
    entity_map = {}
    for e in entities:
        entity_map[int(e["value"])] = e
    return entity_map


def build_relation_map(relations):
    relation_map = {}
    for r in relations:
        relation_map[int(r["id"])] = r
    return relation_map


def get_entity_label(entity_map, eid):
    if eid not in entity_map:
        return f"Entity_{eid}"
    return entity_map[eid].get("label", f"Entity_{eid}")


def get_entity_class(entity_map, eid):
    if eid not in entity_map:
        return "Unknown"
    return entity_map[eid].get("classname", "Unknown")


def get_relation_label(relation_map, rid):
    if rid not in relation_map:
        return f"Relation_{rid}"
    return relation_map[rid].get("label", f"Relation_{rid}")


# =========================
# 图结构处理
# =========================
def build_direct_edge_set(edges):
    return {(e["head"], e["relation"], e["tail"]) for e in edges}


def group_paths_by_tail(paths):
    tail2paths = defaultdict(list)
    for p in paths:
        tail2paths[p["tail"]].append(p)
    return tail2paths


def build_aux_neighbors(edges, target_relation):
    aux_out = defaultdict(list)
    aux_in = defaultdict(list)
    for e in edges:
        h, r, t = e["head"], e["relation"], e["tail"]
        if r != target_relation:
            aux_out[h].append((r, t))
            aux_in[t].append((r, h))
    return aux_out, aux_in


def count_shared_support(candidate, candidate_set, aux_out, aux_in):
    count = 0
    for _, nb in aux_out.get(candidate, []):
        if nb in candidate_set:
            count += 1
    for _, nb in aux_in.get(candidate, []):
        if nb in candidate_set:
            count += 1
    return count


def count_target_relation_edges_from_head(h, r, edges):
    return sum(1 for e in edges if e["head"] == h and e["relation"] == r)


# =========================
# 路径模式摘要
# =========================
def summarize_relation_path_patterns(paths, relation_map, topk=5):
    """
    统计最常见的路径关系模式:
    e.g. [23] / [23,223]
    """
    pattern_counter = Counter()

    for p in paths:
        rels = tuple(p.get("relations", []))
        if len(rels) == 0:
            continue
        pattern_counter[rels] += 1

    if not pattern_counter:
        return ["- No explicit path patterns were extracted."]

    lines = []
    for rel_seq, cnt in pattern_counter.most_common(topk):
        rel_labels = [get_relation_label(relation_map, rid) for rid in rel_seq]
        pattern_str = " -> ".join(rel_labels)
        lines.append(f"- {pattern_str} (count={cnt})")

    return lines


# =========================
# 候选打分
# =========================
def get_range_candidates_set(rel_info, topk=5):
    result = set()
    for item in rel_info.get("range_candidates", [])[:topk]:
        cls = item.get("classname", "Unknown")
        if cls and cls != "Unknown":
            result.add(cls)
    return result


def compute_candidate_features(
    t,
    h,
    r,
    rel_info,
    entity_map,
    direct_edges,
    tail2paths,
    aux_out,
    aux_in,
    key_nodes,
    candidate_set
):
    expected_range = rel_info.get("range", "Unknown")
    top_range_classes = get_range_candidates_set(rel_info, topk=5)

    t_class = get_entity_class(entity_map, t)

    type_match = 1 if (expected_range != "Unknown" and t_class == expected_range) else 0
    soft_type_match = 1 if t_class in top_range_classes else 0
    direct = 1 if (h, r, t) in direct_edges else 0
    path_count = len(tail2paths.get(t, []))
    aux_support = count_shared_support(t, candidate_set, aux_out, aux_in)
    key_flag = 1 if t in key_nodes else 0

    # 预排序分数：偏向“迷惑性强”的候选
    score = (
        3.0 * type_match
        + 1.5 * soft_type_match
        + 3.0 * direct
        + 1.5 * math.log1p(path_count)
        + 1.0 * math.log1p(aux_support)
        + 1.0 * key_flag
    )

    return {
        "candidate_id": t,
        "candidate_label": get_entity_label(entity_map, t),
        "candidate_class": t_class,
        "features": {
            "type_match": type_match,
            "soft_type_match": soft_type_match,
            "direct": direct,
            "path_count": path_count,
            "aux_support": aux_support,
            "key": key_flag
        },
        "score": round(score, 6)
    }


# =========================
# 候选采样：确保只有一个真答案
# =========================
def select_candidates_with_single_gold(
    candidate_tails,
    gold_tail,
    h,
    r,
    rel_info,
    entity_map,
    direct_edges,
    tail2paths,
    aux_out,
    aux_in,
    key_nodes,
    candidate_limit=20,
    random_seed=42
):
    if gold_tail is None:
        raise ValueError("Subgraph missing gold_tail; cannot enforce exactly one true candidate.")

    candidate_set_full = set(candidate_tails)

    all_infos = []
    for t in candidate_tails:
        info = compute_candidate_features(
            t=t,
            h=h,
            r=r,
            rel_info=rel_info,
            entity_map=entity_map,
            direct_edges=direct_edges,
            tail2paths=tail2paths,
            aux_out=aux_out,
            aux_in=aux_in,
            key_nodes=key_nodes,
            candidate_set=candidate_set_full
        )
        all_infos.append(info)

    gold_info = None
    negatives = []
    for info in all_infos:
        if info["candidate_id"] == gold_tail:
            gold_info = info
        else:
            negatives.append(info)

    if gold_info is None:
        raise ValueError(f"gold_tail={gold_tail} not found in candidate_tails")

    if len(negatives) < candidate_limit - 1:
        raise ValueError(
            f"Not enough negative candidates: need {candidate_limit - 1}, got {len(negatives)}"
        )

    rng = random.Random(random_seed)

    # 随机选 19 个负样本
    selected_negatives = rng.sample(negatives, candidate_limit - 1)

    # 加入 gold
    selected = [gold_info] + selected_negatives

    # 再随机打乱 20 个候选的顺序
    rng.shuffle(selected)

    return selected


# =========================
# Prompt摘要
# =========================
def make_ontology_summary(h, r, entity_map, relation_map):
    rel_info = relation_map.get(r, {})
    h_label = get_entity_label(entity_map, h)
    h_class = get_entity_class(entity_map, h)
    r_label = get_relation_label(relation_map, r)
    domain = rel_info.get("domain", "Unknown")
    range_ = rel_info.get("range", "Unknown")

    lines = [
        f"- Query relation: {r_label}",
        f"- Head entity: {h_label}",
        f"- Head class: {h_class}",
        f"- Relation domain: {domain}",
        f"- Relation range: {range_}",
    ]

    if domain != "Unknown":
        if h_class == domain:
            lines.append("- Domain check: matched")
        else:
            lines.append("- Domain check: not clearly matched")

    if range_ != "Unknown":
        lines.append(f"- Tail entities of class '{range_}' should be preferred.")

    return "\n".join(lines)


def make_subgraph_summary(h, r, subgraph, relation_map, entity_map):
    r_label = get_relation_label(relation_map, r)
    target_rel_count = count_target_relation_edges_from_head(h, r, subgraph["edges"])
    num_nodes = len(subgraph.get("nodes", []))
    num_edges = len(subgraph.get("edges", []))
    num_paths = len(subgraph.get("paths", []))
    key_nodes = subgraph.get("key_nodes", [])

    lines = [
        f"- Local subgraph: nodes={num_nodes}, edges={num_edges}, paths={num_paths}",
        f"- Head outgoing edges under target relation {r_label}: {target_rel_count}",
    ]

    if len(key_nodes) > 0:
        key_labels = [get_entity_label(entity_map, x) for x in key_nodes[:8]]
        lines.append(f"- Key support nodes: {', '.join(key_labels)}")

    rel_counter = Counter()
    for e in subgraph["edges"]:
        if e["relation"] != r:
            rel_counter[e["relation"]] += 1

    if rel_counter:
        top_aux = rel_counter.most_common(3)
        aux_desc = []
        for rid, cnt in top_aux:
            aux_desc.append(f"{get_relation_label(relation_map, rid)}({cnt})")
        lines.append(f"- Main auxiliary relations: {', '.join(aux_desc)}")

    return "\n".join(lines)


def make_candidate_compact_line(idx, info):
    """
    更紧凑的候选特征标签
    """
    f = info["features"]
    return (
        f"{idx}. {info['candidate_label']} "
        f"[id={info['candidate_id']} | "
        f"class={info['candidate_class']} | "
        f"type={f['type_match']} | "
        f"soft_type={f['soft_type_match']} | "
        f"direct={f['direct']} | "
        f"paths={f['path_count']} | "
        f"aux={f['aux_support']} | "
        f"key={f['key']}]"
    )


def build_prompt(subgraph, entity_map, relation_map, candidate_limit=20, random_seed=42):
    h = subgraph["query"]["head"]
    r = subgraph["query"]["relation"]
    candidate_tails = subgraph["query"]["candidate_tails"]
    gold_tail = subgraph.get("gold_tail", None)

    if gold_tail is None:
        raise ValueError("gold_tail not found in subgraph json")

    rel_info = relation_map.get(r, {"label": f"Relation_{r}"})

    direct_edges = build_direct_edge_set(subgraph["edges"])
    tail2paths = group_paths_by_tail(subgraph.get("paths", []))
    aux_out, aux_in = build_aux_neighbors(subgraph["edges"], target_relation=r)
    key_nodes = set(subgraph.get("key_nodes", []))

    selected_infos = select_candidates_with_single_gold(
        candidate_tails=candidate_tails,
        gold_tail=gold_tail,
        h=h,
        r=r,
        rel_info=rel_info,
        entity_map=entity_map,
        direct_edges=direct_edges,
        tail2paths=tail2paths,
        aux_out=aux_out,
        aux_in=aux_in,
        key_nodes=key_nodes,
        candidate_limit=candidate_limit,
        random_seed=random_seed
    )

    candidate_id_to_rank_index = {}
    candidate_lines = []

    for idx, info in enumerate(selected_infos, start=1):
        candidate_id_to_rank_index[info["candidate_id"]] = idx
        candidate_lines.append(make_candidate_compact_line(idx, info))

    gold_rank_index = candidate_id_to_rank_index[gold_tail]

    ontology_summary = make_ontology_summary(h, r, entity_map, relation_map)
    subgraph_summary = make_subgraph_summary(h, r, subgraph, relation_map, entity_map)
    path_pattern_lines = summarize_relation_path_patterns(
        subgraph.get("paths", []), relation_map, topk=5
    )

    h_label = get_entity_label(entity_map, h)
    r_label = get_relation_label(relation_map, r)

    prompt = f"""You are given a knowledge graph link prediction ranking task.

Your task is to rank the numbered candidate tail entities for the query (head, relation, ?).

You must follow these rules strictly:
1. Only rank the numbered candidates listed below.
2. Use ontology constraints, path-pattern evidence, and local subgraph evidence jointly.
3. Prefer candidates whose class matches the expected relation range.
4. Prefer candidates with direct target-relation evidence, stronger path support, and denser auxiliary-neighborhood support.
5. Output only the ranking of candidate numbers in descending likelihood.
6. Do not explain your reasoning.
7. Do not output entity names.
8. Do not output anything except one Python-style list of candidate numbers.
9. Do not omit any number and do not repeat any number.

Expected output format:
[3, 7, 1, 5, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

Query:
(head={h_label}, relation={r_label}, tail=?)

Ontology summary:
{ontology_summary}

Local subgraph summary:
{subgraph_summary}

Frequent relation path patterns:
{chr(10).join(path_pattern_lines)}

Candidates:
{chr(10).join(candidate_lines)}
"""

    result = {
        "query": {
            "head_id": h,
            "head_label": h_label,
            "head_class": get_entity_class(entity_map, h),
            "relation_id": r,
            "relation_label": r_label,
            "gold_tail": gold_tail,
            "gold_tail_label": get_entity_label(entity_map, gold_tail),
            "gold_candidate_rank_index": gold_rank_index
        },
        "candidate_limit": candidate_limit,
        "selected_candidates": [
            {
                "rank_index": i + 1,
                "entity_id": info["candidate_id"],
                "entity_label": info["candidate_label"],
                "entity_class": info["candidate_class"],
                "score": info["score"],
                "features": info["features"],
                "is_gold": 1 if info["candidate_id"] == gold_tail else 0
            }
            for i, info in enumerate(selected_infos)
        ],
        "candidate_id_to_rank_index": candidate_id_to_rank_index,
        "prompt": prompt
    }

    return result


# =========================
# 文件处理
# =========================
def process_one_file(subgraph_path, entity_map, relation_map, output_dir, candidate_limit=20, random_seed=42):
    subgraph = load_json(subgraph_path)

    file_seed = random_seed + abs(hash(os.path.basename(subgraph_path))) % 1000000

    result = build_prompt(
        subgraph=subgraph,
        entity_map=entity_map,
        relation_map=relation_map,
        candidate_limit=candidate_limit,
        random_seed=file_seed
    )

    filename = os.path.basename(subgraph_path)
    stem = os.path.splitext(filename)[0]

    os.makedirs(output_dir, exist_ok=True)

    json_out = os.path.join(output_dir, f"{stem}_prompt.json")
    txt_out = os.path.join(output_dir, f"{stem}_prompt.txt")

    save_json(result, json_out)
    save_text(result["prompt"], txt_out)

    print(f"Saved: {json_out}")
    print(f"Saved: {txt_out}")

def generate_hit1_prompt_with_ontology(subgraph, entity_map, relation_map, candidate_limit=20, random_seed=42):
    h = subgraph["query"]["head"]
    r = subgraph["query"]["relation"]
    candidate_tails = subgraph["query"]["candidate_tails"]
    gold_tail = subgraph.get("gold_tail", None)

    if gold_tail is None:
        raise ValueError("gold_tail not found in subgraph")

    rel_info = relation_map.get(r, {"label": f"Relation_{r}"})
    direct_edges = build_direct_edge_set(subgraph["edges"])
    tail2paths = group_paths_by_tail(subgraph.get("paths", []))
    aux_out, aux_in = build_aux_neighbors(subgraph["edges"], target_relation=r)
    key_nodes = set(subgraph.get("key_nodes", []))

    # 生成候选，保证只有一个 gold
    selected_infos = select_candidates_with_single_gold(
        candidate_tails=candidate_tails,
        gold_tail=gold_tail,
        h=h,
        r=r,
        rel_info=rel_info,
        entity_map=entity_map,
        direct_edges=direct_edges,
        tail2paths=tail2paths,
        aux_out=aux_out,
        aux_in=aux_in,
        key_nodes=key_nodes,
        candidate_limit=candidate_limit,
        random_seed=random_seed
    )

    # 构建紧凑候选行
    candidate_lines = [make_candidate_compact_line(idx + 1, info) for idx, info in enumerate(selected_infos)]
    candidate_id_to_rank_index = {info["candidate_id"]: idx+1 for idx, info in enumerate(selected_infos)}
    gold_rank_index = candidate_id_to_rank_index[gold_tail]

    ontology_summary = make_ontology_summary(h, r, entity_map, relation_map)
    subgraph_summary = make_subgraph_summary(h, r, subgraph, relation_map, entity_map)
    path_pattern_lines = summarize_relation_path_patterns(subgraph.get("paths", []), relation_map)

    prompt = f"""
You are an expert knowledge graph completion system.
Select exactly ONE correct tail entity from the candidate list.

Head: {get_entity_label(entity_map, h)} (class={get_entity_class(entity_map, h)})
Relation: {get_relation_label(relation_map, r)}

Ontology summary:
{ontology_summary}

Subgraph summary:
{subgraph_summary}

Frequent path patterns:
{chr(10).join(path_pattern_lines)}

Candidates:
{chr(10).join(candidate_lines)}

Output format:
{{"selected_index": integer}}
"""

    return {
        "prompt": prompt,
        "query": {
            "head_id": h,
            "relation_id": r,
            "gold_tail": gold_tail,
            "gold_candidate_rank_index": gold_rank_index,
            "selected_candidates": selected_infos
        }
    }

def process_all_files():
    random.seed(RANDOM_SEED)

    entities = load_json(ENTITY_PATH)
    relations = load_json(RELATION_PATH)

    entity_map = build_entity_map(entities)
    relation_map = build_relation_map(relations)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [
        os.path.join(SUBGRAPH_DIR, x)
        for x in os.listdir(SUBGRAPH_DIR)
        if x.endswith(".json")
    ]
    files.sort()

    for path in files:
        try:
            process_one_file(
                subgraph_path=path,
                entity_map=entity_map,
                relation_map=relation_map,
                output_dir=OUTPUT_DIR,
                candidate_limit=CANDIDATE_LIMIT,
                random_seed=RANDOM_SEED
            )
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")


if __name__ == "__main__":
    process_all_files()