import json
import os
from collections import defaultdict, Counter


DATA_DIR = "/home/wenbin.guo/DKGE4R/data/WN18RR"

ENTITY_PATH = os.path.join(DATA_DIR, "entity.json")
RELATION_PATH = os.path.join(DATA_DIR, "relation.json")

TRAIN_PATH = os.path.join(DATA_DIR, "train2id.txt")
VALID_PATH = os.path.join(DATA_DIR, "valid2id.txt")
TEST_PATH = os.path.join(DATA_DIR, "test2id.txt")

OUTPUT_PATH = os.path.join(DATA_DIR, "relation_new.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entity_info(entity_path):
    """
    建立 entity_id -> 实体信息 的映射
    """
    entities = load_json(entity_path)

    entity_map = {}
    for item in entities:
        eid = int(item["value"])
        entity_map[eid] = {
            "freebase_id": item.get("freebase_id"),
            "label": item.get("label"),
            "classname": item.get("classname", "Unknown"),
            "classid": item.get("classid", -1),
        }
    return entity_map


def load_relation_info(relation_path):
    """
    建立 relation_id -> 关系信息 的映射
    """
    relations = load_json(relation_path)

    relation_map = {}
    for item in relations:
        rid = int(item["id"])
        relation_map[rid] = {
            "freebase": item.get("freebase"),
            "id": item.get("id"),
            "label": item.get("label"),
        }
    return relation_map


def parse_triple_file(path):
    """
    解析 train2id.txt / valid2id.txt / test2id.txt

    常见格式有两种：
    1) 第一行是三元组数量，后面每行: h t r
    2) 没有第一行数量，直接每行: h t r

    这里做兼容处理。
    """
    triples = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return triples

    start_idx = 0

    # 若第一行只有一个数字，视为样本数
    first_parts = lines[0].split()
    if len(first_parts) == 1 and first_parts[0].isdigit():
        start_idx = 1

    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) < 3:
            continue

        # FB15k-237 常见是 h t r
        h, t, r = map(int, parts[:3])
        triples.append((h, r, t))

    return triples


def counter_to_sorted_list(counter_obj):
    """
    将 Counter 转成按频次降序排列的 list
    """
    items = sorted(counter_obj.items(), key=lambda x: (-x[1], x[0]))
    result = []
    for cls_name, freq in items:
        result.append({
            "classname": cls_name,
            "count": freq
        })
    return result


def build_domain_range(entity_map, relation_map, triple_files):
    """
    根据三元组统计每个关系的 domain / range
    """
    relation_head_class_counter = defaultdict(Counter)
    relation_tail_class_counter = defaultdict(Counter)

    skipped_triples = 0
    total_triples = 0

    for file_path in triple_files:
        triples = parse_triple_file(file_path)
        print(f"Loaded {len(triples)} triples from {file_path}")

        for h, r, t in triples:
            total_triples += 1

            if h not in entity_map or t not in entity_map:
                skipped_triples += 1
                continue

            head_class = entity_map[h].get("classname", "Unknown")
            tail_class = entity_map[t].get("classname", "Unknown")

            if head_class is None or str(head_class).strip() == "":
                head_class = "Unknown"
            if tail_class is None or str(tail_class).strip() == "":
                tail_class = "Unknown"

            relation_head_class_counter[r][head_class] += 1
            relation_tail_class_counter[r][tail_class] += 1

    print(f"Total triples processed: {total_triples}")
    print(f"Skipped triples (entity id not found): {skipped_triples}")

    new_relations = []

    for rid in sorted(relation_map.keys()):
        rel_info = relation_map[rid]

        head_counter = relation_head_class_counter[rid]
        tail_counter = relation_tail_class_counter[rid]

        domain_candidates = counter_to_sorted_list(head_counter)
        range_candidates = counter_to_sorted_list(tail_counter)

        # 取频次最高的类别作为主 domain / range
        domain = domain_candidates[0]["classname"] if domain_candidates else "Unknown"
        range_ = range_candidates[0]["classname"] if range_candidates else "Unknown"

        new_item = {
            "freebase": rel_info["freebase"],
            "id": rel_info["id"],
            "label": rel_info["label"],
            "domain": domain,
            "range": range_,
            "domain_candidates": domain_candidates,
            "range_candidates": range_candidates
        }

        new_relations.append(new_item)

    return new_relations


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved to: {path}")


def main():
    entity_map = load_entity_info(ENTITY_PATH)
    relation_map = load_relation_info(RELATION_PATH)

    print(f"Loaded {len(entity_map)} entities")
    print(f"Loaded {len(relation_map)} relations")

    triple_files = [TRAIN_PATH, VALID_PATH, TEST_PATH]

    new_relations = build_domain_range(
        entity_map=entity_map,
        relation_map=relation_map,
        triple_files=triple_files
    )

    save_json(new_relations, OUTPUT_PATH)


if __name__ == "__main__":
    main()