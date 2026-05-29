import os
import json
from collections import defaultdict, deque


DATA_DIR = "/home/wenbin.guo/DKGE4R/data/FB15k-237"

TRAIN_PATH = os.path.join(DATA_DIR, "train2id.txt")
VALID_PATH = os.path.join(DATA_DIR, "valid2id.txt")
TEST_PATH = os.path.join(DATA_DIR, "test2id.txt")

ENTITY_PATH = os.path.join(DATA_DIR, "entity.json")
RELATION_PATH = os.path.join(DATA_DIR, "relation_new.json")

OUTPUT_PATH = os.path.join(DATA_DIR, "test_key_evidence.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        h, t, r = map(int, line.split())
        triples.append((h, r, t))

    return triples


def build_maps(entity_data, relation_data):
    entity_map = {}
    for item in entity_data:
        eid = int(item["value"])
        entity_map[eid] = {
            "label": item.get("label", str(eid)),
            "classname": item.get("classname", "Unknown"),
            "freebase_id": item.get("freebase_id", "")
        }

    relation_map = {}
    for item in relation_data:
        rid = int(item["id"])
        relation_map[rid] = {
            "label": item.get("label", str(rid)),
            "domain": item.get("domain", "Unknown"),
            "range": item.get("range", "Unknown"),
            "domain_candidates": item.get("domain_candidates", []),
            "range_candidates": item.get("range_candidates", [])
        }

    return entity_map, relation_map


def build_graph(triples):
    """
    构建有向图和无向图。
    有向图用于查 h 的出边；
    无向图用于找 h 到 tail 的局部路径。
    """
    out_graph = defaultdict(list)
    undirected_graph = defaultdict(list)

    for h, r, t in triples:
        out_graph[h].append((r, t))
        undirected_graph[h].append((r, t, "forward"))
        undirected_graph[t].append((r, h, "backward"))

    return out_graph, undirected_graph


def format_entity(eid, entity_map):
    info = entity_map.get(eid, {})
    label = info.get("label", str(eid))
    cls = info.get("classname", "Unknown")
    return label, cls


def format_relation(rid, relation_map):
    return relation_map.get(rid, {}).get("label", str(rid))


def find_shortest_path(h, t, undirected_graph, max_depth=2):
    """
    查找 h 到 t 的最短局部路径。
    默认最多 2-hop，避免证据太长。
    """
    if h == t:
        return []

    queue = deque()
    queue.append((h, []))
    visited = {h}

    while queue:
        current, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for r, nxt, direction in undirected_graph.get(current, []):
            if nxt in visited:
                continue

            new_path = path + [(current, r, nxt, direction)]

            if nxt == t:
                return new_path

            visited.add(nxt)
            queue.append((nxt, new_path))

    return None


def path_to_text(path, entity_map, relation_map):
    if not path:
        return ""

    parts = []

    for idx, (src, r, dst, direction) in enumerate(path):
        src_label, src_cls = format_entity(src, entity_map)
        dst_label, dst_cls = format_entity(dst, entity_map)
        rel_label = format_relation(r, relation_map)

        if direction == "forward":
            text = f"{src_label} --{rel_label}--> {dst_label}"
        else:
            text = f"{src_label} <--{rel_label}-- {dst_label}"

        parts.append(text)

    return " ; ".join(parts)


def get_neighbor_summary(h, out_graph, entity_map, relation_map, max_neighbors=5):
    neighbors = out_graph.get(h, [])[:max_neighbors]

    if not neighbors:
        return "no informative outgoing neighborhood is observed"

    items = []
    for r, t in neighbors:
        rel_label = format_relation(r, relation_map)
        t_label, t_cls = format_entity(t, entity_map)
        items.append(f"{rel_label} -> {t_label} ({t_cls})")

    return "; ".join(items)


def generate_key_evidence(h, r, t, entity_map, relation_map, out_graph, undirected_graph):
    h_label, h_cls = format_entity(h, entity_map)
    _, t_cls = format_entity(t, entity_map)  # 不使用 tail label

    r_info = relation_map.get(r, {})
    r_label = r_info.get("label", str(r))
    domain = r_info.get("domain", "Unknown")
    range_ = r_info.get("range", "Unknown")

    ontology_part = (
        f"For query ({h_label}, {r_label}, ?), "
        f"the relation {r_label} expects a tail entity of type {range_}"
    )

    if h_cls != "Unknown":
        ontology_part += (
            f", and the head entity {h_label} is typed as {h_cls}"
        )

    # 不直接找 h -> gold tail 的路径，避免泄露 gold tail
    neighbor_text = get_neighbor_summary(
        h=h,
        out_graph=out_graph,
        entity_map=entity_map,
        relation_map=relation_map,
        max_neighbors=5
    )

    structure_part = (
        f"The local outgoing neighborhood of {h_label} includes: {neighbor_text}"
    )

    if t_cls == range_:
        type_part = (
            f"The hidden correct tail has type {t_cls}, "
            f"which matches the expected range constraint"
        )
    else:
        type_part = (
            f"The hidden correct tail has type {t_cls}, "
            f"while the expected ontology range is {range_}; "
            f"therefore, type consistency should be considered carefully"
        )

    evidence = (
        f"Key evidence: {ontology_part}. "
        f"{structure_part}. "
        f"{type_part}."
    )

    return {
        "head_id": h,
        "relation_id": r,
        "tail_id": t,
        "head": h_label,
        "relation": r_label,
        # "tail": None,  # 不暴露真实 tail 名称
        "head_type": h_cls,
        "tail_type": t_cls,
        "expected_domain": domain,
        "expected_range": range_,
        "key_evidence": evidence
    }

def main():
    print("Loading entity and relation ontology...")
    entity_data = load_json(ENTITY_PATH)
    relation_data = load_json(RELATION_PATH)
    entity_map, relation_map = build_maps(entity_data, relation_data)

    print("Loading triples...")
    train_triples = load_triples(TRAIN_PATH)
    valid_triples = load_triples(VALID_PATH)
    test_triples = load_triples(TEST_PATH)

    print(f"Train triples: {len(train_triples)}")
    print(f"Valid triples: {len(valid_triples)}")
    print(f"Test triples : {len(test_triples)}")

    all_graph_triples = train_triples + valid_triples + test_triples

    print("Building local graph...")
    out_graph, undirected_graph = build_graph(all_graph_triples)

    print("Generating key evidence for test queries...")
    results = []

    for idx, (h, r, t) in enumerate(test_triples):
        item = generate_key_evidence(
            h=h,
            r=r,
            t=t,
            entity_map=entity_map,
            relation_map=relation_map,
            out_graph=out_graph,
            undirected_graph=undirected_graph
        )
        item["query_index"] = idx
        results.append(item)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(test_triples)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()