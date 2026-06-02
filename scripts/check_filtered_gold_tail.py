from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def evidence_contains_gold_tail(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    检查证据中是否有包含 gold tail 的 one-hop 或 path 证据
    """
    gold_tail_id = evidence.get("gold_tail_id")
    one_hop_hits = []
    path_hits = []

    if gold_tail_id is None:
        return {
            "contains_gold_tail": False,
            "contains_in_one_hop": False,
            "contains_in_path": False,
            "one_hop_hits": [],
            "path_hits": [],
        }

    # 检查一跳证据
    for item in evidence.get("filtered_one_hop", []):
        if item.get("h_id") == gold_tail_id or item.get("t_id") == gold_tail_id:
            one_hop_hits.append(item)

    # 检查路径证据
    for item in evidence.get("filtered_paths", []):
        path = item.get("path", [])
        matched_steps = []
        for step in path:
            if step.get("h_id") == gold_tail_id or step.get("t_id") == gold_tail_id:
                matched_steps.append(step)
        if matched_steps:
            path_hits.append({"path_item": item, "matched_steps": matched_steps})

    contains_in_one_hop = len(one_hop_hits) > 0
    contains_in_path = len(path_hits) > 0

    return {
        "contains_gold_tail": contains_in_one_hop or contains_in_path,
        "contains_in_one_hop": contains_in_one_hop,
        "contains_in_path": contains_in_path,
        "gold_tail_id": gold_tail_id,
        "gold_tail_label": evidence.get("gold_tail_label"),
        "one_hop_hits": one_hop_hits,
        "path_hits": path_hits,
    }


def analyze_filtered_evidence(
    filtered_path: str | Path,
    max_examples: int = 10,
) -> None:
    """
    分析 filter.py 输出文件中，多少查询证据包含正确尾实体
    """
    evidence_list = load_jsonl(filtered_path)
    total_queries = len(evidence_list)
    gold_hit_count = 0
    one_hop_hit_count = 0
    path_hit_count = 0
    hit_records = []

    for idx, evidence in enumerate(evidence_list):
        result = evidence_contains_gold_tail(evidence)
        if result["contains_gold_tail"]:
            gold_hit_count += 1
            if result["contains_in_one_hop"]:
                one_hop_hit_count += 1
            if result["contains_in_path"]:
                path_hit_count += 1
            hit_records.append({
                "index": idx,
                "query_head_label": evidence.get("query_head_label"),
                "query_relation_label": evidence.get("query_relation_label"),
                "gold_tail_label": evidence.get("gold_tail_label"),
                "one_hop_hit": result["contains_in_one_hop"],
                "path_hit": result["contains_in_path"],
            })

    print("="*100)
    print(f"Total queries: {total_queries}")
    print(f"Queries containing gold tail evidence: {gold_hit_count} ({gold_hit_count/total_queries:.4f})")
    print(f"  One-hop hits: {one_hop_hit_count}")
    print(f"  Path hits: {path_hit_count}")
    print("="*100)

    print("\n[Examples of queries with gold tail in evidence]")
    for rec in hit_records[:max_examples]:
        print(f"Index: {rec['index']}, Query: ({rec['query_head_label']}, {rec['query_relation_label']}, ?), Gold: {rec['gold_tail_label']}, One-hop: {rec['one_hop_hit']}, Path: {rec['path_hit']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check how many filtered evidence queries contain gold tail")
    parser.add_argument(
        "--filtered_path",
        type=str,
        default="import/evidence/FB15k-237/test_filtered_evidence.jsonl",
        help="Path to filtered evidence JSONL file from filter.py"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=10,
        help="Number of example queries to print"
    )

    args = parser.parse_args()

    analyze_filtered_evidence(args.filtered_path, args.max_examples)