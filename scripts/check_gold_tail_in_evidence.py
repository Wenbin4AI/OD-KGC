from __future__ import annotations

import argparse
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
    gold_tail_id = evidence.get("gold_tail_id")
    gold_tail_label = evidence.get("gold_tail_label")

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

    for item in evidence.get("one_hop", []):
        if item.get("h_id") == gold_tail_id or item.get("t_id") == gold_tail_id:
            one_hop_hits.append(
                {
                    "text": item.get("text"),
                    "score": item.get("score"),
                    "h_id": item.get("h_id"),
                    "r_id": item.get("r_id"),
                    "t_id": item.get("t_id"),
                }
            )

    for item in evidence.get("paths", []):
        path = item.get("path", [])
        matched_steps = []

        for step in path:
            if step.get("h_id") == gold_tail_id or step.get("t_id") == gold_tail_id:
                matched_steps.append(
                    {
                        "h_id": step.get("h_id"),
                        "r_id": step.get("r_id"),
                        "t_id": step.get("t_id"),
                        "h_label": step.get("h_label"),
                        "r_label": step.get("r_label"),
                        "t_label": step.get("t_label"),
                    }
                )

        if matched_steps:
            path_hits.append(
                {
                    "text": item.get("text"),
                    "score": item.get("score"),
                    "length": item.get("length"),
                    "terminal_entity_id": item.get("terminal_entity_id"),
                    "terminal_entity_label": item.get("terminal_entity_label"),
                    "matched_steps": matched_steps,
                }
            )

    contains_in_one_hop = len(one_hop_hits) > 0
    contains_in_path = len(path_hits) > 0

    return {
        "contains_gold_tail": contains_in_one_hop or contains_in_path,
        "contains_in_one_hop": contains_in_one_hop,
        "contains_in_path": contains_in_path,
        "gold_tail_id": gold_tail_id,
        "gold_tail_label": gold_tail_label,
        "one_hop_hits": one_hop_hits,
        "path_hits": path_hits,
    }


def analyze_evidence_file(
    evidence_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> None:
    evidence_list = load_jsonl(evidence_path)

    total = len(evidence_list)
    one_hop_count = 0
    path_count = 0
    any_count = 0

    hit_records = []

    for idx, evidence in enumerate(evidence_list):
        result = evidence_contains_gold_tail(evidence)

        if result["contains_in_one_hop"]:
            one_hop_count += 1

        if result["contains_in_path"]:
            path_count += 1

        if result["contains_gold_tail"]:
            any_count += 1

            record = {
                "index": idx,
                "query_head_id": evidence.get("query_head_id"),
                "query_relation_id": evidence.get("query_relation_id"),
                "gold_tail_id": evidence.get("gold_tail_id"),
                "query_head_label": evidence.get("query_head_label"),
                "query_relation_label": evidence.get("query_relation_label"),
                "gold_tail_label": evidence.get("gold_tail_label"),
                "contains_in_one_hop": result["contains_in_one_hop"],
                "contains_in_path": result["contains_in_path"],
                "one_hop_hits": result["one_hop_hits"],
                "path_hits": result["path_hits"],
            }

            hit_records.append(record)

    print("=" * 100)
    print("[Gold Tail Evidence Analysis]")
    print("=" * 100)
    print(f"Evidence file: {evidence_path}")
    print(f"Total queries: {total}")
    print(f"Queries containing gold tail in any evidence: {any_count}")

    if total > 0:
        print(f"Ratio: {any_count / total:.4f}")
        print(f"One-hop hit queries: {one_hop_count} ({one_hop_count / total:.4f})")
        print(f"Path hit queries: {path_count} ({path_count / total:.4f})")

    print("=" * 100)

    print("\n[Examples]")
    for record in hit_records[:10]:
        print("-" * 100)
        print(
            f"Index: {record['index']} | "
            f"Query: ({record['query_head_label']}, "
            f"{record['query_relation_label']}, ?) | "
            f"Gold: {record['gold_tail_label']}"
        )
        print(
            f"One-hop: {record['contains_in_one_hop']} | "
            f"Path: {record['contains_in_path']}"
        )

        if record["one_hop_hits"]:
            print("  One-hop hits:")
            for item in record["one_hop_hits"][:3]:
                print(f"    - {item['text']} | score={item['score']}")

        if record["path_hits"]:
            print("  Path hits:")
            for item in record["path_hits"][:3]:
                print(f"    - {item['text']} | score={item['score']}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in hit_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n[Saved] Hit records saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check how many queries have evidence containing the gold tail entity."
    )

    parser.add_argument(
        "--evidence_path",
        type=str,
        default="import/evidence/FB15k-237/test_evidence.jsonl",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="import/evidence/FB15k-237/gold_tail_evidence_hits.jsonl",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    analyze_evidence_file(
        evidence_path=args.evidence_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()