from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.kg_loader import KGLoader


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    value = str(value).strip()
    value = value.replace("_", " ")
    value = value.replace("/", " / ")
    value = " ".join(value.split())
    return value.lower()


def approx_token_len(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())


def safe_get_score(item: Dict[str, Any]) -> float:
    if "filtered_score" in item:
        return float(item["filtered_score"])
    if "score" in item:
        return float(item["score"])
    return 0.0


def to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, (tuple, set)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, dict):
        results = []
        for k, v in value.items():
            if isinstance(v, (list, tuple, set)):
                results.extend([str(x) for x in v if x is not None and str(x).strip()])
            elif v is not None and str(v).strip():
                results.append(str(v))
            elif k is not None and str(k).strip():
                results.append(str(k))
        return results
    return [str(value)]


def is_numeric_like(value: Any) -> bool:
    if value is None:
        return False
    value = str(value).strip()
    if not value:
        return False
    return value.isdigit()


def deduplicate_clean(values: List[Any]) -> List[str]:
    results = []
    seen = set()
    for value in values:
        value = normalize_text(value)
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            results.append(value)
    return results


def evidence_contains_gold_tail_name(item: Dict[str, Any], gold_tail_label: Any) -> bool:
    if gold_tail_label is None:
        return False

    gold = normalize_text(gold_tail_label)
    if not gold or gold == "none":
        return False

    fields = [
        item.get("text", ""),
        item.get("terminal_entity_label", ""),
    ]

    triple_name = item.get("triple_name")
    if isinstance(triple_name, dict):
        fields.extend([
            triple_name.get("head", ""),
            triple_name.get("relation", ""),
            triple_name.get("tail", ""),
        ])

    if isinstance(item.get("path"), list):
        for step in item["path"]:
            if isinstance(step, dict):
                fields.extend([
                    step.get("head", ""),
                    step.get("relation", ""),
                    step.get("tail", ""),
                    step.get("h_label", ""),
                    step.get("r_label", ""),
                    step.get("t_label", ""),
                    step.get("head_label", ""),
                    step.get("relation_label", ""),
                    step.get("tail_label", ""),
                ])

    merged = normalize_text(" ".join(str(x) for x in fields if x is not None))
    return gold in merged


class DatasetSchemaHelper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.entities = dataset.entities
        self.relations = dataset.relations

    def entity_label(self, entity_id: Optional[int]) -> str:
        entity_id = to_int(entity_id)
        if entity_id is None:
            return "None"
        ent = self.entities.get(entity_id)
        if ent is None:
            return f"[UnknownEntity:{entity_id}]"
        return getattr(ent, "label", None) or str(entity_id)

    def relation_label(self, relation_id: Optional[int]) -> str:
        relation_id = to_int(relation_id)
        if relation_id is None:
            return "None"
        rel = self.relations.get(relation_id)
        if rel is None:
            return f"[UnknownRelation:{relation_id}]"
        return getattr(rel, "label", None) or str(relation_id)

    def entity_classes(self, entity_id: Optional[int]) -> List[str]:
        entity_id = to_int(entity_id)
        if entity_id is None:
            return []
        ent = self.entities.get(entity_id)
        if ent is None:
            return []

        classes = []
        if getattr(ent, "classname", None):
            classes.extend(to_list(ent.classname))

        raw = getattr(ent, "raw", None) or {}
        for key in [
            "classlabel",
            "class_label",
            "classname",
            "class_name",
            "classes",
            "types",
            "type",
        ]:
            if key in raw:
                classes.extend(to_list(raw[key]))

        classes = deduplicate_clean(classes)
        classes = [c for c in classes if not is_numeric_like(c)]
        return classes

    def relation_domain(self, relation_id: Optional[int]) -> List[str]:
        relation_id = to_int(relation_id)
        if relation_id is None:
            return []
        rel = self.relations.get(relation_id)
        if rel is None:
            return []

        domains = []
        if getattr(rel, "domain", None):
            domains.extend(to_list(rel.domain))

        raw = getattr(rel, "raw", None) or {}
        for key in [
            "domain",
            "domains",
            "head_type",
            "head_class",
            "subject_type",
        ]:
            if key in raw:
                domains.extend(to_list(raw[key]))

        domains = deduplicate_clean(domains)
        domains = [d for d in domains if not is_numeric_like(d)]
        return domains

    def relation_range(self, relation_id: Optional[int]) -> List[str]:
        relation_id = to_int(relation_id)
        if relation_id is None:
            return []
        rel = self.relations.get(relation_id)
        if rel is None:
            return []

        ranges = []
        if getattr(rel, "range", None):
            ranges.extend(to_list(rel.range))

        raw = getattr(rel, "raw", None) or {}
        for key in [
            "range",
            "ranges",
            "tail_type",
            "tail_class",
            "tail_domain",
            "object_type",
        ]:
            if key in raw:
                ranges.extend(to_list(raw[key]))

        ranges = deduplicate_clean(ranges)
        ranges = [r for r in ranges if not is_numeric_like(r)]
        return ranges


class EvidenceCompressor:
    def __init__(
        self,
        dataset,
        max_evidence_num: int = 5,
        token_budget: int = 800,
        min_score: Optional[float] = None,
        keep_one_hop: bool = True,
        keep_paths: bool = True,
        remove_duplicate_text: bool = True,
        remove_duplicate_terminal: bool = True,
        remove_duplicate_relation_pattern: bool = True,
        include_score_in_text: bool = False,
        prefer_gold_tail_evidence: bool = True,
    ):
        self.dataset = dataset
        self.schema = DatasetSchemaHelper(dataset)

        self.max_evidence_num = max_evidence_num
        self.token_budget = token_budget
        self.min_score = min_score

        self.keep_one_hop = keep_one_hop
        self.keep_paths = keep_paths

        self.remove_duplicate_text = remove_duplicate_text
        self.remove_duplicate_terminal = remove_duplicate_terminal
        self.remove_duplicate_relation_pattern = remove_duplicate_relation_pattern

        self.include_score_in_text = include_score_in_text
        self.prefer_gold_tail_evidence = prefer_gold_tail_evidence

    def compress_evidence_list(
        self,
        filtered_evidence_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records = []
        for idx, evidence in enumerate(
            tqdm(filtered_evidence_list, desc="[EvidenceCompressor] Compressing", ncols=100)
        ):
            records.append(self.compress_one_evidence(evidence, query_index=idx))
        return records

    def compress_one_evidence(
        self,
        evidence: Dict[str, Any],
        query_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        query_head_id = to_int(evidence.get("query_head_id"))
        query_relation_id = to_int(evidence.get("query_relation_id"))
        gold_tail_id = to_int(evidence.get("gold_tail_id"))

        query_head_label = evidence.get(
            "query_head_label",
            self.schema.entity_label(query_head_id),
        )
        query_relation_label = evidence.get(
            "query_relation_label",
            self.schema.relation_label(query_relation_id),
        )
        gold_tail_label = evidence.get(
            "gold_tail_label",
            self.schema.entity_label(gold_tail_id),
        )

        candidates = self.build_candidate_evidence(evidence)

        selected = self.greedy_select(
            candidates=candidates,
            gold_tail_label=gold_tail_label,
        )

        evidence_text = self.build_evidence_text(selected)

        return {
            "query_index": query_index,
            "query": {
                "head_id": query_head_id,
                "head_label": query_head_label,
                "head_classes": self.schema.entity_classes(query_head_id),
                "relation_id": query_relation_id,
                "relation_label": query_relation_label,
                "relation_domain": self.schema.relation_domain(query_relation_id),
                "relation_range": self.schema.relation_range(query_relation_id),
            },
            "correct_answer": {
                "tail_id": gold_tail_id,
                "tail_label": gold_tail_label,
                "tail_classes": self.schema.entity_classes(gold_tail_id),
            },
            "key_evidence": selected,
            "evidence_text": evidence_text,
        }

    def build_candidate_evidence(
        self,
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates = []

        if self.keep_one_hop:
            one_hop_items = evidence.get("filtered_one_hop") or evidence.get("one_hop", [])
            for item in one_hop_items:
                candidates.append(self.convert_one_hop_item(item))

        if self.keep_paths:
            path_items = evidence.get("filtered_paths") or evidence.get("paths", [])
            for item in path_items:
                candidates.append(self.convert_path_item(item))

        if self.min_score is not None:
            candidates = [item for item in candidates if item["score"] >= self.min_score]

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def convert_one_hop_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        h_id = to_int(item.get("h_id") or item.get("head_id"))
        r_id = to_int(item.get("r_id") or item.get("relation_id"))
        t_id = to_int(item.get("t_id") or item.get("tail_id") or item.get("terminal_entity_id"))

        base_score = safe_get_score(item)

        text = item.get("text") or (
            f"{self.schema.entity_label(h_id)} "
            f"--[{self.schema.relation_label(r_id)}]--> "
            f"{self.schema.entity_label(t_id)}"
        )

        return {
            "type": "one_hop",
            "text": text,
            "base_score": base_score,
            "raw_score": base_score,
            "triple_id": {
                "head_id": h_id,
                "relation_id": r_id,
                "tail_id": t_id,
            },
            "triple_name": {
                "head": self.schema.entity_label(h_id),
                "relation": self.schema.relation_label(r_id),
                "tail": self.schema.entity_label(t_id),
            },
            "terminal_entity_id": t_id,
            "terminal_entity_label": self.schema.entity_label(t_id),
            "relation_pattern": [self.schema.relation_label(r_id)],
            "relation_pattern_ids": [r_id],
            "score": base_score,
        }

    def convert_path_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        base_score = safe_get_score(item)

        path_steps = item.get("path", [])
        terminal_entity_id = to_int(item.get("terminal_entity_id"))

        if terminal_entity_id is None and path_steps:
            terminal_entity_id = to_int(
                path_steps[-1].get("t_id") or path_steps[-1].get("tail_id")
            )

        relation_pattern = []
        relation_pattern_ids = []
        converted_steps = []

        for step in path_steps:
            h_id = to_int(step.get("h_id") or step.get("head_id"))
            r_id = to_int(step.get("r_id") or step.get("relation_id"))
            t_id = to_int(step.get("t_id") or step.get("tail_id"))

            h_label = step.get("h_label") or step.get("head") or self.schema.entity_label(h_id)
            r_label = step.get("r_label") or step.get("relation") or self.schema.relation_label(r_id)
            t_label = step.get("t_label") or step.get("tail") or self.schema.entity_label(t_id)

            relation_pattern.append(r_label)
            relation_pattern_ids.append(r_id)

            converted_steps.append(
                {
                    "head_id": h_id,
                    "relation_id": r_id,
                    "tail_id": t_id,
                    "head": h_label,
                    "relation": r_label,
                    "tail": t_label,
                }
            )

        text = item.get("text") or self.path_to_text(converted_steps)

        return {
            "type": "path",
            "text": text,
            "base_score": base_score,
            "raw_score": base_score,
            "path_length": len(converted_steps),
            "path": converted_steps,
            "terminal_entity_id": terminal_entity_id,
            "terminal_entity_label": self.schema.entity_label(terminal_entity_id),
            "relation_pattern": relation_pattern,
            "relation_pattern_ids": relation_pattern_ids,
            "score": base_score,
        }

    def sort_candidates_for_selection(
        self,
        candidates: List[Dict[str, Any]],
        gold_tail_label: Any,
    ) -> List[Dict[str, Any]]:
        if not self.prefer_gold_tail_evidence:
            return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

        for item in candidates:
            item["contains_gold_tail"] = evidence_contains_gold_tail_name(item, gold_tail_label)

        return sorted(
            candidates,
            key=lambda x: (
                int(x.get("contains_gold_tail", False)),
                x.get("score", 0.0),
            ),
            reverse=True,
        )

    def greedy_select(
        self,
        candidates: List[Dict[str, Any]],
        gold_tail_label: Any = None,
    ) -> List[Dict[str, Any]]:
        selected = []
        used_text = set()
        used_terminal = set()
        used_relation_pattern = set()
        current_tokens = 0

        ordered_candidates = self.sort_candidates_for_selection(
            candidates=candidates,
            gold_tail_label=gold_tail_label,
        )

        for item in ordered_candidates:
            text = item.get("text", "")
            text_key = normalize_text(text)
            terminal_key = item.get("terminal_entity_id")
            relation_pattern_key = tuple(item.get("relation_pattern_ids", []))

            if self.remove_duplicate_text and text_key in used_text:
                continue

            if self.remove_duplicate_terminal and terminal_key is not None and terminal_key in used_terminal:
                continue

            if (
                self.remove_duplicate_relation_pattern
                and relation_pattern_key
                and relation_pattern_key in used_relation_pattern
            ):
                continue

            item_tokens = approx_token_len(text)

            if self.token_budget > 0 and current_tokens + item_tokens > self.token_budget:
                continue

            selected.append(item)

            used_text.add(text_key)

            if terminal_key is not None:
                used_terminal.add(terminal_key)

            if relation_pattern_key:
                used_relation_pattern.add(relation_pattern_key)

            current_tokens += item_tokens

            if len(selected) >= self.max_evidence_num:
                break

        return selected

    @staticmethod
    def build_evidence_text(selected: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, item in enumerate(selected, start=1):
            gold_mark = " gold_tail=True" if item.get("contains_gold_tail") else ""
            lines.append(
                f"{idx}. [{item.get('type', 'evidence')}; "
                f"score={item.get('score', 0.0):.4f}] "
                f"{item.get('text')}"
            )
        return "\n".join(lines)

    @staticmethod
    def path_to_text(path_steps: List[Dict[str, Any]]) -> str:
        if not path_steps:
            return ""

        parts = []

        first_h = (
            path_steps[0].get("h_label")
            or path_steps[0].get("head")
            or path_steps[0].get("head_label")
        )

        parts.append(str(first_h))

        for step in path_steps:
            r = step.get("r_label") or step.get("relation") or step.get("relation_label")
            t = step.get("t_label") or step.get("tail") or step.get("tail_label")
            parts.append(f"--[{r}]-->")
            parts.append(str(t))

        return " ".join(parts)


def inspect_compressed_record(record: Dict[str, Any]) -> None:
    print("=" * 100)
    print("[Compressed Evidence Example]")
    print("=" * 100)

    query = record["query"]
    answer = record["correct_answer"]

    print(f"Query: ({query['head_label']}, {query['relation_label']}, ?)")
    print(f"Correct answer: {answer['tail_label']} | classes={answer.get('tail_classes')}")

    print("\n[Key Evidence]")
    for idx, item in enumerate(record.get("key_evidence", []), start=1):
        gold_mark = " | gold_tail=True" if item.get("contains_gold_tail") else ""
        print(
            f"{idx}. type={item.get('type')} | "
            f"score={item.get('score', 0.0):.4f}{gold_mark} | "
            f"{item.get('text')}"
        )

    print("\n[Evidence Text]")
    print(record.get("evidence_text", ""))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress filtered OD-KGC evidence with gold-tail preference but no forced gold-tail insertion."
    )

    parser.add_argument("--data_path", type=str, default="dataset/FB15k-237")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--import_path", type=str, default="import")

    parser.add_argument("--input_filtered_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--max_evidence_num", type=int, default=5)
    parser.add_argument("--token_budget", type=int, default=1500)
    parser.add_argument("--min_score", type=float, default=None)

    parser.add_argument("--include_score_in_text", action="store_true", default=False)
    parser.add_argument("--only_one_hop", action="store_true", default=False)
    parser.add_argument("--only_paths", action="store_true", default=False)

    parser.add_argument(
        "--no_prefer_gold_tail_evidence",
        action="store_true",
        default=False,
        help="Disable gold-tail preference during compression.",
    )

    parser.add_argument(
        "--no_remove_duplicate_terminal",
        action="store_true",
        default=False,
        help="Disable duplicate terminal entity filtering.",
    )

    parser.add_argument(
        "--no_remove_duplicate_relation_pattern",
        action="store_true",
        default=False,
        help="Disable duplicate relation-pattern filtering.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name
    import_root = Path(args.import_path)

    input_filtered_path = (
        Path(args.input_filtered_path)
        if args.input_filtered_path is not None
        else import_root
        / "evidence"
        / dataset_name
        / f"{args.split}_filtered_evidence.jsonl"
    )

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else import_root
        / "evidence"
        / dataset_name
        / f"{args.split}_compressed_evidence.jsonl"
    )

    print("=" * 100)
    print("[OD-KGC Evidence Compression]")
    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"Input filtered evidence: {input_filtered_path}")
    print(f"Output compressed evidence: {output_path}")
    print(f"Prefer gold-tail evidence: {not args.no_prefer_gold_tail_evidence}")
    print(f"Remove duplicate terminal: {not args.no_remove_duplicate_terminal}")
    print(f"Remove duplicate relation pattern: {not args.no_remove_duplicate_relation_pattern}")
    print("Forced gold-tail insertion: disabled")
    print("=" * 100)

    print("[EvidenceCompressor] Loading dataset...")
    loader = KGLoader(data_path)
    dataset = loader.load()

    print("[EvidenceCompressor] Loading filtered evidence...")
    filtered_evidence = load_jsonl(input_filtered_path)

    keep_one_hop = True
    keep_paths = True

    if args.only_one_hop:
        keep_paths = False

    if args.only_paths:
        keep_one_hop = False

    compressor = EvidenceCompressor(
        dataset=dataset,
        max_evidence_num=args.max_evidence_num,
        token_budget=args.token_budget,
        min_score=args.min_score,
        keep_one_hop=keep_one_hop,
        keep_paths=keep_paths,
        remove_duplicate_terminal=not args.no_remove_duplicate_terminal,
        remove_duplicate_relation_pattern=not args.no_remove_duplicate_relation_pattern,
        include_score_in_text=args.include_score_in_text,
        prefer_gold_tail_evidence=not args.no_prefer_gold_tail_evidence,
    )

    print("[EvidenceCompressor] Compressing evidence...")
    compressed = compressor.compress_evidence_list(filtered_evidence)

    print("[EvidenceCompressor] Saving compressed evidence...")
    save_jsonl(compressed, output_path)

    if compressed:
        inspect_compressed_record(compressed[0])

    print("\n[EvidenceCompressor] Done.")


if __name__ == "__main__":
    main()