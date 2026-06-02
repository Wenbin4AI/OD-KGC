from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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


def normalize_text(value: Any) -> str:
    value = str(value).strip()
    value = value.replace("_", " ")
    value = value.replace("/", " / ")
    value = " ".join(value.split())
    return value.lower()


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
                results.extend(
                    [str(x) for x in v if x is not None and str(x).strip()]
                )
            elif v is not None and str(v).strip():
                results.append(str(v))
            elif k is not None and str(k).strip():
                results.append(str(k))
        return results

    return [str(value)]


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


def is_numeric_like(value: Any) -> bool:
    if value is None:
        return False

    value = str(value).strip()

    if not value:
        return False

    return value.isdigit()


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


class DatasetSchemaHelper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.entities = dataset.entities
        self.relations = dataset.relations

    def entity_label(self, entity_id: Optional[int]) -> str:
        if entity_id is None:
            return "None"

        ent = self.entities.get(int(entity_id))

        if ent is None:
            return f"[UnknownEntity:{entity_id}]"

        return getattr(ent, "label", None) or str(entity_id)

    def relation_label(self, relation_id: Optional[int]) -> str:
        if relation_id is None:
            return "None"

        rel = self.relations.get(int(relation_id))

        if rel is None:
            return f"[UnknownRelation:{relation_id}]"

        return getattr(rel, "label", None) or str(relation_id)

    def entity_classes(self, entity_id: Optional[int]) -> List[str]:
        if entity_id is None:
            return []

        ent = self.entities.get(int(entity_id))

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
        if relation_id is None:
            return []

        rel = self.relations.get(int(relation_id))

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
        if relation_id is None:
            return []

        rel = self.relations.get(int(relation_id))

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

    def compress_evidence_list(
        self,
        filtered_evidence_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records = []

        for idx, evidence in enumerate(filtered_evidence_list):
            record = self.compress_one_evidence(evidence, query_index=idx)
            records.append(record)

            if (idx + 1) % 100 == 0:
                print(
                    f"[EvidenceCompressor] Compressed "
                    f"{idx + 1}/{len(filtered_evidence_list)} items."
                )

        return records

    def compress_one_evidence(
        self,
        evidence: Dict[str, Any],
        query_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        query_head_id = int(evidence["query_head_id"])
        query_relation_id = int(evidence["query_relation_id"])

        gold_tail_id = evidence.get("gold_tail_id")
        if gold_tail_id is not None:
            gold_tail_id = int(gold_tail_id)

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
        selected = self.greedy_select(candidates)
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
            one_hop_items = evidence.get("filtered_one_hop")
            if one_hop_items is None:
                one_hop_items = evidence.get("one_hop", [])

            for item in one_hop_items:
                candidates.append(self.convert_one_hop_item(item))

        if self.keep_paths:
            path_items = evidence.get("filtered_paths")
            if path_items is None:
                path_items = evidence.get("paths", [])

            for item in path_items:
                candidates.append(self.convert_path_item(item))

        if self.min_score is not None:
            candidates = [
                item for item in candidates
                if item["score"] >= self.min_score
            ]

        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates

    def convert_one_hop_item(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        h_id = int(item["h_id"])
        r_id = int(item["r_id"])
        t_id = int(item["t_id"])

        score = safe_get_score(item)
        raw_score = float(item.get("score", score))
        eta = float(item.get("ontology_eta", 1.0))

        h_label = item.get("h_label", self.schema.entity_label(h_id))
        r_label = item.get("r_label", self.schema.relation_label(r_id))
        t_label = item.get("t_label", self.schema.entity_label(t_id))

        text = item.get("text")
        if not text:
            text = f"{h_label} --[{r_label}]--> {t_label}"

        if self.include_score_in_text:
            text = f"{text} (score={score:.4f}, eta={eta:.2f})"

        return {
            "type": "one_hop",
            "text": text,
            "score": score,
            "raw_score": raw_score,
            "ontology_eta": eta,
            "triple_id": {
                "head_id": h_id,
                "relation_id": r_id,
                "tail_id": t_id,
            },
            "triple_name": {
                "head": h_label,
                "relation": r_label,
                "tail": t_label,
            },
            "terminal_entity_id": t_id,
            "terminal_entity_label": t_label,
            "relation_pattern": [r_label],
            "relation_pattern_ids": [r_id],
        }

    def convert_path_item(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = safe_get_score(item)
        raw_score = float(item.get("score", score))
        eta_product = float(item.get("ontology_eta_product", 1.0))

        path_steps = item.get("path", [])

        terminal_entity_id = item.get("terminal_entity_id")
        if terminal_entity_id is None and path_steps:
            terminal_entity_id = path_steps[-1].get("t_id")

        if terminal_entity_id is not None:
            terminal_entity_id = int(terminal_entity_id)

        terminal_entity_label = item.get(
            "terminal_entity_label",
            self.schema.entity_label(terminal_entity_id),
        )

        text = item.get("text")
        if not text:
            text = self.path_to_text(path_steps)

        if self.include_score_in_text:
            text = f"{text} (score={score:.4f}, eta_path={eta_product:.2f})"

        relation_pattern = []
        relation_pattern_ids = []
        converted_steps = []

        ontology_step_eta = item.get("ontology_step_eta", [])

        for idx, step in enumerate(path_steps):
            h_id = int(step["h_id"])
            r_id = int(step["r_id"])
            t_id = int(step["t_id"])

            h_label = step.get("h_label", self.schema.entity_label(h_id))
            r_label = step.get("r_label", self.schema.relation_label(r_id))
            t_label = step.get("t_label", self.schema.entity_label(t_id))

            relation_pattern.append(r_label)
            relation_pattern_ids.append(r_id)

            step_info = {
                "head_id": h_id,
                "relation_id": r_id,
                "tail_id": t_id,
                "head": h_label,
                "relation": r_label,
                "tail": t_label,
            }

            if idx < len(ontology_step_eta):
                step_info["ontology_eta"] = ontology_step_eta[idx].get("ontology_eta")
                step_info["ontology_relation"] = ontology_step_eta[idx].get(
                    "ontology_relation"
                )

            converted_steps.append(step_info)

        return {
            "type": "path",
            "text": text,
            "score": score,
            "raw_score": raw_score,
            "ontology_eta_product": eta_product,
            "path_length": len(path_steps),
            "path": converted_steps,
            "terminal_entity_id": terminal_entity_id,
            "terminal_entity_label": terminal_entity_label,
            "relation_pattern": relation_pattern,
            "relation_pattern_ids": relation_pattern_ids,
        }

    def greedy_select(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected = []

        used_text = set()
        used_terminal = set()
        used_relation_pattern = set()

        current_tokens = 0

        for item in candidates:
            text = item.get("text", "")
            text_key = normalize_text(text)
            terminal_key = item.get("terminal_entity_id")
            relation_pattern_key = tuple(item.get("relation_pattern_ids", []))

            if self.remove_duplicate_text and text_key in used_text:
                continue

            if (
                self.remove_duplicate_terminal
                and terminal_key is not None
                and terminal_key in used_terminal
            ):
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
            evidence_type = item.get("type", "evidence")
            score = float(item.get("score", 0.0))
            text = item.get("text", "")

            lines.append(
                f"{idx}. [{evidence_type}; score={score:.4f}] {text}"
            )

        return "\n".join(lines)

    @staticmethod
    def path_to_text(path_steps: List[Dict[str, Any]]) -> str:
        if not path_steps:
            return ""

        parts = []

        first_h = path_steps[0].get("h_label") or path_steps[0].get("head")
        parts.append(str(first_h))

        for step in path_steps:
            r = step.get("r_label") or step.get("relation")
            t = step.get("t_label") or step.get("tail")

            parts.append(f"--[{r}]-->")
            parts.append(str(t))

        return " ".join(parts)


def inspect_compressed_record(record: Dict[str, Any]) -> None:
    print("=" * 100)
    print("[Compressed Evidence Example]")
    print("=" * 100)

    query = record["query"]
    answer = record["correct_answer"]

    print(
        f"Query: ({query['head_label']}, "
        f"{query['relation_label']}, ?)"
    )
    print(
        f"Correct answer: {answer['tail_label']} | "
        f"classes={answer.get('tail_classes')}"
    )

    print("\n[Key Evidence]")
    for idx, item in enumerate(record.get("key_evidence", []), start=1):
        print(
            f"{idx}. type={item.get('type')} | "
            f"score={item.get('score'):.4f} | "
            f"{item.get('text')}"
        )

    print("\n[Evidence Text]")
    print(record.get("evidence_text", ""))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress filtered OD-KGC evidence into compact key evidence."
    )

    parser.add_argument("--data_path", type=str, default="dataset/FB15k-237")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--import_path", type=str, default="import")

    parser.add_argument("--input_filtered_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--max_evidence_num", type=int, default=5)
    parser.add_argument("--token_budget", type=int, default=800)
    parser.add_argument("--min_score", type=float, default=None)

    parser.add_argument("--include_score_in_text", action="store_true", default=False)

    parser.add_argument("--only_one_hop", action="store_true", default=False)
    parser.add_argument("--only_paths", action="store_true", default=False)

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
        / "filtered_evidence"
        / dataset_name
        / f"{args.split}_filtered_evidence_no_llm.jsonl"
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
        include_score_in_text=args.include_score_in_text,
    )

    compressed = compressor.compress_evidence_list(filtered_evidence)

    print("[EvidenceCompressor] Saving compressed evidence...")
    save_jsonl(compressed, output_path)

    if compressed:
        inspect_compressed_record(compressed[0])

    print("\n[EvidenceCompressor] Done.")


if __name__ == "__main__":
    main()