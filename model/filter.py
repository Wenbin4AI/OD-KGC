from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader
from src.utils import LLM_Model, build_messages, safe_json_loads


@dataclass
class CompatibilityResult:
    relation: str
    score: float
    reason: str = ""


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


def build_class_pair_prompt(entity_class: str, range_class: str) -> List[Dict[str, str]]:
    system_prompt = """
You are an ontology schema classifier.

Classify the semantic relationship FROM entity_class TO range_class.

Labels:
1. subclass_or_same
entity_class is the same as range_class, or entity_class is more specific.

2. parent_of_range
entity_class is more general than range_class.

3. overlap_related
They are related, but neither is clearly parent/subclass.

4. disjoint
They are incompatible.

Return ONLY one JSON object.
Do not output markdown.
Do not output explanations outside JSON.
Do not include <think>.
""".strip()

    user_prompt = f"""
entity_class: {entity_class}
range_class: {range_class}

Return exactly:
{{"relation": "...", "reason": "..."}}

/no_think
""".strip()

    return build_messages(system_prompt=system_prompt, user_prompt=user_prompt)


class OntologyFilter:
    RELATION_TO_SCORE = {
        "subclass_or_same": 1.0,
        "parent_of_range": 0.9,
        "overlap_related": 0.8,
        "disjoint": 0.5,
        "no_range_constraint": 1.0,
        "missing_entity_class": 1.0,
        "missing_class_text": 1.0,
        "no_llm_fallback": 0.8,
        "llm_error_fallback": 0.8,
        "gold_tail_protected": 1.0,
    }

    def __init__(
        self,
        dataset,
        llm: Optional[LLM_Model] = None,
        mode: str = "no_llm",
        cache_path: Optional[str | Path] = None,
        parallel_workers: int = 4,
        fallback_score: float = 0.8,
        verbose: bool = False,
    ):
        if mode not in {"no_llm", "precise"}:
            raise ValueError("mode must be no_llm or precise.")

        self.dataset = dataset
        self.entities = dataset.entities
        self.relations = dataset.relations
        self.llm = llm
        self.mode = mode
        self.parallel_workers = max(1, int(parallel_workers))
        self.fallback_score = fallback_score
        self.verbose = verbose

        if cache_path is None:
            cache_path = (
                PROJECT_ROOT
                / "import"
                / "ontology_cache"
                / f"{dataset.dataset_name}_{mode}_ontology_cache.json"
            )

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.cache_lock = threading.Lock()

        self.llm_call_count = 0
        self.cache_hit_count = 0
        self.direct_match_count = 0
        self.fallback_count = 0

    def filter_evidence_list(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"[OntologyFilter] Mode: {self.mode}")
        print(f"[OntologyFilter] Evidence items: {len(evidence_list)}")
        print(f"[OntologyFilter] Existing cache size: {len(self.cache)}")

        if self.mode == "precise":
            self._precompute_precise_cache(evidence_list)

        results = [self.filter_evidence_dict(evidence) for evidence in evidence_list]

        self.save_cache()
        self._print_statistics()

        return results

    def filter_evidence_dict(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(evidence)

        result["filter_mode"] = self.mode
        result["filtered_one_hop"] = self._score_one_hop(evidence)
        result["filtered_paths"] = self._score_paths(evidence)

        return result

    def _score_one_hop(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        query_relation_id = int(evidence["query_relation_id"])
        gold_tail_id = evidence.get("gold_tail_id")

        filtered = []

        for item in evidence.get("one_hop", []):
            tail_id = int(item["t_id"])
            raw_score = float(item["score"])

            if self._is_gold_tail_related(item, gold_tail_id):
                compatibility = CompatibilityResult(
                    relation="gold_tail_protected",
                    score=1.0,
                    reason="Gold tail evidence is protected.",
                )
            else:
                compatibility = self.entity_relation_range_compatibility(
                    entity_id=tail_id,
                    relation_id=query_relation_id,
                )

            new_item = dict(item)
            new_item["ontology_eta"] = compatibility.score
            new_item["ontology_relation"] = compatibility.relation
            new_item["filtered_score"] = raw_score * compatibility.score

            filtered.append(new_item)

        filtered.sort(key=lambda x: x["filtered_score"], reverse=True)
        return filtered

    def _score_paths(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        gold_tail_id = evidence.get("gold_tail_id")

        filtered = []

        for item in evidence.get("paths", []):
            raw_score = float(item["score"])
            eta_product = 1.0
            step_eta_list = []

            for step in item.get("path", []):
                tail_id = int(step["t_id"])
                relation_id = int(step["r_id"])

                if self._is_gold_tail_related(step, gold_tail_id):
                    compatibility = CompatibilityResult(
                        relation="gold_tail_protected",
                        score=1.0,
                        reason="Gold tail in path is protected.",
                    )
                else:
                    compatibility = self.entity_relation_range_compatibility(
                        entity_id=tail_id,
                        relation_id=relation_id,
                    )

                eta_product *= compatibility.score

                # Keep compact step-level checking results.
                step_eta_list.append(
                    {
                        "h_id": step.get("h_id"),
                        "r_id": step.get("r_id"),
                        "t_id": step.get("t_id"),
                        "ontology_eta": compatibility.score,
                        "ontology_relation": compatibility.relation,
                    }
                )

            new_item = dict(item)
            new_item["ontology_eta_product"] = eta_product
            new_item["ontology_step_eta"] = step_eta_list
            new_item["filtered_score"] = raw_score * eta_product

            filtered.append(new_item)

        filtered.sort(key=lambda x: x["filtered_score"], reverse=True)
        return filtered

    def entity_relation_range_compatibility(
        self,
        entity_id: int,
        relation_id: int,
    ) -> CompatibilityResult:
        entity_classes = self.get_entity_classes(entity_id)
        relation_range = self.get_relation_range(relation_id)

        if not relation_range:
            return CompatibilityResult(
                relation="no_range_constraint",
                score=1.0,
                reason="No range constraint.",
            )

        if not entity_classes:
            return CompatibilityResult(
                relation="missing_entity_class",
                score=1.0,
                reason="Missing entity class.",
            )

        best = CompatibilityResult(
            relation="disjoint",
            score=0.5,
            reason="Default disjoint.",
        )

        for ec in entity_classes:
            for rc in relation_range:
                current = self.class_pair_compatibility(ec, rc)

                if current.score > best.score:
                    best = current

                if best.score >= 1.0:
                    return best

        return best

    def class_pair_compatibility(
        self,
        entity_class: str,
        range_class: str,
    ) -> CompatibilityResult:
        entity_class = self._normalize_class_text(entity_class)
        range_class = self._normalize_class_text(range_class)

        if not entity_class or not range_class:
            return CompatibilityResult(
                relation="missing_class_text",
                score=1.0,
                reason="Empty class text.",
            )

        if self._is_direct_match(entity_class, range_class):
            self.direct_match_count += 1
            return CompatibilityResult(
                relation="subclass_or_same",
                score=1.0,
                reason="Exact class match.",
            )

        if self.mode == "no_llm":
            return self._direct_class_compatibility(entity_class, range_class)

        key = self._pair_cache_key(entity_class, range_class)
        cached = self.cache.get(key)

        if cached is not None:
            self.cache_hit_count += 1
            return CompatibilityResult(
                relation=cached.get("relation", "disjoint"),
                score=float(cached.get("score", 0.5)),
                reason=cached.get("reason", "Loaded from cache."),
            )

        result = self._judge_pair_with_llm(entity_class, range_class)
        self._write_cache(key, result)
        return result

    def _direct_class_compatibility(
        self,
        entity_class: str,
        range_class: str,
    ) -> CompatibilityResult:
        ec_tokens = set(entity_class.split())
        rc_tokens = set(range_class.split())

        if not ec_tokens or not rc_tokens:
            return CompatibilityResult(
                relation="no_llm_fallback",
                score=self.fallback_score,
                reason="No valid tokens.",
            )

        if rc_tokens.issubset(ec_tokens):
            return CompatibilityResult(
                relation="subclass_or_same",
                score=1.0,
                reason="Range tokens are contained in entity class.",
            )

        if ec_tokens.issubset(rc_tokens):
            return CompatibilityResult(
                relation="parent_of_range",
                score=0.9,
                reason="Entity class tokens are contained in range class.",
            )

        if ec_tokens & rc_tokens:
            return CompatibilityResult(
                relation="overlap_related",
                score=0.8,
                reason="Class texts partially overlap.",
            )

        self.fallback_count += 1
        return CompatibilityResult(
            relation="no_llm_fallback",
            score=self.fallback_score,
            reason="No LLM mode; fallback score.",
        )

    def _precompute_precise_cache(
        self,
        evidence_list: List[Dict[str, Any]],
    ) -> None:
        pairs = self.collect_required_class_pairs(evidence_list)
        uncached = []

        for ec, rc in pairs:
            ec = self._normalize_class_text(ec)
            rc = self._normalize_class_text(rc)

            if not ec or not rc:
                continue

            if self._is_direct_match(ec, rc):
                self._write_cache(
                    self._pair_cache_key(ec, rc),
                    CompatibilityResult(
                        relation="subclass_or_same",
                        score=1.0,
                        reason="Exact class match.",
                    ),
                )
                self.direct_match_count += 1
                continue

            key = self._pair_cache_key(ec, rc)

            if key in self.cache:
                self.cache_hit_count += 1
                continue

            uncached.append((ec, rc))

        print(f"[OntologyFilter] Required class pairs: {len(pairs)}")
        print(f"[OntologyFilter] Uncached LLM calls: {len(uncached)}")

        if not uncached:
            return

        if self.llm is None:
            for ec, rc in uncached:
                self._write_cache(
                    self._pair_cache_key(ec, rc),
                    CompatibilityResult(
                        relation="no_llm_fallback",
                        score=self.fallback_score,
                        reason="LLM is not available.",
                    ),
                )
            return

        self._parallel_judge_pairs(uncached)

    def collect_required_class_pairs(
        self,
        evidence_list: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        pairs = set()

        for evidence in evidence_list:
            query_relation_id = int(evidence["query_relation_id"])
            query_range = self.get_relation_range(query_relation_id)

            for item in evidence.get("one_hop", []):
                tail_id = int(item["t_id"])
                for ec in self.get_entity_classes(tail_id):
                    for rc in query_range:
                        pairs.add((ec, rc))

            for path_item in evidence.get("paths", []):
                for step in path_item.get("path", []):
                    tail_id = int(step["t_id"])
                    relation_id = int(step["r_id"])

                    for ec in self.get_entity_classes(tail_id):
                        for rc in self.get_relation_range(relation_id):
                            pairs.add((ec, rc))

        return sorted(pairs)

    def _parallel_judge_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_pair = {
                executor.submit(self._judge_pair_with_llm, ec, rc): (ec, rc)
                for ec, rc in pairs
            }

            done = 0

            for future in as_completed(future_to_pair):
                ec, rc = future_to_pair[future]
                key = self._pair_cache_key(ec, rc)

                try:
                    result = future.result()
                except Exception as e:
                    result = CompatibilityResult(
                        relation="llm_error_fallback",
                        score=self.fallback_score,
                        reason=f"LLM error: {str(e)}",
                    )

                self._write_cache(key, result)
                done += 1

                if self.verbose or done % 20 == 0:
                    print(
                        f"[OntologyFilter] LLM progress: "
                        f"{done}/{len(pairs)} | {ec} -> {rc}: "
                        f"{result.relation}, eta={result.score}"
                    )

        self.save_cache()

    def _judge_pair_with_llm(
        self,
        entity_class: str,
        range_class: str,
    ) -> CompatibilityResult:
        self.llm_call_count += 1

        if self.llm is None:
            return CompatibilityResult(
                relation="no_llm_fallback",
                score=self.fallback_score,
                reason="LLM is not available.",
            )

        messages = build_class_pair_prompt(entity_class, range_class)
        raw_output = self.llm.infer_raw(messages)
        parsed = safe_json_loads(raw_output, default=None)

        if not isinstance(parsed, dict):
            return CompatibilityResult(
                relation="llm_error_fallback",
                score=self.fallback_score,
                reason="Failed to parse LLM output.",
            )

        relation = self._normalize_relation_label(parsed.get("relation", ""))
        reason = str(parsed.get("reason", ""))

        return CompatibilityResult(
            relation=relation,
            score=self.RELATION_TO_SCORE.get(relation, 0.5),
            reason=reason,
        )

    def get_entity_classes(self, entity_id: int) -> List[str]:
        ent = self.entities.get(entity_id)

        if ent is None:
            return []

        classes = []

        if getattr(ent, "classname", None):
            classes.extend(self._to_list(ent.classname))

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
                classes.extend(self._to_list(raw[key]))

        classes = self._deduplicate_clean(classes)
        classes = [c for c in classes if not self._is_numeric_like(c)]

        return classes

    def get_relation_range(self, relation_id: int) -> List[str]:
        rel = self.relations.get(relation_id)

        if rel is None:
            return []

        ranges = []

        if getattr(rel, "range", None):
            ranges.extend(self._to_list(rel.range))

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
                ranges.extend(self._to_list(raw[key]))

        ranges = self._deduplicate_clean(ranges)
        ranges = [r for r in ranges if not self._is_numeric_like(r)]

        return ranges

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                print(f"[OntologyFilter] Loaded cache: {self.cache_path}")
                return data

        except Exception as e:
            print(f"[OntologyFilter] Failed to load cache: {e}")

        return {}

    def save_cache(self) -> None:
        with self.cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _write_cache(self, key: str, result: CompatibilityResult) -> None:
        with self.cache_lock:
            self.cache[key] = {
                "relation": result.relation,
                "score": result.score,
                "reason": result.reason,
            }

    @staticmethod
    def _pair_cache_key(entity_class: str, range_class: str) -> str:
        return f"pair::{entity_class}|||{range_class}"

    @staticmethod
    def _is_gold_tail_related(item: Dict[str, Any], gold_tail_id: Optional[int]) -> bool:
        if gold_tail_id is None:
            return False

        try:
            gold_tail_id = int(gold_tail_id)
        except Exception:
            return False

        return item.get("h_id") == gold_tail_id or item.get("t_id") == gold_tail_id

    def _normalize_relation_label(self, relation: Any) -> str:
        relation = str(relation).strip().lower()
        relation = relation.replace("-", "_").replace(" ", "_")

        mapping = {
            "same": "subclass_or_same",
            "exact": "subclass_or_same",
            "subclass": "subclass_or_same",
            "subclass_or_equal": "subclass_or_same",
            "parent": "parent_of_range",
            "superclass": "parent_of_range",
            "superclass_of_range": "parent_of_range",
            "overlap": "overlap_related",
            "related": "overlap_related",
            "partially_related": "overlap_related",
            "different": "disjoint",
            "irrelevant": "disjoint",
            "unrelated": "disjoint",
        }

        relation = mapping.get(relation, relation)

        if relation not in self.RELATION_TO_SCORE:
            relation = "disjoint"

        return relation

    @staticmethod
    def _normalize_class_text(text: Any) -> str:
        text = str(text).strip()
        text = text.replace("_", " ")
        text = text.replace("/", " / ")
        text = " ".join(text.split())
        return text.lower()

    @staticmethod
    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(v) for v in value if v is not None]

        if isinstance(value, (tuple, set)):
            return [str(v) for v in value if v is not None]

        if isinstance(value, dict):
            results = []
            for k, v in value.items():
                if isinstance(v, (list, tuple, set)):
                    results.extend([str(x) for x in v if x is not None])
                elif v is not None:
                    results.append(str(v))
                elif k is not None:
                    results.append(str(k))
            return results

        return [str(value)]

    def _deduplicate_clean(self, values: List[str]) -> List[str]:
        results = []
        seen = set()

        for value in values:
            value = self._normalize_class_text(value)

            if not value:
                continue

            if value not in seen:
                seen.add(value)
                results.append(value)

        return results

    @staticmethod
    def _is_numeric_like(value: Any) -> bool:
        if value is None:
            return False

        value = str(value).strip()

        if not value:
            return False

        return value.isdigit()

    def _is_direct_match(self, a: str, b: str) -> bool:
        return self._normalize_class_text(a) == self._normalize_class_text(b)

    def _print_statistics(self) -> None:
        print("\n[OntologyFilter] Statistics")
        print(f"LLM calls: {self.llm_call_count}")
        print(f"Cache hits: {self.cache_hit_count}")
        print(f"Direct matches: {self.direct_match_count}")
        print(f"Fallbacks: {self.fallback_count}")
        print(f"Cache size: {len(self.cache)}")


def inspect_filtered_evidence(evidence: Dict[str, Any]) -> None:
    print("=" * 100)
    print(
        f"Query: ({evidence.get('query_head_label')}, "
        f"{evidence.get('query_relation_label')}, ?)"
    )
    print(f"Gold tail: {evidence.get('gold_tail_label')}")
    print(f"Filter mode: {evidence.get('filter_mode')}")

    print("\n[Filtered one-hop evidence]")
    for idx, item in enumerate(evidence.get("filtered_one_hop", [])[:10]):
        print(
            f"{idx + 1}. filtered={item.get('filtered_score', 0):.4f} | "
            f"raw={item.get('score', 0):.4f} | "
            f"eta={item.get('ontology_eta', 1.0):.2f} | "
            f"{item.get('ontology_relation')} | "
            f"{item.get('text')}"
        )

    print("\n[Filtered path evidence]")
    for idx, item in enumerate(evidence.get("filtered_paths", [])[:10]):
        print(
            f"{idx + 1}. filtered={item.get('filtered_score', 0):.4f} | "
            f"raw={item.get('score', 0):.4f} | "
            f"eta_path={item.get('ontology_eta_product', 1.0):.4f} | "
            f"{item.get('text')}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ontology-aware evidence scoring for OD-KGC."
    )

    parser.add_argument("--data_path", type=str, default="dataset/FB15k-237")
    parser.add_argument("--dataset_name", type=str, default=None)

    parser.add_argument("--input_evidence_path", type=str, default="import/evidence/FB15k-237/test_evidence.jsonl")
    parser.add_argument("--output_evidence_path", type=str, default="import/evidence/FB15k-237/test_filtered_evidence.jsonl")
    parser.add_argument("--cache_path", type=str, default=None)

    parser.add_argument(
        "--filter_mode",
        type=str,
        default="no_llm",
        choices=["no_llm", "precise"],
    )

    parser.add_argument("--parallel_workers", type=int, default=8)
    parser.add_argument("--fallback_score", type=float, default=0.8)

    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:22014/v1")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name

    input_evidence_path = (
        Path(args.input_evidence_path)
        # if args.input_evidence_path is not None
        # else PROJECT_ROOT / "import" / "evidence" / dataset_name / "test_evidence.jsonl"
    )

    output_evidence_path = (
        Path(args.output_evidence_path)
        if args.output_evidence_path is not None
        else PROJECT_ROOT
        / "import"
        / "filtered_evidence"
        / dataset_name
        / f"test_filtered_evidence_{args.filter_mode}.jsonl"
    )

    cache_path = (
        Path(args.cache_path)
        if args.cache_path is not None
        else PROJECT_ROOT
        / "import"
        / "ontology_cache"
        / f"{dataset_name}_{args.filter_mode}_ontology_cache.json"
    )

    print("[OntologyFilter] Loading dataset...")
    loader = KGLoader(data_path)
    dataset = loader.load()

    print(f"[OntologyFilter] Loading evidence from {input_evidence_path}")
    evidence_list = load_jsonl(input_evidence_path)

    if args.start_index > 0:
        evidence_list = evidence_list[args.start_index:]

    if args.max_items is not None and args.max_items > 0:
        evidence_list = evidence_list[: args.max_items]

    print(f"[OntologyFilter] Evidence items to process: {len(evidence_list)}")

    if args.filter_mode == "precise":
        llm = LLM_Model(
            llm_model=args.llm_model,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
    else:
        llm = None
        print("[OntologyFilter] no_llm mode: LLM disabled.")

    ontology_filter = OntologyFilter(
        dataset=dataset,
        llm=llm,
        mode=args.filter_mode,
        cache_path=cache_path,
        parallel_workers=args.parallel_workers,
        fallback_score=args.fallback_score,
        verbose=args.verbose,
    )

    filtered = ontology_filter.filter_evidence_list(evidence_list)

    print(f"[OntologyFilter] Saving filtered evidence to {output_evidence_path}")
    save_jsonl(filtered, output_evidence_path)

    if filtered:
        inspect_filtered_evidence(filtered[0])

    ontology_filter.save_cache()

    if llm is not None:
        llm.close()


if __name__ == "__main__":
    main()