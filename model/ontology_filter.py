# OD-KGC/model/ontology_filter.py

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader
from src.utils import (
    LLM_Model,
    build_messages,
    safe_json_loads,
    load_jsonl,
    save_jsonl,
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class CompatibilityResult:
    relation: str
    score: float
    reason: str = ""


# ============================================================
# Prompt: precise pair judgment
# ============================================================

def build_class_pair_prompt(
    entity_class: str,
    range_class: str,
) -> List[Dict[str, str]]:

    system_prompt = """
You are an ontology schema classifier.

Your task is to classify the relationship FROM the entity class TO the relation range class.

Important direction:
- entity_class -> range_class

Labels:

1. subclass_or_same
Use this ONLY when entity_class is the same as range_class,
or entity_class is more specific than range_class.
Example:
entity_class = "city", range_class = "location" => subclass_or_same
entity_class = "basketball player", range_class = "person" => subclass_or_same

2. parent_of_range
Use this when entity_class is more general than range_class.
Example:
entity_class = "location", range_class = "city" => parent_of_range
entity_class = "place", range_class = "city" => parent_of_range
entity_class = "geographical location", range_class = "city" => parent_of_range
entity_class = "time", range_class = "month" => parent_of_range

3. overlap_related
Use this when the two classes are related but neither is clearly more general.
Example:
entity_class = "town", range_class = "city" => overlap_related
entity_class = "actor", range_class = "director" => overlap_related

4. disjoint
Use this when the two classes are incompatible.
Example:
entity_class = "river", range_class = "city" => disjoint
entity_class = "person", range_class = "chemical compound" => disjoint

Return ONLY one valid JSON object.
Do not include <think>.
Do not explain.
Do not output markdown.
Do not output any text before or after JSON.
""".strip()

    user_prompt = f"""
entity_class: {entity_class}
range_class: {range_class}

Return exactly:
{{"relation": "...", "reason": "..."}}

/no_think
""".strip()

    return build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# ============================================================
# Prompt: fast query-level judgment
# ============================================================

def build_fast_query_prompt(
    query_relation_label: str,
    range_classes: List[str],
    entity_classes: List[str],
) -> List[Dict[str, str]]:
    system_prompt = """
You are an ontology class-matching classifier.

Task:
For each entity class, judge its semantic relationship TO the relation range class.

Important:
- Only compare entity class with range classes.
- Do NOT reason about whether the entity can be the subject/member/owner of the relation.
- Judge whether the entity class itself is compatible with the range class.
- Return JSON only.
- Do not output <think>.
- Do not output explanations.
- Do not output markdown.

Labels:
subclass_or_same: entity_class is the same as or more specific than the range class.
parent_of_range: entity_class is more general than the range class.
overlap_related: related but neither is clearly parent/subclass.
disjoint: incompatible.

Examples:
range = ["city"]
entity_class = "city" -> subclass_or_same
entity_class = "location" -> parent_of_range
entity_class = "place" -> parent_of_range
entity_class = "town" -> overlap_related
entity_class = "river" -> disjoint
entity_class = "sports team" -> disjoint

range = ["month"]
entity_class = "month" -> subclass_or_same
entity_class = "time" -> parent_of_range

range = ["music group"]
entity_class = "band" -> subclass_or_same
entity_class = "musicgroup" -> subclass_or_same
entity_class = "person" -> disjoint
entity_class = "musical instrument" -> disjoint
""".strip()

    user_prompt = f"""
Range classes:
{json.dumps(range_classes, ensure_ascii=False)}

Entity classes:
{json.dumps(entity_classes, ensure_ascii=False)}

Return exactly this compact JSON format:
{{
  "judgments": {{
    "entity class 1": "label",
    "entity class 2": "label"
  }}
}}

/no_think
""".strip()

    return build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


# ============================================================
# Ontology Filter
# ============================================================

class OntologyFilter:
    """
    Ontology-constrained filtering for OD-KGC evidence.

    Modes:
        precise:
            Judge each unique (entity_class, range_class) pair.
            Cache avoids repeated LLM calls.
            Parallel LLM calls are supported.

        fast_query:
            For each query, call LLM once to judge all evidence classes
            against the query relation range.
            This is faster but approximate for multi-hop paths.

        no_llm:
            Do not call LLM. Use exact match and fallback scores only.
    """

    RELATION_TO_SCORE = {
        "subclass_or_same": 1.0,
        "parent_of_range": 0.9,
        "overlap_related": 0.8,
        "disjoint": 0.5,
        "no_range_constraint": 1.0,
        "missing_entity_class": 0.8,
        "missing_class_text": 0.8,
        "no_llm_fallback": 0.8,
        "llm_error_fallback": 0.8,
    }

    def __init__(
        self,
        dataset,
        llm: Optional[LLM_Model] = None,
        cache_path: Optional[str | Path] = None,
        mode: str = "precise",
        parallel_workers: int = 4,
        missing_range_score: float = 1.0,
        missing_entity_class_score: float = 0.8,
        direct_match_score: float = 1.0,
        fallback_score: float = 0.8,
        top_k_one_hop: int = 10,
        top_k_paths: int = 10,
        continue_on_llm_error: bool = True,
        verbose: bool = False,
    ):
        if mode not in {"precise", "fast_query", "no_llm"}:
            raise ValueError("mode must be precise, fast_query, or no_llm.")

        self.dataset = dataset
        self.entities = dataset.entities
        self.relations = dataset.relations

        self.llm = llm
        self.mode = mode
        self.parallel_workers = parallel_workers

        self.missing_range_score = missing_range_score
        self.missing_entity_class_score = missing_entity_class_score
        self.direct_match_score = direct_match_score
        self.fallback_score = fallback_score

        self.top_k_one_hop = top_k_one_hop
        self.top_k_paths = top_k_paths
        self.continue_on_llm_error = continue_on_llm_error
        self.verbose = verbose

        if cache_path is None:
            cache_path = (
                PROJECT_ROOT
                / "import"
                / "ontology_cache"
                / f"{dataset.dataset_name}_ontology_cache.json"
            )

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.cache_lock = threading.Lock()

        self.llm_call_count = 0
        self.cache_hit_count = 0
        self.direct_match_count = 0
        self.fallback_count = 0

    # ========================================================
    # Main APIs
    # ========================================================

    @staticmethod
    def _is_numeric_like(value: Any) -> bool:
        if value is None:
            return False

        value = str(value).strip()

        if value == "":
            return False

        return value.isdigit()

    def filter_evidence_list(
        self,
        evidence_list: List[Dict[str, Any]],
        save_cache_every: int = 100,
    ) -> List[Dict[str, Any]]:
        print(f"[OntologyFilter] Mode: {self.mode}")
        print(f"[OntologyFilter] Evidence items: {len(evidence_list)}")
        print(f"[OntologyFilter] Parallel workers: {self.parallel_workers}")
        print(f"[OntologyFilter] Existing cache size: {len(self.cache)}")

        if self.mode == "precise":
            self._precompute_precise_cache(evidence_list)

        elif self.mode == "fast_query":
            self._precompute_fast_query_cache(evidence_list)

        elif self.mode == "no_llm":
            print("[OntologyFilter] no_llm mode: skip all LLM calls.")

        results = []

        for idx, evidence in enumerate(evidence_list):
            filtered = self.filter_evidence_dict(evidence)
            results.append(filtered)

            if (idx + 1) % save_cache_every == 0:
                self.save_cache()
                print(
                    f"[OntologyFilter] Filtered {idx + 1}/{len(evidence_list)} items."
                )

        self.save_cache()
        self._print_statistics()

        return results

    def filter_evidence_dict(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "fast_query":
            return self._filter_evidence_dict_fast_query(evidence)

        return self._filter_evidence_dict_precise(evidence)

    # ========================================================
    # Precise mode: precompute all unique pair judgments
    # ========================================================

    def _precompute_precise_cache(
        self,
        evidence_list: List[Dict[str, Any]],
    ) -> None:
        required_pairs = self.collect_required_class_pairs(evidence_list)
        uncached_pairs = []

        for entity_class, range_class in required_pairs:
            key = self._pair_cache_key(entity_class, range_class)

            if self._is_direct_match(entity_class, range_class):
                self._write_cache(
                    key,
                    CompatibilityResult(
                        relation="subclass_or_same",
                        score=self.direct_match_score,
                        reason="Exact class match.",
                    ),
                )
                self.direct_match_count += 1
                continue

            if key in self.cache:
                self.cache_hit_count += 1
                continue

            if self.mode == "no_llm" or self.llm is None:
                self._write_cache(
                    key,
                    CompatibilityResult(
                        relation="no_llm_fallback",
                        score=self.fallback_score,
                        reason="No LLM used; fallback score.",
                    ),
                )
                self.fallback_count += 1
                continue

            uncached_pairs.append((entity_class, range_class))

        print(f"[OntologyFilter] Required class pairs: {len(required_pairs)}")
        print(f"[OntologyFilter] Uncached LLM pair calls: {len(uncached_pairs)}")

        if not uncached_pairs:
            return

        self._parallel_judge_pairs(uncached_pairs)

    def collect_required_class_pairs(
        self,
        evidence_list: List[Dict[str, Any]],
    ) -> List[Tuple[str, str]]:
        pairs = set()

        for evidence in evidence_list:
            query_relation_id = int(evidence["query_relation_id"])

            # one-hop: tail class vs query relation range
            query_range = self.get_relation_range(query_relation_id)

            for item in evidence.get("one_hop", []):
                tail_id = int(item["t_id"])
                entity_classes = self.get_entity_classes(tail_id)

                for ec in entity_classes:
                    for rc in query_range:
                        pairs.add((ec, rc))

            # path: each step tail class vs step relation range
            for path_item in evidence.get("paths", []):
                for step in path_item.get("path", []):
                    tail_id = int(step["t_id"])
                    relation_id = int(step["r_id"])

                    entity_classes = self.get_entity_classes(tail_id)
                    relation_range = self.get_relation_range(relation_id)

                    for ec in entity_classes:
                        for rc in relation_range:
                            pairs.add((ec, rc))

        return sorted(pairs)

    def _parallel_judge_pairs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> None:
        max_workers = max(1, int(self.parallel_workers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(self._judge_pair_with_llm, ec, rc): (ec, rc)
                for ec, rc in pairs
            }

            done_count = 0

            for future in as_completed(future_to_pair):
                ec, rc = future_to_pair[future]
                key = self._pair_cache_key(ec, rc)

                try:
                    result = future.result()
                except Exception as e:
                    if self.continue_on_llm_error:
                        result = CompatibilityResult(
                            relation="llm_error_fallback",
                            score=self.fallback_score,
                            reason=f"LLM error: {str(e)}",
                        )
                    else:
                        raise

                self._write_cache(key, result)

                done_count += 1

                if self.verbose or done_count % 20 == 0:
                    print(
                        f"[OntologyFilter] Pair LLM progress: "
                        f"{done_count}/{len(pairs)} | "
                        f"{ec} -> {rc}: {result.relation}, eta={result.score}"
                    )

        self.save_cache()

    def _judge_pair_with_llm(
        self,
        entity_class: str,
        range_class: str,
    ) -> CompatibilityResult:
        self.llm_call_count += 1

        messages = build_class_pair_prompt(
            entity_class=entity_class,
            range_class=range_class,
        )

        raw_output = self.llm.infer_raw(messages)
        parsed = safe_json_loads(raw_output, default=None)

        if not isinstance(parsed, dict):
            return CompatibilityResult(
                relation="llm_error_fallback",
                score=self.fallback_score,
                reason=f"Failed to parse LLM output: {raw_output}",
            )

        relation = self._normalize_relation_label(parsed.get("relation", ""))
        reason = str(parsed.get("reason", ""))

        return CompatibilityResult(
            relation=relation,
            score=self.RELATION_TO_SCORE.get(relation, 0.5),
            reason=reason,
        )

    # ========================================================
    # Fast query mode: one LLM call per query
    # ========================================================

    def _precompute_fast_query_cache(
        self,
        evidence_list: List[Dict[str, Any]],
    ) -> None:
        tasks = []

        for idx, evidence in enumerate(evidence_list):
            query_key, query_task = self._build_fast_query_task(evidence, idx)

            if query_task is None:
                continue

            if query_key in self.cache:
                self.cache_hit_count += 1
                continue

            if self.llm is None:
                fallback = {
                    "mode": "fast_query",
                    "judgments": {},
                    "reason": "No LLM used; fallback will be applied.",
                }
                self._write_raw_cache(query_key, fallback)
                self.fallback_count += 1
                continue

            tasks.append((query_key, query_task))

        print(f"[OntologyFilter] Fast query tasks: {len(tasks)}")
        print(
            "[OntologyFilter] In fast_query mode, each task is one LLM call "
            "for one query."
        )

        if not tasks:
            return

        self._parallel_judge_fast_queries(tasks)

    def _build_fast_query_task(
        self,
        evidence: Dict[str, Any],
        evidence_index: int,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        query_relation_id = int(evidence["query_relation_id"])
        query_relation_label = evidence.get(
            "query_relation_label",
            self._relation_label(query_relation_id),
        )

        range_classes = self.get_relation_range(query_relation_id)

        if not range_classes:
            return "", None

        entity_classes = self.collect_query_entity_classes(evidence)

        if not entity_classes:
            return "", None

        query_key = self._fast_query_cache_key(
            evidence_index=evidence_index,
            query_relation_id=query_relation_id,
            range_classes=range_classes,
            entity_classes=entity_classes,
        )

        query_task = {
            "query_relation_id": query_relation_id,
            "query_relation_label": query_relation_label,
            "range_classes": range_classes,
            "entity_classes": entity_classes,
        }

        return query_key, query_task

    def collect_query_entity_classes(
        self,
        evidence: Dict[str, Any],
    ) -> List[str]:
        classes = []

        for item in evidence.get("one_hop", []):
            tail_id = int(item["t_id"])
            classes.extend(self.get_entity_classes(tail_id))

        for path_item in evidence.get("paths", []):
            for step in path_item.get("path", []):
                tail_id = int(step["t_id"])
                classes.extend(self.get_entity_classes(tail_id))

        return self._deduplicate_clean(classes)

    def _parallel_judge_fast_queries(
        self,
        tasks: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        max_workers = max(1, int(self.parallel_workers))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._judge_fast_query_with_llm, task): (key, task)
                for key, task in tasks
            }

            done_count = 0

            for future in as_completed(future_to_task):
                key, task = future_to_task[future]

                try:
                    result = future.result()
                except Exception as e:
                    if self.continue_on_llm_error:
                        result = {
                            "mode": "fast_query",
                            "judgments": {},
                            "relation": "llm_error_fallback",
                            "score": self.fallback_score,
                            "reason": f"LLM error: {str(e)}",
                        }
                    else:
                        raise

                self._write_raw_cache(key, result)

                done_count += 1

                if self.verbose or done_count % 10 == 0:
                    print(
                        f"[OntologyFilter] Fast-query LLM progress: "
                        f"{done_count}/{len(tasks)} | "
                        f"relation={task['query_relation_label']} | "
                        f"classes={len(task['entity_classes'])}"
                    )

        self.save_cache()

    def _judge_fast_query_with_llm(
        self,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.llm_call_count += 1

        messages = build_fast_query_prompt(
            query_relation_label=task["query_relation_label"],
            range_classes=task["range_classes"],
            entity_classes=task["entity_classes"],
        )

        raw_output = self.llm.infer_raw(messages)
        parsed = safe_json_loads(raw_output, default=None)

        if not isinstance(parsed, dict):
            return {
                "mode": "fast_query",
                "judgments": {},
                "relation": "llm_error_fallback",
                "score": self.fallback_score,
                "reason": f"Failed to parse LLM output: {raw_output}",
            }

        judgments = parsed.get("judgments", [])

        if not isinstance(judgments, list):
            judgments = []

        result_map = {}

        for item in judgments:
            if not isinstance(item, dict):
                continue

            entity_class = self._normalize_class_text(
                item.get("entity_class", "")
            )
            relation = self._normalize_relation_label(
                item.get("relation", "")
            )
            reason = str(item.get("reason", ""))

            if not entity_class:
                continue

            result_map[entity_class] = {
                "relation": relation,
                "score": self.RELATION_TO_SCORE.get(relation, 0.5),
                "reason": reason,
            }

        return {
            "mode": "fast_query",
            "judgments": result_map,
            "range_classes": task["range_classes"],
        }

    # ========================================================
    # Filtering: precise/no_llm
    # ========================================================

    def _filter_evidence_dict_precise(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_relation_id = int(evidence["query_relation_id"])

        filtered_one_hop = []

        for item in evidence.get("one_hop", []):
            tail_id = int(item["t_id"])
            raw_score = float(item["score"])

            compatibility = self.entity_relation_range_compatibility_precise(
                entity_id=tail_id,
                relation_id=query_relation_id,
            )

            filtered_item = dict(item)
            filtered_item["ontology_eta"] = compatibility.score
            filtered_item["ontology_relation"] = compatibility.relation
            filtered_item["ontology_reason"] = compatibility.reason
            filtered_item["filtered_score"] = raw_score * compatibility.score

            filtered_one_hop.append(filtered_item)

        filtered_one_hop.sort(
            key=lambda x: x["filtered_score"],
            reverse=True,
        )
        filtered_one_hop = filtered_one_hop[: self.top_k_one_hop]

        filtered_paths = []

        for item in evidence.get("paths", []):
            raw_score = float(item["score"])
            eta_product = 1.0
            step_results = []

            for step in item.get("path", []):
                tail_id = int(step["t_id"])
                relation_id = int(step["r_id"])

                compatibility = self.entity_relation_range_compatibility_precise(
                    entity_id=tail_id,
                    relation_id=relation_id,
                )

                eta_product *= compatibility.score

                step_result = dict(step)
                step_result["entity_classes"] = self.get_entity_classes(tail_id)
                step_result["relation_range"] = self.get_relation_range(relation_id)
                step_result["ontology_eta"] = compatibility.score
                step_result["ontology_relation"] = compatibility.relation
                step_result["ontology_reason"] = compatibility.reason

                step_results.append(step_result)

            filtered_item = dict(item)
            filtered_item["ontology_eta_product"] = eta_product
            filtered_item["ontology_step_results"] = step_results
            filtered_item["filtered_score"] = raw_score * eta_product

            filtered_paths.append(filtered_item)

        filtered_paths.sort(
            key=lambda x: x["filtered_score"],
            reverse=True,
        )
        filtered_paths = filtered_paths[: self.top_k_paths]

        result = dict(evidence)
        result["filter_mode"] = self.mode
        result["filtered_one_hop"] = filtered_one_hop
        result["filtered_paths"] = filtered_paths

        return result

    def entity_relation_range_compatibility_precise(
        self,
        entity_id: int,
        relation_id: int,
    ) -> CompatibilityResult:
        entity_classes = self.get_entity_classes(entity_id)
        relation_range = self.get_relation_range(relation_id)

        if not relation_range:
            return CompatibilityResult(
                relation="no_range_constraint",
                score=self.missing_range_score,
                reason="The relation has no available range constraint.",
            )

        if not entity_classes:
            return CompatibilityResult(
                relation="missing_entity_class",
                score=self.missing_entity_class_score,
                reason="The entity has no available class information.",
            )

        best = CompatibilityResult(
            relation="disjoint",
            score=0.5,
            reason="Default disjoint.",
        )

        for ec in entity_classes:
            for rc in relation_range:
                current = self.class_pair_compatibility_from_cache(ec, rc)

                if current.score > best.score:
                    best = current

                if best.score >= 1.0:
                    return best

        return best

    def class_pair_compatibility_from_cache(
        self,
        entity_class: str,
        range_class: str,
    ) -> CompatibilityResult:
        entity_class = self._normalize_class_text(entity_class)
        range_class = self._normalize_class_text(range_class)

        if not entity_class or not range_class:
            return CompatibilityResult(
                relation="missing_class_text",
                score=self.missing_entity_class_score,
                reason="Empty class text.",
            )

        if self._is_direct_match(entity_class, range_class):
            return CompatibilityResult(
                relation="subclass_or_same",
                score=self.direct_match_score,
                reason="Exact class match.",
            )

        key = self._pair_cache_key(entity_class, range_class)
        cached = self.cache.get(key)

        if cached is None:
            # This should be rare because precise mode precomputes the cache.
            return CompatibilityResult(
                relation="no_llm_fallback",
                score=self.fallback_score,
                reason="Pair not found in cache; fallback score.",
            )

        return CompatibilityResult(
            relation=cached.get("relation", "disjoint"),
            score=float(cached.get("score", 0.5)),
            reason=cached.get("reason", "Loaded from cache."),
        )

    # ========================================================
    # Filtering: fast_query mode
    # ========================================================

    def _filter_evidence_dict_fast_query(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_relation_id = int(evidence["query_relation_id"])
        range_classes = self.get_relation_range(query_relation_id)
        entity_classes = self.collect_query_entity_classes(evidence)

        query_key = self._fast_query_cache_key(
            evidence_index=None,
            query_relation_id=query_relation_id,
            range_classes=range_classes,
            entity_classes=entity_classes,
        )

        # Because precompute uses evidence index for uniqueness by default,
        # this fallback searches any compatible key.
        fast_cache = self._find_fast_query_cache(
            query_relation_id=query_relation_id,
            range_classes=range_classes,
            entity_classes=entity_classes,
        )

        filtered_one_hop = []

        for item in evidence.get("one_hop", []):
            tail_id = int(item["t_id"])
            raw_score = float(item["score"])

            compatibility = self.entity_compatibility_fast_query(
                entity_id=tail_id,
                fast_cache=fast_cache,
            )

            filtered_item = dict(item)
            filtered_item["ontology_eta"] = compatibility.score
            filtered_item["ontology_relation"] = compatibility.relation
            filtered_item["ontology_reason"] = compatibility.reason
            filtered_item["filtered_score"] = raw_score * compatibility.score

            filtered_one_hop.append(filtered_item)

        filtered_one_hop.sort(
            key=lambda x: x["filtered_score"],
            reverse=True,
        )
        filtered_one_hop = filtered_one_hop[: self.top_k_one_hop]

        filtered_paths = []

        for item in evidence.get("paths", []):
            raw_score = float(item["score"])
            eta_product = 1.0
            step_results = []

            for step in item.get("path", []):
                tail_id = int(step["t_id"])

                compatibility = self.entity_compatibility_fast_query(
                    entity_id=tail_id,
                    fast_cache=fast_cache,
                )

                eta_product *= compatibility.score

                step_result = dict(step)
                step_result["entity_classes"] = self.get_entity_classes(tail_id)
                step_result["fast_query_range"] = range_classes
                step_result["ontology_eta"] = compatibility.score
                step_result["ontology_relation"] = compatibility.relation
                step_result["ontology_reason"] = compatibility.reason

                step_results.append(step_result)

            filtered_item = dict(item)
            filtered_item["ontology_eta_product"] = eta_product
            filtered_item["ontology_step_results"] = step_results
            filtered_item["filtered_score"] = raw_score * eta_product

            filtered_paths.append(filtered_item)

        filtered_paths.sort(
            key=lambda x: x["filtered_score"],
            reverse=True,
        )
        filtered_paths = filtered_paths[: self.top_k_paths]

        result = dict(evidence)
        result["filter_mode"] = "fast_query"
        result["fast_query_note"] = (
            "All evidence classes are judged against the query relation range. "
            "Path-step relation-specific ranges are ignored for speed."
        )
        result["filtered_one_hop"] = filtered_one_hop
        result["filtered_paths"] = filtered_paths

        return result

    def entity_compatibility_fast_query(
        self,
        entity_id: int,
        fast_cache: Optional[Dict[str, Any]],
    ) -> CompatibilityResult:
        entity_classes = self.get_entity_classes(entity_id)

        if not entity_classes:
            return CompatibilityResult(
                relation="missing_entity_class",
                score=self.missing_entity_class_score,
                reason="The entity has no available class information.",
            )

        if fast_cache is None:
            return CompatibilityResult(
                relation="no_llm_fallback",
                score=self.fallback_score,
                reason="No fast-query cache found; fallback score.",
            )

        judgments = fast_cache.get("judgments", {})

        best = CompatibilityResult(
            relation="disjoint",
            score=0.5,
            reason="Default disjoint in fast-query mode.",
        )

        for ec in entity_classes:
            ec_norm = self._normalize_class_text(ec)
            item = judgments.get(ec_norm)

            if item is None:
                current = CompatibilityResult(
                    relation="no_llm_fallback",
                    score=self.fallback_score,
                    reason="Class not returned by fast-query LLM; fallback score.",
                )
            else:
                current = CompatibilityResult(
                    relation=item.get("relation", "disjoint"),
                    score=float(item.get("score", 0.5)),
                    reason=item.get("reason", "Loaded from fast-query cache."),
                )

            if current.score > best.score:
                best = current

            if best.score >= 1.0:
                return best

        return best

    def _find_fast_query_cache(
        self,
        query_relation_id: int,
        range_classes: List[str],
        entity_classes: List[str],
    ) -> Optional[Dict[str, Any]]:
        # Search by deterministic suffix because precompute may include evidence index.
        signature = self._fast_query_signature(
            query_relation_id=query_relation_id,
            range_classes=range_classes,
            entity_classes=entity_classes,
        )

        for key, value in self.cache.items():
            if key.startswith("fast_query::") and key.endswith(signature):
                return value

        return None

    # ========================================================
    # Ontology access helpers
    # ========================================================

    def get_entity_classes(self, entity_id: int) -> List[str]:

        ent = self.entities.get(entity_id)

        if ent is None:
            return []

        natural_classes = []

        # 1. First use the normalized field loaded by KGLoader
        if getattr(ent, "classname", None):
            natural_classes.extend(self._to_list(ent.classname))

        raw = getattr(ent, "raw", None) or {}

        # 2. Then use raw natural-language class fields
        natural_class_keys = [
            "classlabel",
            "class_label",
            "classname",
            "class_name",
            "classes",
            "types",
            "type",
        ]

        for key in natural_class_keys:
            if key in raw:
                natural_classes.extend(self._to_list(raw[key]))

        natural_classes = self._deduplicate_clean(natural_classes)

        # 3. Remove numeric values from natural classes
        natural_classes = [
            c for c in natural_classes
            if not self._is_numeric_like(c)
        ]

        # 4. If natural-language class labels exist, return them directly
        if natural_classes:
            return natural_classes

        # 5. Only when no natural class label exists, fallback to classid
        fallback_classes = []

        if getattr(ent, "class_id", None):
            fallback_classes.extend(self._to_list(ent.class_id))

        numeric_class_keys = [
            "classid",
            "class_id",
            "class",
        ]

        for key in numeric_class_keys:
            if key in raw:
                fallback_classes.extend(self._to_list(raw[key]))

        fallback_classes = self._deduplicate_clean(fallback_classes)

        # 6. Still avoid sending pure numbers to LLM if possible
        fallback_classes = [
            c for c in fallback_classes
            if not self._is_numeric_like(c)
        ]

        return fallback_classes

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

        return self._deduplicate_clean(ranges)

    def get_relation_domain(self, relation_id: int) -> List[str]:
        rel = self.relations.get(relation_id)

        if rel is None:
            return []

        domains = []

        if getattr(rel, "domain", None):
            domains.extend(self._to_list(rel.domain))

        raw = getattr(rel, "raw", None) or {}

        for key in [
            "domain",
            "domains",
            "head_type",
            "head_class",
            "subject_type",
        ]:
            if key in raw:
                domains.extend(self._to_list(raw[key]))

        return self._deduplicate_clean(domains)

    def _entity_label(self, entity_id: int) -> str:
        ent = self.entities.get(entity_id)
        if ent is None:
            return f"[UnknownEntity:{entity_id}]"
        return ent.label or str(entity_id)

    def _relation_label(self, relation_id: int) -> str:
        rel = self.relations.get(relation_id)
        if rel is None:
            return f"[UnknownRelation:{relation_id}]"
        return rel.label or str(relation_id)

    # ========================================================
    # Cache helpers
    # ========================================================

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

    def _write_cache(
        self,
        key: str,
        result: CompatibilityResult,
    ) -> None:
        with self.cache_lock:
            self.cache[key] = {
                "relation": result.relation,
                "score": result.score,
                "reason": result.reason,
            }

    def _write_raw_cache(
        self,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        with self.cache_lock:
            self.cache[key] = value

    @staticmethod
    def _pair_cache_key(entity_class: str, range_class: str) -> str:
        return f"pair::{entity_class}|||{range_class}"

    def _fast_query_cache_key(
        self,
        evidence_index: Optional[int],
        query_relation_id: int,
        range_classes: List[str],
        entity_classes: List[str],
    ) -> str:
        signature = self._fast_query_signature(
            query_relation_id=query_relation_id,
            range_classes=range_classes,
            entity_classes=entity_classes,
        )

        if evidence_index is None:
            return f"fast_query::{signature}"

        return f"fast_query::{evidence_index}::{signature}"

    @staticmethod
    def _fast_query_signature(
        query_relation_id: int,
        range_classes: List[str],
        entity_classes: List[str],
    ) -> str:
        range_part = "||".join(sorted(range_classes))
        class_part = "||".join(sorted(entity_classes))
        return f"r={query_relation_id}::range={range_part}::classes={class_part}"

    # ========================================================
    # General helpers
    # ========================================================

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
    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(v) for v in value if v is not None]

        if isinstance(value, tuple) or isinstance(value, set):
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

    @staticmethod
    def _normalize_class_text(text: Any) -> str:
        text = str(text).strip()
        text = text.replace("_", " ")
        text = text.replace("/", " / ")
        text = " ".join(text.split())
        return text.lower()

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

    def _is_direct_match(self, a: str, b: str) -> bool:
        return self._normalize_class_text(a) == self._normalize_class_text(b)

    def _print_statistics(self) -> None:
        print("\n[OntologyFilter] Statistics")
        print(f"LLM calls: {self.llm_call_count}")
        print(f"Cache hits: {self.cache_hit_count}")
        print(f"Direct matches: {self.direct_match_count}")
        print(f"Fallbacks: {self.fallback_count}")
        print(f"Cache size: {len(self.cache)}")


# ============================================================
# IO and inspection
# ============================================================

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
            f"{idx + 1}. filtered={item['filtered_score']:.4f} | "
            f"raw={item['score']:.4f} | "
            f"eta={item['ontology_eta']:.2f} | "
            f"{item['ontology_relation']} | "
            f"{item.get('text')}"
        )

    print("\n[Filtered path evidence]")
    for idx, item in enumerate(evidence.get("filtered_paths", [])[:10]):
        print(
            f"{idx + 1}. filtered={item['filtered_score']:.4f} | "
            f"raw={item['score']:.4f} | "
            f"eta_path={item.get('ontology_eta_product', 1.0):.4f} | "
            f"{item.get('text')}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ontology-constrained filtering for OD-KGC evidence."
    )

    parser.add_argument("--data_path", type=str, default="data/FB15k-237")
    parser.add_argument("--dataset_name", type=str, default=None)

    parser.add_argument("--input_evidence_path", type=str, default=None)
    parser.add_argument("--output_evidence_path", type=str, default=None)
    parser.add_argument("--cache_path", type=str, default=None)

    parser.add_argument(
        "--filter_mode",
        type=str,
        default="precise",
        choices=["precise", "fast_query", "no_llm"],
        help=(
            "precise: cache + parallel class-pair LLM judgments; "
            "fast_query: one LLM call per query; "
            "no_llm: no LLM calls."
        ),
    )

    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=16,
        help="Number of parallel LLM requests.",
    )

    parser.add_argument("--llm_model", type=str, default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:22014/v1")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--top_k_one_hop", type=int, default=10)
    parser.add_argument("--top_k_paths", type=int, default=10)

    parser.add_argument("--missing_range_score", type=float, default=1.0)
    parser.add_argument("--missing_entity_class_score", type=float, default=0.8)
    parser.add_argument("--fallback_score", type=float, default=0.8)

    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="Only process the first N evidence items. Use -1 for all.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index of evidence items.",
    )

    parser.add_argument(
        "--strict_llm_error",
        action="store_true",
        default=False,
        help="If set, raise error when LLM call fails.",
    )

    parser.add_argument("--verbose", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name

    if args.input_evidence_path is None:
        input_evidence_path = (
            PROJECT_ROOT
            / "import"
            / "evidence"
            / dataset_name
            / "test_evidence.jsonl"
        )
    else:
        input_evidence_path = Path(args.input_evidence_path)

    if args.output_evidence_path is None:
        output_evidence_path = (
            PROJECT_ROOT
            / "import"
            / "filtered_evidence"
            / dataset_name
            / f"test_filtered_evidence_{args.filter_mode}.jsonl"
        )
    else:
        output_evidence_path = Path(args.output_evidence_path)

    if args.cache_path is None:
        cache_path = (
            PROJECT_ROOT
            / "import"
            / "ontology_cache"
            / f"{dataset_name}_{args.filter_mode}_ontology_cache.json"
        )
    else:
        cache_path = Path(args.cache_path)

    print("[OntologyFilter] Loading dataset...")
    loader = KGLoader(data_path)
    dataset = loader.load()

    print(f"[OntologyFilter] Loading evidence from {input_evidence_path}")
    evidence_list = load_jsonl(input_evidence_path)

    print(f"[OntologyFilter] Original evidence items: {len(evidence_list)}")

    if args.start_index > 0:
        evidence_list = evidence_list[args.start_index:]

    if args.max_items is not None and args.max_items > 0:
        evidence_list = evidence_list[: args.max_items]

    print(f"[OntologyFilter] Evidence items to process: {len(evidence_list)}")

    if args.filter_mode == "no_llm":
        llm = None
        print("[OntologyFilter] no_llm mode: LLM disabled.")
    else:
        llm = LLM_Model(
            llm_model=args.llm_model,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )

    ontology_filter = OntologyFilter(
        dataset=dataset,
        llm=llm,
        cache_path=cache_path,
        mode=args.filter_mode,
        parallel_workers=args.parallel_workers,
        missing_range_score=args.missing_range_score,
        missing_entity_class_score=args.missing_entity_class_score,
        fallback_score=args.fallback_score,
        top_k_one_hop=args.top_k_one_hop,
        top_k_paths=args.top_k_paths,
        continue_on_llm_error=not args.strict_llm_error,
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