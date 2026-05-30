# OD-KGC/model/evidence_compressor.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader
from src.utils import LLM_Model, load_jsonl, save_jsonl

from model.KGE_model import get_or_train_rotate
from model.evidence_extractor import EvidenceExtractor, save_evidence_jsonl
from model.ontology_filter import OntologyFilter


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_numeric_like(value: Any) -> bool:
    if value is None:
        return False

    value = str(value).strip()

    if value == "":
        return False

    return value.isdigit()


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
        return [str(v) for v in value if v is not None and str(v).strip() != ""]

    if isinstance(value, tuple) or isinstance(value, set):
        return [str(v) for v in value if v is not None and str(v).strip() != ""]

    if isinstance(value, dict):
        results = []

        for k, v in value.items():
            if isinstance(v, (list, tuple, set)):
                results.extend([str(x) for x in v if x is not None and str(x).strip() != ""])
            elif v is not None and str(v).strip() != "":
                results.append(str(v))
            elif k is not None and str(k).strip() != "":
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


def approx_token_len(text: str) -> int:
    """
    简单估计 token 数。
    英文 KG 文本里用 whitespace 数量近似即可。
    """

    if not text:
        return 0

    return len(str(text).split())


def safe_get_score(item: Dict[str, Any]) -> float:
    if "filtered_score" in item:
        return float(item["filtered_score"])

    if "score" in item:
        return float(item["score"])

    return 0.0


# ============================================================
# Dataset schema helpers
# ============================================================

class DatasetSchemaHelper:
    """
    从 kg_loader.py 加载后的 dataset 中提取 label / class / domain / range。

    这里使用更安全的 class 提取策略：
        - 优先 classlabel / classname；
        - 有自然语言 class 时不使用数字 classid；
        - 避免把 52、789 这类 classid 发给后续 LLM。
    """

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

        natural_classes = []

        # 1. kg_loader normalized field
        if getattr(ent, "classname", None):
            natural_classes.extend(to_list(ent.classname))

        raw = getattr(ent, "raw", None) or {}

        # 2. raw natural language fields
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
                natural_classes.extend(to_list(raw[key]))

        natural_classes = deduplicate_clean(natural_classes)

        # 3. remove numeric values
        natural_classes = [
            c for c in natural_classes
            if not is_numeric_like(c)
        ]

        if natural_classes:
            return natural_classes

        # 4. fallback to class id only when no class label exists
        fallback_classes = []

        if getattr(ent, "class_id", None):
            fallback_classes.extend(to_list(ent.class_id))

        for key in ["classid", "class_id", "class"]:
            if key in raw:
                fallback_classes.extend(to_list(raw[key]))

        fallback_classes = deduplicate_clean(fallback_classes)

        # For compression cache, avoid numeric class ids if possible.
        fallback_classes = [
            c for c in fallback_classes
            if not is_numeric_like(c)
        ]

        return fallback_classes

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


# ============================================================
# Evidence Compressor
# ============================================================

class EvidenceCompressor:
    """
    Compression after ontology filtering.

    Input:
        filtered evidence dict

    Output:
        compact cache record for LLM inference.

    Compression strategy:
        1. Convert filtered_one_hop and filtered_paths into candidates.
        2. Sort by filtered_score.
        3. Greedy select under max_evidence_num and token_budget.
        4. Reduce redundancy by:
            - duplicate text;
            - duplicate terminal entity;
            - duplicate relation path pattern.
    """

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

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def compress_evidence_list(
        self,
        filtered_evidence_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records = []

        for idx, evidence in enumerate(filtered_evidence_list):
            record = self.compress_one_evidence(evidence, query_index=idx)
            records.append(record)

            if (idx + 1) % 100 == 0:
                print(f"[EvidenceCompressor] Compressed {idx + 1}/{len(filtered_evidence_list)} items.")

        return records

    def compress_one_evidence(
        self,
        evidence: Dict[str, Any],
        query_index: Optional[int] = None,
    ) -> Dict[str, Any]:

        query_head_id = int(evidence["query_head_id"])
        query_relation_id = int(evidence["query_relation_id"])
        gold_tail_id = evidence.get("gold_tail_id", None)

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

        record = {
            "dataset": self.dataset.dataset_name,
            "query_index": query_index,

            "query_triple_id": {
                "head_id": query_head_id,
                "relation_id": query_relation_id,
                "tail_id": gold_tail_id,
            },

            "query_triple_name": {
                "head": query_head_label,
                "relation": query_relation_label,
                "tail": gold_tail_label,
            },

            "query": {
                "head_id": query_head_id,
                "head_label": query_head_label,
                "head_classes": self.schema.entity_classes(query_head_id),

                "relation_id": query_relation_id,
                "relation_label": query_relation_label,
                "relation_domain": self.schema.relation_domain(query_relation_id),
                "relation_range": self.schema.relation_range(query_relation_id),

                "tail_id": gold_tail_id,
                "tail_label": gold_tail_label,
                "tail_classes": self.schema.entity_classes(gold_tail_id),
            },

            "correct_answer": {
                "id": gold_tail_id,
                "label": gold_tail_label,
                "classes": self.schema.entity_classes(gold_tail_id),
            },

            "compression_config": {
                "max_evidence_num": self.max_evidence_num,
                "token_budget": self.token_budget,
                "min_score": self.min_score,
                "keep_one_hop": self.keep_one_hop,
                "keep_paths": self.keep_paths,
                "remove_duplicate_text": self.remove_duplicate_text,
                "remove_duplicate_terminal": self.remove_duplicate_terminal,
                "remove_duplicate_relation_pattern": self.remove_duplicate_relation_pattern,
            },

            "key_evidence": selected,
            "evidence_text": evidence_text,

            "source_statistics": {
                "num_one_hop_source": len(evidence.get("filtered_one_hop", evidence.get("one_hop", []))),
                "num_path_source": len(evidence.get("filtered_paths", evidence.get("paths", []))),
                "num_candidate_evidence": len(candidates),
                "num_selected_evidence": len(selected),
                "estimated_tokens": approx_token_len(evidence_text),
            },
        }

        return record

    # --------------------------------------------------------
    # Candidate construction
    # --------------------------------------------------------

    def build_candidate_evidence(
        self,
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates = []

        if self.keep_one_hop:
            one_hop_items = evidence.get("filtered_one_hop", None)

            if one_hop_items is None:
                one_hop_items = evidence.get("one_hop", [])

            for item in one_hop_items:
                candidate = self.convert_one_hop_item(item)
                candidates.append(candidate)

        if self.keep_paths:
            path_items = evidence.get("filtered_paths", None)

            if path_items is None:
                path_items = evidence.get("paths", [])

            for item in path_items:
                candidate = self.convert_path_item(item)
                candidates.append(candidate)

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

        candidate = {
            "type": "one_hop",
            "text": text,
            "score": score,
            "raw_score": raw_score,
            "ontology_eta": eta,
            "ontology_relation": item.get("ontology_relation"),
            "ontology_reason": item.get("ontology_reason"),

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

            "head_classes": self.schema.entity_classes(h_id),
            "relation_domain": self.schema.relation_domain(r_id),
            "relation_range": self.schema.relation_range(r_id),
            "tail_classes": self.schema.entity_classes(t_id),

            "terminal_entity_id": t_id,
            "terminal_entity_label": t_label,
            "relation_pattern": [r_label],
            "relation_pattern_ids": [r_id],
        }

        return candidate

    def convert_path_item(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = safe_get_score(item)
        raw_score = float(item.get("score", score))
        eta_product = float(item.get("ontology_eta_product", 1.0))

        path_steps = item.get("path", [])
        step_results = item.get("ontology_step_results", [])

        terminal_entity_id = item.get("terminal_entity_id", None)

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
                "head_classes": self.schema.entity_classes(h_id),
                "relation_domain": self.schema.relation_domain(r_id),
                "relation_range": self.schema.relation_range(r_id),
                "tail_classes": self.schema.entity_classes(t_id),
            }

            if idx < len(step_results):
                step_info["ontology_eta"] = step_results[idx].get("ontology_eta")
                step_info["ontology_relation"] = step_results[idx].get("ontology_relation")
                step_info["ontology_reason"] = step_results[idx].get("ontology_reason")

            converted_steps.append(step_info)

        candidate = {
            "type": "path",
            "text": text,
            "score": score,
            "raw_score": raw_score,
            "ontology_eta_product": eta_product,

            "path_length": len(path_steps),
            "path": converted_steps,

            "terminal_entity_id": terminal_entity_id,
            "terminal_entity_label": terminal_entity_label,
            "terminal_entity_classes": self.schema.entity_classes(terminal_entity_id),

            "relation_pattern": relation_pattern,
            "relation_pattern_ids": relation_pattern_ids,
        }

        return candidate

    # --------------------------------------------------------
    # Greedy selection
    # --------------------------------------------------------

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
            score = item.get("score", 0.0)
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


# ============================================================
# Pipeline functions
# ============================================================

def load_or_build_structural_evidence(
    dataset,
    data_path: Path,
    dataset_name: str,
    import_root: Path,
    split: str,
    use_cache: bool,
    max_queries: Optional[int],
    top_k_one_hop: int,
    top_k_paths: int,
    max_hops: int,
    max_branch_per_node: int,
    cuda: bool,
    gpu_id: int,
) -> List[Dict[str, Any]]:
    evidence_path = (
        import_root
        / "evidence"
        / dataset_name
        / f"{split}_evidence.jsonl"
    )

    if use_cache and evidence_path.exists():
        print(f"[EvidenceCompressor] Load structural evidence cache: {evidence_path}")
        return load_jsonl(evidence_path)

    print("[EvidenceCompressor] Structural evidence cache not used or not found.")
    print("[EvidenceCompressor] Building structural evidence...")

    rotate = get_or_train_rotate(
        data_path=str(data_path),
        import_path=str(import_root / "KGE_model"),
        dataset_name=dataset_name,
        load_if_exists=True,
        force_train=False,
        cuda=cuda,
        gpu_id=gpu_id,
    )

    extractor = EvidenceExtractor(
        dataset=dataset,
        rotate_manager=rotate,
        use_train=True,
        use_valid=True,
        use_test=False,
        mask_answer_entity=True,
        normalize_scores=True,
    )

    evidence_objects = extractor.extract_for_split(
        split=split,
        max_queries=max_queries,
        top_k_one_hop=top_k_one_hop,
        top_k_paths=top_k_paths,
        max_hops=max_hops,
        max_branch_per_node=max_branch_per_node,
    )

    save_evidence_jsonl(evidence_objects, evidence_path)

    return load_jsonl(evidence_path)


def load_or_build_filtered_evidence(
    dataset,
    dataset_name: str,
    import_root: Path,
    split: str,
    structural_evidence: List[Dict[str, Any]],
    use_cache: bool,
    filter_mode: str,
    llm_model: str,
    openai_api_key: str,
    openai_base_url: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    parallel_workers: int,
    top_k_one_hop: int,
    top_k_paths: int,
    fallback_score: float,
) -> List[Dict[str, Any]]:
    filtered_path = (
        import_root
        / "evidence"
        / dataset_name
        / f"{split}_filtered_evidence_{filter_mode}.jsonl"
    )

    cache_path = (
        import_root
        / "ontology_cache"
        / f"{dataset_name}_{filter_mode}_ontology_cache.json"
    )

    if use_cache and filtered_path.exists():
        print(f"[EvidenceCompressor] Load filtered evidence cache: {filtered_path}")
        return load_jsonl(filtered_path)

    print("[EvidenceCompressor] Filtered evidence cache not used or not found.")
    print("[EvidenceCompressor] Building ontology-filtered evidence...")

    if filter_mode == "no_llm":
        llm = None
    else:
        llm = LLM_Model(
            llm_model=llm_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    ontology_filter = OntologyFilter(
        dataset=dataset,
        llm=llm,
        cache_path=cache_path,
        mode=filter_mode,
        parallel_workers=parallel_workers,
        top_k_one_hop=top_k_one_hop,
        top_k_paths=top_k_paths,
        fallback_score=fallback_score,
        continue_on_llm_error=True,
        verbose=False,
    )

    filtered = ontology_filter.filter_evidence_list(structural_evidence)
    save_jsonl(filtered, filtered_path)

    ontology_filter.save_cache()

    if llm is not None:
        llm.close()

    return filtered


def load_or_build_compressed_evidence(
    dataset,
    dataset_name: str,
    import_root: Path,
    split: str,
    filtered_evidence: List[Dict[str, Any]],
    use_cache: bool,
    compression_suffix: str,
    max_evidence_num: int,
    token_budget: int,
    min_score: Optional[float],
    keep_one_hop: bool,
    keep_paths: bool,
    include_score_in_text: bool,
) -> List[Dict[str, Any]]:
    compressed_path = (
        import_root
        / "evidence"
        / dataset_name
        / f"{split}_compressed_evidence{compression_suffix}.jsonl"
    )

    if use_cache and compressed_path.exists():
        print(f"[EvidenceCompressor] Load compressed evidence cache: {compressed_path}")
        return load_jsonl(compressed_path)

    print("[EvidenceCompressor] Building compressed key evidence...")

    compressor = EvidenceCompressor(
        dataset=dataset,
        max_evidence_num=max_evidence_num,
        token_budget=token_budget,
        min_score=min_score,
        keep_one_hop=keep_one_hop,
        keep_paths=keep_paths,
        remove_duplicate_text=True,
        remove_duplicate_terminal=True,
        remove_duplicate_relation_pattern=True,
        include_score_in_text=include_score_in_text,
    )

    compressed = compressor.compress_evidence_list(filtered_evidence)
    save_jsonl(compressed, compressed_path)

    print(f"[EvidenceCompressor] Saved compressed evidence to: {compressed_path}")

    return compressed


# ============================================================
# Inspection
# ============================================================

def inspect_compressed_record(record: Dict[str, Any]) -> None:
    print("=" * 100)
    print("[Compressed Evidence Example]")
    print("=" * 100)

    query = record["query"]
    answer = record["correct_answer"]

    print(f"Dataset: {record.get('dataset')}")
    print(
        f"Query ID: "
        f"({query['head_id']}, {query['relation_id']}, {query['tail_id']})"
    )
    print(
        f"Query Name: "
        f"({query['head_label']}, {query['relation_label']}, ?)"
    )
    print(f"Head classes: {query.get('head_classes')}")
    print(f"Relation domain: {query.get('relation_domain')}")
    print(f"Relation range: {query.get('relation_range')}")
    print(f"Correct answer: {answer.get('label')} | classes={answer.get('classes')}")

    print("\n[Key Evidence]")
    for idx, item in enumerate(record.get("key_evidence", []), start=1):
        print(
            f"{idx}. type={item.get('type')} | "
            f"score={item.get('score'):.4f} | "
            f"{item.get('text')}"
        )

    print("\n[Evidence Text]")
    print(record.get("evidence_text", ""))

    print("\n[Statistics]")
    print(json.dumps(record.get("source_statistics", {}), ensure_ascii=False, indent=2))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evidence compression and cache generation for OD-KGC."
    )

    # Dataset
    parser.add_argument("--data_path", type=str, default="data/FB15k-237")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--import_path", type=str, default="import")

    # Cache control
    parser.add_argument(
        "--use_evidence_cache",
        action="store_true",
        default=True,
        help="Use structural evidence cache if exists.",
    )
    parser.add_argument(
        "--regenerate_evidence",
        dest="use_evidence_cache",
        action="store_false",
        help="Regenerate structural evidence even if cache exists.",
    )

    parser.add_argument(
        "--use_filtered_cache",
        action="store_true",
        default=True,
        help="Use filtered evidence cache if exists.",
    )
    parser.add_argument(
        "--regenerate_filtered",
        dest="use_filtered_cache",
        action="store_false",
        help="Regenerate filtered evidence even if cache exists.",
    )

    parser.add_argument(
        "--use_compressed_cache",
        action="store_true",
        default=False,
        help="Use compressed evidence cache if exists.",
    )

    # Evidence extraction config
    parser.add_argument("--max_queries", type=int, default=5)
    parser.add_argument("--top_k_one_hop", type=int, default=10)
    parser.add_argument("--top_k_paths", type=int, default=10)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_branch_per_node", type=int, default=20)

    # Ontology filter config
    parser.add_argument(
        "--filter_mode",
        type=str,
        default="fast_query",
        choices=["precise", "fast_query", "no_llm"],
        help="Use fast_query for quick cache generation; precise for final experiments.",
    )
    parser.add_argument("--parallel_workers", type=int, default=16)
    parser.add_argument("--fallback_score", type=float, default=0.8)

    # LLM config
    parser.add_argument("--llm_model", type=str, default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:22014/v1")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)

    # KGE / GPU config
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--gpu_id", type=int, default=0)

    # Compression config
    parser.add_argument("--max_evidence_num", type=int, default=5)
    parser.add_argument("--token_budget", type=int, default=800)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--compression_suffix", type=str, default="")
    parser.add_argument("--include_score_in_text", action="store_true", default=False)

    parser.add_argument(
        "--only_one_hop",
        action="store_true",
        default=False,
        help="Only keep one-hop evidence.",
    )
    parser.add_argument(
        "--only_paths",
        action="store_true",
        default=False,
        help="Only keep path evidence.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name
    import_root = Path(args.import_path)

    max_queries = None if args.max_queries == -1 else args.max_queries

    print("=" * 100)
    print("[OD-KGC Evidence Compression Pipeline]")
    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print(f"Split: {args.split}")
    print(f"Import root: {import_root}")
    print(f"Filter mode: {args.filter_mode}")
    print(f"Max queries: {max_queries}")
    print("=" * 100)

    print("[EvidenceCompressor] Loading dataset...")
    loader = KGLoader(data_path)
    dataset = loader.load()

    structural_evidence = load_or_build_structural_evidence(
        dataset=dataset,
        data_path=data_path,
        dataset_name=dataset_name,
        import_root=import_root,
        split=args.split,
        use_cache=args.use_evidence_cache,
        max_queries=max_queries,
        top_k_one_hop=args.top_k_one_hop,
        top_k_paths=args.top_k_paths,
        max_hops=args.max_hops,
        max_branch_per_node=args.max_branch_per_node,
        cuda=args.cuda,
        gpu_id=args.gpu_id,
    )

    filtered_evidence = load_or_build_filtered_evidence(
        dataset=dataset,
        dataset_name=dataset_name,
        import_root=import_root,
        split=args.split,
        structural_evidence=structural_evidence,
        use_cache=args.use_filtered_cache,
        filter_mode=args.filter_mode,
        llm_model=args.llm_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        parallel_workers=args.parallel_workers,
        top_k_one_hop=args.top_k_one_hop,
        top_k_paths=args.top_k_paths,
        fallback_score=args.fallback_score,
    )

    keep_one_hop = True
    keep_paths = True

    if args.only_one_hop:
        keep_paths = False

    if args.only_paths:
        keep_one_hop = False

    compressed = load_or_build_compressed_evidence(
        dataset=dataset,
        dataset_name=dataset_name,
        import_root=import_root,
        split=args.split,
        filtered_evidence=filtered_evidence,
        use_cache=args.use_compressed_cache,
        compression_suffix=args.compression_suffix,
        max_evidence_num=args.max_evidence_num,
        token_budget=args.token_budget,
        min_score=args.min_score,
        keep_one_hop=keep_one_hop,
        keep_paths=keep_paths,
        include_score_in_text=args.include_score_in_text,
    )

    if compressed:
        inspect_compressed_record(compressed[0])

    print("\n[EvidenceCompressor] Done.")


if __name__ == "__main__":
    main()