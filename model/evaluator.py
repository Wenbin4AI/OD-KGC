# OD-KGC/model/llm_rank_evaluator.py

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader
from src.utils import LLM_Model, build_messages, load_jsonl
from model.KGE_model import get_or_train_rotate
from model.evidence_compressor import DatasetSchemaHelper


# ============================================================
# Basic IO
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def set_random_seed(seed: int) -> None:
    random.seed(seed)


def class_list_to_text(classes: Any) -> str:
    if classes is None:
        return "Unknown"

    if isinstance(classes, list):
        values = [str(x).strip() for x in classes if str(x).strip()]
    else:
        values = [str(classes).strip()]

    return ", ".join(values) if values else "Unknown"


# ============================================================
# LLM output parsing
# ============================================================

def clean_llm_output(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Remove complete think blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract the first JSON object from text.

    Useful when Qwen outputs:
        <think>...</think>
        {"selected_indices": [...]}
    """

    if text is None:
        return None

    text = str(text)
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    return text[start:end + 1]


def extract_selected_indices(
    llm_output: str,
    k: int = 10,
    num_candidates: int = 20,
) -> Optional[List[int]]:
    """
    Extract ranked candidate indices from LLM output.

    Expected:
        {"selected_indices": [1, 3, 5, ...]}

    Also supports:
        {"ranked_indices": [...]}
        {"top10": [...]}
        {"indices": [...]}
    """

    if llm_output is None:
        return None

    raw_text = str(llm_output)
    cleaned_output = clean_llm_output(raw_text)

    possible_json_strings = [cleaned_output]

    extracted = extract_json_object(cleaned_output)
    if extracted is not None:
        possible_json_strings.append(extracted)

    extracted_raw = extract_json_object(raw_text)
    if extracted_raw is not None:
        possible_json_strings.append(extracted_raw)

    for json_text in possible_json_strings:
        try:
            data = json.loads(json_text)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        indices = (
            data.get("selected_indices")
            or data.get("ranked_indices")
            or data.get("top10")
            or data.get("indices")
        )

        if isinstance(indices, list):
            parsed = []

            for item in indices:
                try:
                    idx = int(item)
                except Exception:
                    continue

                if 0 <= idx < num_candidates and idx not in parsed:
                    parsed.append(idx)

                if len(parsed) >= k:
                    break

            if parsed:
                return parsed[:k]

    # Fallback: extract numbers, but only keep valid candidate indices.
    numbers = re.findall(r"\d+", cleaned_output)

    parsed = []
    for x in numbers:
        idx = int(x)

        if 0 <= idx < num_candidates and idx not in parsed:
            parsed.append(idx)

        if len(parsed) >= k:
            break

    if parsed:
        return parsed[:k]

    return None


def complete_topk_indices(
    pred_indices: Optional[List[int]],
    num_candidates: int = 20,
    k: int = 10,
    fill_by_default_order: bool = True,
) -> List[int]:
    """
    Ensure we always have k unique indices.

    If parsing fails or the model returns fewer than k, fill the rest
    by the original candidate order.
    """

    results = []

    if pred_indices is not None:
        for idx in pred_indices:
            if 0 <= idx < num_candidates and idx not in results:
                results.append(idx)

            if len(results) >= k:
                return results[:k]

    if fill_by_default_order:
        for idx in range(num_candidates):
            if idx not in results:
                results.append(idx)

            if len(results) >= k:
                break

    return results[:k]


def parse_generated_question(raw_output: Optional[str], fallback_question: str) -> str:
    """
    Parse the LLM-generated natural language question.

    Expected:
        {"question": "..."}

    Fallback:
        Use the first non-empty line or fallback_question.
    """

    if raw_output is None:
        return fallback_question

    text = clean_llm_output(raw_output)

    json_text = extract_json_object(text) or extract_json_object(raw_output)
    if json_text:
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                question = data.get("question") or data.get("query") or data.get("natural_language_question")
                if question and str(question).strip():
                    return str(question).strip()
        except Exception:
            pass

    # Remove common prefixes and take the first useful line.
    text = text.replace("Answer:", "").strip()
    lines = [line.strip(" -\t\n\r\"'") for line in text.splitlines() if line.strip()]
    for line in lines:
        if "?" in line or len(line.split()) >= 4:
            return line

    return fallback_question


# ============================================================
# True tail map
# ============================================================

def build_true_tail_map(dataset) -> Dict[Tuple[int, int], Set[int]]:
    """
    Build all true tails for filtered candidate construction.

    For query (h, r, ?), all true tails from train/valid/test are treated
    as correct answers. During candidate construction, all true tails except
    the current gold tail are removed.
    """

    true_tail_map: Dict[Tuple[int, int], Set[int]] = {}

    all_triples = []

    if hasattr(dataset, "train_triples"):
        all_triples.extend(dataset.train_triples)

    if hasattr(dataset, "valid_triples"):
        all_triples.extend(dataset.valid_triples)

    if hasattr(dataset, "test_triples"):
        all_triples.extend(dataset.test_triples)

    for tri in all_triples:
        h = int(tri.h_id)
        r = int(tri.r_id)
        t = int(tri.t_id)

        true_tail_map.setdefault((h, r), set()).add(t)

    return true_tail_map


def build_entity_relation_pattern_map(
    dataset,
    schema: DatasetSchemaHelper,
    use_train: bool = True,
    use_valid: bool = True,
    use_test: bool = False,
) -> Dict[int, Set[str]]:
    """
    Build one-hop neighborhood relation-pattern sets for each entity.

    A pattern keeps relation direction:
        - out:<relation_label> means entity --relation--> neighbor
        - in:<relation_label> means neighbor --relation--> entity

    By default this uses train + valid triples only and avoids test triples,
    so the signal will not directly leak the test answer.
    """

    pattern_map: Dict[int, Set[str]] = {}
    triples = []

    if use_train and hasattr(dataset, "train_triples"):
        triples.extend(dataset.train_triples)

    if use_valid and hasattr(dataset, "valid_triples"):
        triples.extend(dataset.valid_triples)

    if use_test and hasattr(dataset, "test_triples"):
        triples.extend(dataset.test_triples)

    for tri in triples:
        h = int(tri.h_id)
        r = int(tri.r_id)
        t = int(tri.t_id)
        relation_label = schema.relation_label(r)

        pattern_map.setdefault(h, set()).add(f"out:{relation_label}")
        pattern_map.setdefault(t, set()).add(f"in:{relation_label}")

    return pattern_map


def relation_patterns_to_text(patterns: Any, max_patterns: int = 6) -> str:
    """
    Convert shared relation patterns into compact text for prompts/debugging.
    """

    if not patterns:
        return "none"

    values = sorted([str(x) for x in patterns if str(x).strip()])

    if not values:
        return "none"

    shown = values[:max_patterns]
    text = ", ".join(shown)

    if len(values) > max_patterns:
        text += f", ... (+{len(values) - max_patterns} more)"

    return text


# ============================================================
# Candidate construction
# ============================================================

class CandidateBuilder:
    """
    Build candidate set for LLM ranking.

    Supported modes:

    1. filtered_rotate
        - remove other true tails of (h, r);
        - select RotatE top-(candidate_size - 1) negative candidates;
        - add current gold tail;
        - sort final candidate list by RotatE score.

    2. random
        - remove other true tails of (h, r);
        - randomly sample candidate_size - 1 negative candidates;
        - add current gold tail;
        - shuffle final candidate list.
    """

    def __init__(
        self,
        dataset,
        rotate_manager,
        candidate_size: int = 20,
        candidate_mode: str = "random",
        exclude_head: bool = True,
        random_seed: int = 2026,
    ):
        if candidate_mode not in {"filtered_rotate", "random"}:
            raise ValueError("candidate_mode must be filtered_rotate or random.")

        self.dataset = dataset
        self.rotate = rotate_manager
        self.schema = DatasetSchemaHelper(dataset)

        self.candidate_size = candidate_size
        self.candidate_mode = candidate_mode
        self.exclude_head = exclude_head
        self.random_seed = random_seed

        self.true_tail_map = build_true_tail_map(dataset)
        self.entity_relation_patterns = build_entity_relation_pattern_map(
            dataset=dataset,
            schema=self.schema,
            use_train=True,
            use_valid=True,
            use_test=False,
        )
        self.rng = random.Random(random_seed)

    def get_shared_relation_patterns(
        self,
        head_id: int,
        candidate_entity_id: int,
    ) -> List[str]:
        """
        Return one-hop relation patterns shared by the head entity and a candidate.

        This is a candidate-aware structural signal. If a candidate and the head
        share relation patterns such as out:locatedIn or in:contains, they may
        be structurally similar or contextually related in the KG.
        """

        head_patterns = self.entity_relation_patterns.get(int(head_id), set())
        candidate_patterns = self.entity_relation_patterns.get(int(candidate_entity_id), set())

        shared = sorted(head_patterns.intersection(candidate_patterns))
        return shared

    def build_tail_candidates(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
        query_index: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        if self.candidate_mode == "filtered_rotate":
            return self._build_filtered_rotate_candidates(
                head_id=head_id,
                relation_id=relation_id,
                gold_tail_id=gold_tail_id,
            )

        if self.candidate_mode == "random":
            return self._build_random_candidates(
                head_id=head_id,
                relation_id=relation_id,
                gold_tail_id=gold_tail_id,
                query_index=query_index,
            )

        raise ValueError(f"Unsupported candidate_mode: {self.candidate_mode}")

    def _prepare_candidate_pools(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
    ) -> Dict[str, Any]:

        head_id = int(head_id)
        relation_id = int(relation_id)
        gold_tail_id = int(gold_tail_id)

        all_entity_ids = set(int(eid) for eid in self.dataset.entities.keys())

        all_true_tails = set(
            self.true_tail_map.get((head_id, relation_id), set())
        )

        other_true_tails = set(all_true_tails)
        other_true_tails.discard(gold_tail_id)

        candidate_pool = set(all_entity_ids)

        # Remove other known correct tails.
        candidate_pool -= other_true_tails

        # Remove head entity.
        if self.exclude_head and head_id in candidate_pool:
            candidate_pool.remove(head_id)

        # Negative pool excludes current gold tail.
        negative_pool = set(candidate_pool)
        if gold_tail_id in negative_pool:
            negative_pool.remove(gold_tail_id)

        return {
            "all_entity_ids": all_entity_ids,
            "all_true_tails": all_true_tails,
            "other_true_tails": other_true_tails,
            "candidate_pool": candidate_pool,
            "negative_pool": negative_pool,
        }

    def _build_filtered_rotate_candidates(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        head_id = int(head_id)
        relation_id = int(relation_id)
        gold_tail_id = int(gold_tail_id)

        pools = self._prepare_candidate_pools(
            head_id=head_id,
            relation_id=relation_id,
            gold_tail_id=gold_tail_id,
        )

        negative_pool_list = sorted(list(pools["negative_pool"]))

        scored_negatives = self.rotate.score_tail_candidates(
            head_id=head_id,
            relation_id=relation_id,
            candidate_tail_ids=negative_pool_list,
        )

        num_negatives = max(0, self.candidate_size - 1)
        selected_negative_items = scored_negatives[:num_negatives]

        candidate_score_dict = {
            int(eid): float(score)
            for eid, score in selected_negative_items
        }

        gold_score = float(
            self.rotate.score_triples(
                [(head_id, relation_id, gold_tail_id)]
            )[0]
        )

        candidate_score_dict[gold_tail_id] = gold_score

        sorted_items = sorted(
            candidate_score_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        )[: self.candidate_size]

        candidates = self._format_candidates(
            items=sorted_items,
            head_id=head_id,
            gold_tail_id=gold_tail_id,
            other_true_tails=pools["other_true_tails"],
        )

        candidate_info = self._build_candidate_info(
            mode="filtered_rotate",
            head_id=head_id,
            relation_id=relation_id,
            gold_tail_id=gold_tail_id,
            candidates=candidates,
            pools=pools,
            gold_score=gold_score,
            extra_info={
                "candidate_policy": "filtered_rotate_top19_negatives_plus_gold",
                "candidate_order": "RotatE score descending",
            },
        )

        return candidates, candidate_info

    def _build_random_candidates(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
        query_index: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        head_id = int(head_id)
        relation_id = int(relation_id)
        gold_tail_id = int(gold_tail_id)

        pools = self._prepare_candidate_pools(
            head_id=head_id,
            relation_id=relation_id,
            gold_tail_id=gold_tail_id,
        )

        negative_pool_list = sorted(list(pools["negative_pool"]))

        num_negatives = max(0, self.candidate_size - 1)

        if len(negative_pool_list) < num_negatives:
            selected_negative_ids = list(negative_pool_list)
        else:
            # Use query-specific seed for reproducible but different random candidates.
            query_seed = self.random_seed
            if query_index is not None:
                query_seed = self.random_seed + int(query_index)

            local_rng = random.Random(query_seed)
            selected_negative_ids = local_rng.sample(
                negative_pool_list,
                num_negatives,
            )

        candidate_ids = selected_negative_ids + [gold_tail_id]

        # Shuffle final candidate order, so gold position is not fixed.
        query_seed = self.random_seed
        if query_index is not None:
            query_seed = self.random_seed + int(query_index) + 999999

        local_rng = random.Random(query_seed)
        local_rng.shuffle(candidate_ids)

        # Score candidates only for logging/debugging and RotatE baseline.
        triple_list = [
            (head_id, relation_id, int(entity_id))
            for entity_id in candidate_ids
        ]

        score_arr = self.rotate.score_triples(triple_list)

        id_score_pairs = [
            (int(entity_id), float(score))
            for entity_id, score in zip(candidate_ids, score_arr)
        ]

        gold_score = None
        for eid, score in id_score_pairs:
            if eid == gold_tail_id:
                gold_score = score
                break

        candidates = self._format_candidates(
            items=id_score_pairs,
            head_id=head_id,
            gold_tail_id=gold_tail_id,
            other_true_tails=pools["other_true_tails"],
        )

        candidate_info = self._build_candidate_info(
            mode="random",
            head_id=head_id,
            relation_id=relation_id,
            gold_tail_id=gold_tail_id,
            candidates=candidates,
            pools=pools,
            gold_score=gold_score,
            extra_info={
                "candidate_policy": "random_19_negatives_plus_gold",
                "candidate_order": "random shuffled",
                "random_seed": self.random_seed,
                "query_index": query_index,
                "num_sampled_negatives": len(selected_negative_ids),
            },
        )

        return candidates, candidate_info

    def _format_candidates(
        self,
        items: List[Tuple[int, float]],
        head_id: int,
        gold_tail_id: int,
        other_true_tails: Set[int],
    ) -> List[Dict[str, Any]]:

        candidates = []

        for index, (entity_id, score) in enumerate(items):
            shared_patterns = self.get_shared_relation_patterns(
                head_id=head_id,
                candidate_entity_id=int(entity_id),
            )

            candidates.append(
                {
                    "index": index,
                    "entity_id": int(entity_id),
                    "label": self.schema.entity_label(entity_id),
                    "classes": self.schema.entity_classes(entity_id),
                    "rotate_score": float(score),
                    "is_gold": int(entity_id) == int(gold_tail_id),
                    "is_other_true_tail": int(entity_id) in other_true_tails,
                    "shared_relation_patterns_with_head": shared_patterns,
                    "shared_relation_pattern_count": len(shared_patterns),
                    "shared_relation_pattern_text": relation_patterns_to_text(shared_patterns),
                }
            )

        return candidates

    def _build_candidate_info(
        self,
        mode: str,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
        candidates: List[Dict[str, Any]],
        pools: Dict[str, Any],
        gold_score: Optional[float],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        other_true_tails = pools["other_true_tails"]
        all_true_tails = pools["all_true_tails"]

        gold_candidate_index = None
        for cand in candidates:
            if cand["is_gold"]:
                gold_candidate_index = cand["index"]
                break

        leaked_other_true_tails = [
            cand for cand in candidates
            if cand["entity_id"] in other_true_tails
        ]

        candidate_info = {
            "candidate_mode": mode,
            "candidate_size": len(candidates),
            "gold_tail_id": int(gold_tail_id),
            "gold_tail_label": self.schema.entity_label(gold_tail_id),
            "gold_candidate_index": gold_candidate_index,
            "gold_rotate_score": gold_score,
            "num_all_true_tails_for_query": len(all_true_tails),
            "all_true_tail_ids_for_query": sorted(list(all_true_tails)),
            "all_true_tail_labels_for_query": [
                self.schema.entity_label(tid)
                for tid in sorted(list(all_true_tails))
            ],
            "filtered_other_true_tail_ids": sorted(list(other_true_tails)),
            "filtered_other_true_tail_labels": [
                self.schema.entity_label(tid)
                for tid in sorted(list(other_true_tails))
            ],
            "num_filtered_other_true_tails": len(other_true_tails),
            "num_negative_pool": len(pools["negative_pool"]),
            "leaked_other_true_tail_count": len(leaked_other_true_tails),
            "leaked_other_true_tails": leaked_other_true_tails,
        }

        if extra_info:
            candidate_info.update(extra_info)

        if gold_candidate_index is None:
            raise RuntimeError(
                f"Gold tail {gold_tail_id} is missing from candidate set. "
                f"This should never happen."
            )

        if leaked_other_true_tails:
            raise RuntimeError(
                f"Candidate set contains other true tails: {leaked_other_true_tails}"
            )

        return candidate_info


# ============================================================
# Prompt construction
# ============================================================

def split_camel_or_symbol_relation(relation_label: str) -> str:
    """
    A deterministic fallback verbalization for relation labels.
    Examples:
        hasMonthlyClimate -> has monthly climate
        filmWrittenBy -> film written by
        /people/person/place_of_birth -> people person place of birth
    """

    text = str(relation_label)
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace(".", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = " ".join(text.split())
    return text.lower() if text else str(relation_label)


def build_question_generation_prompt(
    head_label: str,
    relation_label: str,
    head_classes: List[str],
    relation_domain: List[str],
    relation_range: List[str],
) -> str:
    head_class_text = class_list_to_text(head_classes)
    domain_text = class_list_to_text(relation_domain)
    range_text = class_list_to_text(relation_range)
    relation_text = split_camel_or_symbol_relation(relation_label)

    prompt = f"""
You are a knowledge graph query verbalization assistant.

Your task is to convert a symbolic tail-prediction query into one natural-language question.

Symbolic query:
({head_label}, {relation_label}, ?)

Known schema information:
- Head entity: {head_label}
- Head entity type: {head_class_text}
- Relation label: {relation_label}
- Relation verbalization hint: {relation_text}
- Relation domain: {domain_text}
- Expected tail entity type: {range_text}

Requirements:
- Generate one concise English question.
- The question must ask for the missing tail entity.
- The question must preserve the direction from head entity to tail entity.
- Do not include candidate answers.
- Do not answer the question.
- Return only valid JSON.

Output format:
{{
  "question": "..."
}}

/no_think
""".strip()

    return prompt


def fallback_question(head_label: str, relation_label: str) -> str:
    relation_text = split_camel_or_symbol_relation(relation_label)
    return (
        f'Given the head entity "{head_label}", which candidate entity best completes '
        f'the relation "{relation_text}"?'
    )


def build_ranking_prompt(
    generated_question: str,
    head_label: str,
    relation_label: str,
    head_classes: List[str],
    relation_domain: List[str],
    relation_range: List[str],
    candidates: List[Dict[str, Any]],
    evidence_text: str,
    top_k: int = 10,
    use_evidence: bool = True,
    include_rotate_score: bool = True,
    include_candidate_class: bool = True,
) -> str:
    """
    Build the final QA-style LLM ranking prompt.

    Design:
        1. First use a generated natural-language question.
        2. Attach expected tail type and key evidence.
        3. Candidate list includes entity name, entity type, and RotatE score.
        4. LLM returns ranked candidate indices only.
    """

    candidate_lines = []
    for cand in candidates:
        details = []

        if include_candidate_class:
            details.append(f"type: {class_list_to_text(cand.get('classes', []))}")

        if include_rotate_score:
            details.append(f"RotatE score: {float(cand.get('rotate_score', 0.0)):.4f}")

        shared_pattern_text = cand.get("shared_relation_pattern_text")
        if shared_pattern_text is None:
            shared_pattern_text = relation_patterns_to_text(
                cand.get("shared_relation_patterns_with_head", [])
            )
        details.append(f"shared relation patterns with head: {shared_pattern_text}")

        detail_text = "; ".join(details)
        if detail_text:
            candidate_lines.append(f"{cand['index']}. {cand['label']} ({detail_text})")
        else:
            candidate_lines.append(f"{cand['index']}. {cand['label']}")

    indexed_candidates = "\n".join(candidate_lines)

    head_class_text = class_list_to_text(head_classes)
    domain_text = class_list_to_text(relation_domain)
    range_text = class_list_to_text(relation_range)

    expected_type_evidence = f"The expected tail entity type should be {range_text}."
    head_type_evidence = f"The head entity {head_label} has type {head_class_text}."
    relation_schema_evidence = (
        f"The relation {relation_label} usually maps from domain type {domain_text} "
        f"to range type {range_text}."
    )
    shared_pattern_evidence = (
        "For each candidate, shared relation patterns with the head are provided. "
        "These patterns are one-hop incoming/outgoing relation labels that appear in both "
        "the head entity neighborhood and the candidate entity neighborhood; use them as weak "
        "structural compatibility evidence, not as the only decision rule."
    )

    if use_evidence and evidence_text.strip():
        evidence_lines = [line.strip() for line in evidence_text.splitlines() if line.strip()]
        cleaned_evidence_lines = []
        for line in evidence_lines:
            line = re.sub(r"^\d+\.\s*", "", line).strip()
            cleaned_evidence_lines.append(line)

        all_evidence_lines = [
            expected_type_evidence,
            head_type_evidence,
            relation_schema_evidence,
            shared_pattern_evidence,
        ] + cleaned_evidence_lines
    else:
        all_evidence_lines = [
            expected_type_evidence,
            head_type_evidence,
            relation_schema_evidence,
            shared_pattern_evidence,
            "No additional structural evidence is provided.",
        ]

    key_evidence_text = "\n".join(
        f"{idx}. {line}" for idx, line in enumerate(all_evidence_lines, start=1)
    )

    prompt = f"""
You are an expert knowledge graph question-answering and entity-ranking system.

You will be given:
1. A natural-language question converted from a knowledge graph query.
2. Key evidence from the knowledge graph.
3. A fixed list of candidate tail entities.

Your task is to rank the candidate entities as answers to the question.

Question:
{generated_question}

Original symbolic query:
({head_label}, {relation_label}, ?)

Important rules:
- You are predicting ONLY the missing tail entity.
- The head entity "{head_label}" is NOT a valid answer.
- You MUST select exactly {top_k} candidate indices.
- The selected indices must be ordered from MOST likely to LEAST likely.
- You MUST only choose indices from the candidate list.
- Candidate indices are 0-based.
- Do NOT output entity names.
- Do NOT generate new entities.
- Do NOT repeat indices.
- Return ONLY valid JSON.
- Do NOT output reasoning.

Use the following information carefully:
- The natural-language question tells you what is being asked.
- The key evidence provides local graph context and expected answer type.
- Candidate entity types indicate semantic compatibility.
- RotatE scores provide structural plausibility, but they are not always sufficient.
- Shared relation patterns with the head indicate whether a candidate has similar one-hop neighborhood relation patterns to the head; use this as weak structural evidence.

Key evidence from the knowledge graph:
{key_evidence_text}

Candidate tail entities:
{indexed_candidates}

Output format:
{{
  "selected_indices": [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
}}

/no_think
""".strip()

    return prompt


def print_candidate_debug_table(
    query_index: Any,
    head_label: str,
    relation_label: str,
    gold_tail_id: int,
    gold_tail_label: str,
    candidates: List[Dict[str, Any]],
    candidate_info: Dict[str, Any],
) -> None:
    """
    Print all candidate entities with their classes for debugging.
    """

    print("\n" + "=" * 120)
    print("[Debug] Candidate entities and classes")
    print("=" * 120)
    print(f"Query index: {query_index}")
    print(f"Query: ({head_label}, {relation_label}, ?)")
    print(f"Correct entity: {gold_tail_label} | id={gold_tail_id}")
    print(f"Gold candidate index: {candidate_info.get('gold_candidate_index')}")
    print(f"Candidate mode: {candidate_info.get('candidate_mode')}")
    print(f"Candidate policy: {candidate_info.get('candidate_policy')}")
    print("-" * 120)

    for cand in candidates:
        index = cand.get("index")
        entity_id = cand.get("entity_id")
        label = cand.get("label")
        class_text = class_list_to_text(cand.get("classes", []))
        rotate_score = cand.get("rotate_score")
        is_gold = cand.get("is_gold", False)

        gold_mark = " <== GOLD / CORRECT" if is_gold else ""

        print(
            f"[{index:02d}] "
            f"id={entity_id} | "
            f"name={label} | "
            f"class=[{class_text}] | "
            f"RotatE={float(rotate_score):.4f}"
            f"{gold_mark}"
        )

    print("-" * 120)
    print("Filtered other true tails:")
    print(candidate_info.get("filtered_other_true_tail_labels", []))
    print("=" * 120 + "\n")


# ============================================================
# Metrics
# ============================================================

def compute_rank_metrics(
    ranked_entity_ids: List[int],
    gold_tail_id: int,
) -> Dict[str, Any]:
    gold_tail_id = int(gold_tail_id)

    if gold_tail_id in ranked_entity_ids:
        rank = ranked_entity_ids.index(gold_tail_id) + 1
    else:
        rank = None

    if rank is None:
        return {
            "rank": None,
            "MRR": 0.0,
            "Hit@1": 0,
            "Hit@3": 0,
            "Hit@10": 0,
        }

    return {
        "rank": rank,
        "MRR": 1.0 / rank,
        "Hit@1": int(rank <= 1),
        "Hit@3": int(rank <= 3),
        "Hit@10": int(rank <= 10),
    }


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "MRR": 0.0,
            "Hit@1": 0.0,
            "Hit@3": 0.0,
            "Hit@10": 0.0,
            "failed_count": 0,
            "parse_failed_count": 0,
            "candidate_error_count": 0,
        }

    return {
        "total": total,
        "MRR": sum(x.get("MRR", 0.0) for x in results) / total,
        "Hit@1": sum(x.get("Hit@1", 0) for x in results) / total,
        "Hit@3": sum(x.get("Hit@3", 0) for x in results) / total,
        "Hit@10": sum(x.get("Hit@10", 0) for x in results) / total,
        "failed_count": sum(1 for x in results if x.get("status") != "ok"),
        "parse_failed_count": sum(
            1 for x in results
            if x.get("status") == "failed_parse_filled_by_default_order"
        ),
        "candidate_error_count": sum(
            1 for x in results
            if "candidate" in str(x.get("status", ""))
        ),
    }


def aggregate_default_candidate_baseline(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Baseline using the original candidate order.

    In filtered_rotate mode, this is RotatE order.
    In random mode, this is random order.
    """

    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "MRR": 0.0,
            "Hit@1": 0.0,
            "Hit@3": 0.0,
            "Hit@10": 0.0,
        }

    return {
        "total": total,
        "MRR": sum(x.get("default_candidate_baseline", {}).get("MRR", 0.0) for x in results) / total,
        "Hit@1": sum(x.get("default_candidate_baseline", {}).get("Hit@1", 0) for x in results) / total,
        "Hit@3": sum(x.get("default_candidate_baseline", {}).get("Hit@3", 0) for x in results) / total,
        "Hit@10": sum(x.get("default_candidate_baseline", {}).get("Hit@10", 0) for x in results) / total,
    }


# ============================================================
# Evaluator
# ============================================================

class LLMRankEvaluator:
    def __init__(
        self,
        dataset,
        rotate_manager,
        llm: Optional[LLM_Model],
        candidate_size: int = 20,
        candidate_mode: str = "random",
        random_seed: int = 2026,
        top_k: int = 10,
        use_evidence: bool = True,
        llm_enabled: bool = True,
        use_query_verbalization: bool = True,
        include_rotate_score: bool = True,
        include_candidate_class: bool = True,
        fill_parse_failed_by_default_order: bool = True,
        save_prompt: bool = True,
        save_raw_output: bool = True,
        save_question_prompt: bool = True,
        max_retries: int = 2,
        retry_sleep: float = 2.0,
        debug_print_prompt: bool = False,
        debug_print_candidates: bool = False,
        debug_print_raw_output: bool = False,
        debug_print_prediction: bool = False,
    ):
        self.dataset = dataset
        self.rotate = rotate_manager
        self.llm = llm

        self.candidate_size = candidate_size
        self.candidate_mode = candidate_mode
        self.random_seed = random_seed
        self.top_k = top_k

        self.use_evidence = use_evidence
        self.llm_enabled = llm_enabled
        self.use_query_verbalization = use_query_verbalization

        self.include_rotate_score = include_rotate_score
        self.include_candidate_class = include_candidate_class
        self.fill_parse_failed_by_default_order = fill_parse_failed_by_default_order

        self.save_prompt = save_prompt
        self.save_raw_output = save_raw_output
        self.save_question_prompt = save_question_prompt

        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        self.debug_print_prompt = debug_print_prompt
        self.debug_print_candidates = debug_print_candidates
        self.debug_print_raw_output = debug_print_raw_output
        self.debug_print_prediction = debug_print_prediction

        self.schema = DatasetSchemaHelper(dataset)
        self.candidate_builder = CandidateBuilder(
            dataset=dataset,
            rotate_manager=rotate_manager,
            candidate_size=candidate_size,
            candidate_mode=candidate_mode,
            exclude_head=True,
            random_seed=random_seed,
        )

    def evaluate_one_record(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:

        query = record["query"]

        query_index = record.get("query_index")

        head_id = int(query["head_id"])
        relation_id = int(query["relation_id"])
        gold_tail_id = int(query["tail_id"])

        head_label = query.get("head_label", self.schema.entity_label(head_id))
        relation_label = query.get("relation_label", self.schema.relation_label(relation_id))
        gold_tail_label = query.get("tail_label", self.schema.entity_label(gold_tail_id))

        head_classes = query.get("head_classes", self.schema.entity_classes(head_id))
        relation_domain = query.get("relation_domain", self.schema.relation_domain(relation_id))
        relation_range = query.get("relation_range", self.schema.relation_range(relation_id))

        evidence_text = record.get("evidence_text", "")

        if not self.use_evidence:
            evidence_text_for_prompt = ""
        else:
            evidence_text_for_prompt = evidence_text

        candidates, candidate_info = self.candidate_builder.build_tail_candidates(
            head_id=head_id,
            relation_id=relation_id,
            gold_tail_id=gold_tail_id,
            query_index=query_index,
        )

        default_ranked_entity_ids = [
            int(c["entity_id"])
            for c in candidates[: self.top_k]
        ]

        default_candidate_baseline = compute_rank_metrics(
            ranked_entity_ids=default_ranked_entity_ids,
            gold_tail_id=gold_tail_id,
        )

        fallback_q = fallback_question(head_label, relation_label)
        question_prompt = build_question_generation_prompt(
            head_label=head_label,
            relation_label=relation_label,
            head_classes=head_classes,
            relation_domain=relation_domain,
            relation_range=relation_range,
        )

        if self.llm_enabled and self.use_query_verbalization:
            generated_question_raw = self.query_llm_raw(
                prompt=question_prompt,
                system_prompt=(
                    "You are a strict JSON-output knowledge graph query verbalization assistant. "
                    "Return JSON only."
                ),
            )
            generated_question = parse_generated_question(generated_question_raw, fallback_q)
        else:
            generated_question_raw = None
            generated_question = fallback_q

        prompt = build_ranking_prompt(
            generated_question=generated_question,
            head_label=head_label,
            relation_label=relation_label,
            head_classes=head_classes,
            relation_domain=relation_domain,
            relation_range=relation_range,
            candidates=candidates,
            evidence_text=evidence_text_for_prompt,
            top_k=self.top_k,
            use_evidence=self.use_evidence,
            include_rotate_score=self.include_rotate_score,
            include_candidate_class=self.include_candidate_class,
        )

        if self.debug_print_candidates:
            print_candidate_debug_table(
                query_index=query_index,
                head_label=head_label,
                relation_label=relation_label,
                gold_tail_id=gold_tail_id,
                gold_tail_label=gold_tail_label,
                candidates=candidates,
                candidate_info=candidate_info,
            )

        if self.debug_print_prompt:
            print("\n" + "=" * 100)
            print("[Debug] Question verbalization prompt")
            print("=" * 100)
            print(f"Query index: {query_index}")
            print(f"Query: ({head_label}, {relation_label}, ?)")
            print("-" * 100)
            print(question_prompt)
            print("=" * 100 + "\n")

            print("\n" + "=" * 100)
            print("[Debug] Generated question")
            print("=" * 100)
            print(f"Generated question: {generated_question}")
            print(f"Raw verbalization output: {generated_question_raw}")
            print("=" * 100 + "\n")

            print("\n" + "=" * 100)
            print("[Debug] Final ranking prompt sent to LLM")
            print("=" * 100)
            print(f"Query index: {query_index}")
            print(f"Correct entity: {gold_tail_label} | id={gold_tail_id}")
            print("-" * 100)
            print(prompt)
            print("=" * 100 + "\n")

        if not self.llm_enabled:
            llm_raw_output = None
            pred_indices_raw = list(range(min(self.top_k, len(candidates))))
            pred_indices_final = pred_indices_raw
            status = "default_candidate_order_only"

        else:
            llm_raw_output = self.query_llm_raw(
                prompt=prompt,
                system_prompt="You are a strict JSON-output knowledge graph QA ranker. Return JSON only.",
            )

            if self.debug_print_raw_output:
                print("\n" + "=" * 100)
                print("[Debug] Ranking LLM raw output")
                print("=" * 100)
                print(llm_raw_output)
                print("=" * 100 + "\n")

            if llm_raw_output is None:
                pred_indices_raw = None
                pred_indices_final = complete_topk_indices(
                    pred_indices=None,
                    num_candidates=len(candidates),
                    k=self.top_k,
                    fill_by_default_order=self.fill_parse_failed_by_default_order,
                )
                status = "failed_llm_api"

            else:
                pred_indices_raw = extract_selected_indices(
                    llm_raw_output,
                    k=self.top_k,
                    num_candidates=len(candidates),
                )

                pred_indices_final = complete_topk_indices(
                    pred_indices=pred_indices_raw,
                    num_candidates=len(candidates),
                    k=self.top_k,
                    fill_by_default_order=self.fill_parse_failed_by_default_order,
                )

                if pred_indices_raw is None:
                    status = "failed_parse_filled_by_default_order"
                else:
                    status = "ok"

        ranked_candidates = [
            candidates[idx]
            for idx in pred_indices_final
            if 0 <= idx < len(candidates)
        ]

        ranked_entity_ids = [
            int(item["entity_id"])
            for item in ranked_candidates
        ]

        metrics = compute_rank_metrics(
            ranked_entity_ids=ranked_entity_ids,
            gold_tail_id=gold_tail_id,
        )

        extracted_result = {
            "pred_indices_raw": pred_indices_raw,
            "pred_indices_final": pred_indices_final,
            "predicted_entity_ids": ranked_entity_ids,
            "predicted_labels": [item["label"] for item in ranked_candidates],
            "predicted_classes": [item.get("classes", []) for item in ranked_candidates],
        }

        if self.debug_print_prediction:
            print("\n" + "=" * 100)
            print("[Debug] Prediction result")
            print("=" * 100)
            print(f"Query index: {query_index}")
            print(f"Question: {generated_question}")
            print(f"Original query: ({head_label}, {relation_label}, ?)")
            print(f"Correct entity: {gold_tail_label} | id={gold_tail_id}")
            print(f"Correct entity classes: {class_list_to_text(self.schema.entity_classes(gold_tail_id))}")
            print(f"Gold candidate index: {candidate_info.get('gold_candidate_index')}")
            print(f"Raw extracted indices: {pred_indices_raw}")
            print(f"Final used indices: {pred_indices_final}")
            print("\nPredicted entity sequence:")

            for rank_id, item in enumerate(ranked_candidates, start=1):
                mark = " <== CORRECT" if int(item["entity_id"]) == gold_tail_id else ""
                print(
                    f"{rank_id}. "
                    f"candidate_index={item['index']} | "
                    f"id={item['entity_id']} | "
                    f"name={item['label']} | "
                    f"class=[{class_list_to_text(item.get('classes', []))}] | "
                    f"RotatE={float(item.get('rotate_score', 0.0)):.4f} | "
                    f"shared_patterns=[{item.get('shared_relation_pattern_text', 'none')}]"
                    f"{mark}"
                )

            print(f"\nFinal rank of correct entity: {metrics['rank']}")
            print(f"Hit@1={metrics['Hit@1']} | Hit@3={metrics['Hit@3']} | Hit@10={metrics['Hit@10']}")
            print("=" * 100 + "\n")

        result = {
            "query_index": query_index,

            "query_triple_id": {
                "head_id": head_id,
                "relation_id": relation_id,
                "tail_id": gold_tail_id,
            },

            "query_triple_name": {
                "head": head_label,
                "relation": relation_label,
                "tail": gold_tail_label,
            },

            "generated_question": generated_question,
            "generated_question_raw": generated_question_raw,
            "question_prompt": question_prompt if self.save_question_prompt else None,

            "query_info": {
                "head_classes": head_classes,
                "relation_domain": relation_domain,
                "relation_range": relation_range,
                "tail_classes": self.schema.entity_classes(gold_tail_id),
            },

            "correct_answer": {
                "id": gold_tail_id,
                "label": gold_tail_label,
                "classes": self.schema.entity_classes(gold_tail_id),
            },

            "candidate_info": candidate_info,
            "candidate_mode": self.candidate_mode,
            "candidate_size": len(candidates),
            "top_k": self.top_k,

            "candidates": candidates,

            "use_evidence": self.use_evidence,
            "key_evidence": record.get("key_evidence", []) if self.use_evidence else [],
            "evidence_text": evidence_text_for_prompt,

            "llm_enabled": self.llm_enabled,
            "use_query_verbalization": self.use_query_verbalization,
            "llm_raw_output": llm_raw_output if self.save_raw_output else None,
            "extracted_result": extracted_result,

            "pred_indices_raw": pred_indices_raw,
            "pred_indices_final": pred_indices_final,

            "predicted_tails": [
                {
                    "rank": rank + 1,
                    "candidate_index": item["index"],
                    "entity_id": item["entity_id"],
                    "label": item["label"],
                    "classes": item.get("classes", []),
                    "rotate_score": item.get("rotate_score"),
                    "shared_relation_patterns_with_head": item.get("shared_relation_patterns_with_head", []),
                    "shared_relation_pattern_count": item.get("shared_relation_pattern_count", 0),
                    "shared_relation_pattern_text": item.get("shared_relation_pattern_text", "none"),
                    "is_gold": item.get("is_gold", False),
                }
                for rank, item in enumerate(ranked_candidates)
            ],

            "default_candidate_baseline": default_candidate_baseline,

            "rank": metrics["rank"],
            "MRR": metrics["MRR"],
            "Hit@1": metrics["Hit@1"],
            "Hit@3": metrics["Hit@3"],
            "Hit@10": metrics["Hit@10"],

            "status": status,
        }

        if self.save_prompt:
            result["prompt"] = prompt

        return result

    def query_llm_raw(self, prompt: str, system_prompt: str) -> Optional[str]:
        """
        Use the existing LLM_Model wrapper but keep raw content as much as possible.
        """

        if self.llm is None:
            return None

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.openai_client.chat.completions.create(
                    **self.llm.llm_config,
                    messages=messages,
                )

                content = response.choices[0].message.content

                if content is None:
                    return ""

                return content

            except Exception as e:
                last_error = repr(e)
                print(
                    f"\n[Warning] LLM request failed, "
                    f"attempt {attempt + 1}/{self.max_retries + 1}: {last_error}"
                )
                time.sleep(self.retry_sleep)

        return None


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    output_path: str | Path,
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    metrics = aggregate_metrics(results)
    default_metrics = aggregate_default_candidate_baseline(results)

    save_obj = {
        "config": config,
        "metrics": metrics,
        "default_candidate_baseline_metrics": default_metrics,
        "results": results,
    }

    save_json(save_obj, output_path)

    print(f"\n[Checkpoint Saved] {output_path}")
    print(
        f"LLM: MRR={metrics['MRR']:.4f} | "
        f"Hit@1={metrics['Hit@1']:.4f} | "
        f"Hit@3={metrics['Hit@3']:.4f} | "
        f"Hit@10={metrics['Hit@10']:.4f} | "
        f"Failed={metrics['failed_count']} | "
        f"ParseFailed={metrics['parse_failed_count']}"
    )
    print(
        f"Default candidate order baseline: "
        f"MRR={default_metrics['MRR']:.4f} | "
        f"Hit@1={default_metrics['Hit@1']:.4f} | "
        f"Hit@3={default_metrics['Hit@3']:.4f} | "
        f"Hit@10={default_metrics['Hit@10']:.4f}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM QA-style ranking evaluation for OD-KGC."
    )

    # Dataset and cache
    parser.add_argument("--data_path", type=str, default="data/FB15k-237")
    parser.add_argument("--dataset_name", type=str, default="FB15k-237")
    parser.add_argument("--import_path", type=str, default="import")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument(
        "--compressed_evidence_path",
        type=str,
        default="import/evidence/FB15k-237/test_compressed_evidence.jsonl",
        help="Path to compressed evidence jsonl.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="import/eval/FB15k-237/test_llm_rank_debug.json",
        help="Path to save evaluation results.",
    )

    # Candidate config
    parser.add_argument(
        "--candidate_mode",
        type=str,
        default="filtered_rotate",
        choices=["filtered_rotate", "random"],
        help=(
            "filtered_rotate: RotatE top-19 negatives + gold; "
            "random: random 19 negatives + gold."
        ),
    )
    parser.add_argument("--candidate_size", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=206)
    parser.add_argument("--top_k", type=int, default=10)

    # Evidence switch
    parser.add_argument(
        "--disable_evidence",
        action="store_true",
        default=False,
        help="Do not use compressed evidence in the prompt.",
    )

    parser.add_argument(
        "--disable_query_verbalization",
        action="store_true",
        default=False,
        help="Do not call LLM to verbalize the query. Use a deterministic generic question instead.",
    )

    parser.add_argument(
        "--disable_rotate_score",
        action="store_true",
        default=False,
        help="Do not show RotatE scores in candidate list.",
    )

    parser.add_argument(
        "--disable_candidate_class",
        action="store_true",
        default=False,
        help="Do not show candidate classes in candidate list.",
    )

    # LLM / baseline switch
    parser.add_argument(
        "--rotate_only",
        action="store_true",
        default=False,
        help="Do not call LLM. Use default candidate order as prediction.",
    )

    parser.add_argument(
        "--no_fill_parse_failed",
        action="store_true",
        default=False,
        help="Do not fill parse-failed LLM outputs by default candidate order.",
    )

    # Evaluation range
    parser.add_argument("--max_items", type=int, default=1000)
    parser.add_argument("--start_index", type=int, default=0)

    # RotatE config
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--gpu_id", type=int, default=0)

    # LLM config: defaults are your successful command
    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:22014/v1")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)

    # Runtime
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=20)

    # Debug output
    parser.add_argument("--debug_print_prompt", action="store_true", default=False)
    parser.add_argument("--debug_print_candidates", action="store_true", default=False)
    parser.add_argument("--debug_print_raw_output", action="store_true", default=False)
    parser.add_argument("--debug_print_prediction", action="store_true", default=False)

    parser.add_argument(
        "--no_save_prompt",
        action="store_true",
        default=False,
        help="Do not save full prompt in result json.",
    )

    parser.add_argument(
        "--no_save_raw_output",
        action="store_true",
        default=False,
        help="Do not save raw LLM output in result json.",
    )

    parser.add_argument(
        "--no_save_question_prompt",
        action="store_true",
        default=False,
        help="Do not save question verbalization prompt in result json.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_random_seed(args.random_seed)

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name
    import_root = Path(args.import_path)

    compressed_evidence_path = Path(args.compressed_evidence_path)
    output_path = Path(args.output_path)

    use_evidence = not args.disable_evidence
    llm_enabled = not args.rotate_only
    use_query_verbalization = not args.disable_query_verbalization

    print("=" * 100)
    print("[OD-KGC QA-style LLM Ranking Evaluation]")
    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print(f"Compressed evidence: {compressed_evidence_path}")
    print(f"Output path: {output_path}")
    print(f"Candidate mode: {args.candidate_mode}")
    print(f"Candidate size: {args.candidate_size}")
    print(f"Random seed: {args.random_seed}")
    print(f"Top K: {args.top_k}")
    print(f"Use evidence: {use_evidence}")
    print(f"Use query verbalization: {use_query_verbalization}")
    print(f"LLM enabled: {llm_enabled}")
    print(f"LLM model: {args.llm_model}")
    print(f"OpenAI base URL: {args.openai_base_url}")
    print("=" * 100)

    print("[Evaluator] Loading dataset...")
    loader = KGLoader(data_path)
    dataset = loader.load()

    print("[Evaluator] Loading compressed evidence...")
    compressed_records = load_jsonl(compressed_evidence_path)

    print(f"[Evaluator] Original records: {len(compressed_records)}")

    if args.start_index > 0:
        compressed_records = compressed_records[args.start_index:]

    if args.max_items is not None and args.max_items > 0:
        compressed_records = compressed_records[: args.max_items]

    print(f"[Evaluator] Records to evaluate: {len(compressed_records)}")

    print("[Evaluator] Loading trained RotatE...")
    rotate = get_or_train_rotate(
        data_path=str(data_path),
        import_path=str(import_root / "KGE_model"),
        dataset_name=dataset_name,
        load_if_exists=True,
        force_train=False,
        cuda=args.cuda,
        gpu_id=args.gpu_id,
    )

    if llm_enabled:
        print("[Evaluator] Initializing LLM...")
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
        print("[Evaluator] rotate_only mode: LLM disabled.")

    evaluator = LLMRankEvaluator(
        dataset=dataset,
        rotate_manager=rotate,
        llm=llm,
        candidate_size=args.candidate_size,
        candidate_mode=args.candidate_mode,
        random_seed=args.random_seed,
        top_k=args.top_k,
        use_evidence=use_evidence,
        llm_enabled=llm_enabled,
        use_query_verbalization=use_query_verbalization,
        include_rotate_score=not args.disable_rotate_score,
        include_candidate_class=not args.disable_candidate_class,
        fill_parse_failed_by_default_order=not args.no_fill_parse_failed,
        save_prompt=not args.no_save_prompt,
        save_raw_output=not args.no_save_raw_output,
        save_question_prompt=not args.no_save_question_prompt,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        debug_print_prompt=args.debug_print_prompt,
        debug_print_candidates=args.debug_print_candidates,
        debug_print_raw_output=args.debug_print_raw_output,
        debug_print_prediction=args.debug_print_prediction,
    )

    config = {
        "dataset": dataset_name,
        "data_path": str(data_path),
        "compressed_evidence_path": str(compressed_evidence_path),
        "candidate_mode": args.candidate_mode,
        "candidate_size": args.candidate_size,
        "random_seed": args.random_seed,
        "top_k": args.top_k,
        "use_evidence": use_evidence,
        "use_query_verbalization": use_query_verbalization,
        "llm_enabled": llm_enabled,
        "prompt_style": "qa_question_with_evidence_candidate_type_rotate_score_and_shared_relation_patterns",
        "include_rotate_score": not args.disable_rotate_score,
        "include_candidate_class": not args.disable_candidate_class,
        "fill_parse_failed_by_default_order": not args.no_fill_parse_failed,
        "save_prompt": not args.no_save_prompt,
        "save_raw_output": not args.no_save_raw_output,
        "save_question_prompt": not args.no_save_question_prompt,
        "llm_model": args.llm_model,
        "openai_base_url": args.openai_base_url,
        "max_items": args.max_items,
        "start_index": args.start_index,
    }

    results = []

    for idx, record in enumerate(
        tqdm(compressed_records, desc="LLM QA Ranking", ncols=100),
        start=1,
    ):
        try:
            result = evaluator.evaluate_one_record(record)

        except Exception as e:
            result = {
                "query_index": record.get("query_index"),
                "query_triple_id": record.get("query_triple_id"),
                "query_triple_name": record.get("query_triple_name"),
                "correct_answer": record.get("correct_answer"),
                "rank": None,
                "MRR": 0.0,
                "Hit@1": 0,
                "Hit@3": 0,
                "Hit@10": 0,
                "status": f"failed_exception: {repr(e)}",
            }

        results.append(result)

        if idx % args.save_every == 0:
            save_checkpoint(
                output_path=output_path,
                results=results,
                config=config,
            )

            last = results[-1]
            print("=" * 60)
            print(f"Processed: {idx}/{len(compressed_records)}")
            print(f"Last status: {last.get('status')}")
            print(f"Last query: {last.get('query_triple_name')}")
            print(f"Last generated question: {last.get('generated_question')}")
            print(f"Last pred raw: {last.get('pred_indices_raw')}")
            print(f"Last pred final: {last.get('pred_indices_final')}")
            print(f"Last rank: {last.get('rank')}")
            print("Last predicted entities:")
            for item in last.get("predicted_tails", []):
                mark = " <== CORRECT" if item.get("is_gold") else ""
                print(
                    f"{item.get('rank')}. "
                    f"index={item.get('candidate_index')} | "
                    f"name={item.get('label')} | "
                    f"class=[{class_list_to_text(item.get('classes', []))}] | "
                    f"RotatE={float(item.get('rotate_score', 0.0)):.4f} | "
                    f"shared_patterns=[{item.get('shared_relation_pattern_text', 'none')}]"
                    f"{mark}"
                )
            print("=" * 60)

    save_checkpoint(
        output_path=output_path,
        results=results,
        config=config,
    )

    final_metrics = aggregate_metrics(results)
    default_metrics = aggregate_default_candidate_baseline(results)

    print("\n" + "=" * 100)
    print("[Final Metrics]")
    print("=" * 100)
    print("[LLM QA Ranking]")
    print(f"Total: {final_metrics['total']}")
    print(f"MRR: {final_metrics['MRR']:.4f}")
    print(f"Hit@1: {final_metrics['Hit@1']:.4f}")
    print(f"Hit@3: {final_metrics['Hit@3']:.4f}")
    print(f"Hit@10: {final_metrics['Hit@10']:.4f}")
    print(f"Failed Count: {final_metrics['failed_count']}")
    print(f"Parse Failed Count: {final_metrics['parse_failed_count']}")
    print(f"Candidate Error Count: {final_metrics['candidate_error_count']}")

    print("\n[Default Candidate Order Baseline]")
    print(f"MRR: {default_metrics['MRR']:.4f}")
    print(f"Hit@1: {default_metrics['Hit@1']:.4f}")
    print(f"Hit@3: {default_metrics['Hit@3']:.4f}")
    print(f"Hit@10: {default_metrics['Hit@10']:.4f}")

    print(f"\nSaved to: {output_path}")

    if llm is not None:
        llm.close()


if __name__ == "__main__":
    main()
