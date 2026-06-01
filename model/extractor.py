from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader
from model.KGE_model import load_trained_rotate, get_or_train_rotate

TripleID = Tuple[int, int, int]  # (h_id, r_id, t_id)

# ============================================================
# Data structures
# ============================================================

@dataclass
class KGEdge:
    h_id: int
    r_id: int
    t_id: int
    h_label: str
    r_label: str
    t_label: str
    h_classname: Optional[str] = None
    t_classname: Optional[str] = None


@dataclass
class OneHopEvidence:
    h_id: int
    r_id: int
    t_id: int
    h_label: str
    r_label: str
    t_label: str
    query_relation_id: int
    query_relation_label: str
    relation_similarity: float
    triple_reliability: float
    score: float
    h_classname: Optional[str] = None
    t_classname: Optional[str] = None
    text: Optional[str] = None


@dataclass
class PathStep:
    h_id: int
    r_id: int
    t_id: int
    h_label: str
    r_label: str
    t_label: str
    h_classname: Optional[str] = None
    t_classname: Optional[str] = None


@dataclass
class PathEvidence:
    path: List[PathStep]
    query_head_id: int
    query_relation_id: int
    terminal_entity_id: int
    query_head_label: str
    query_relation_label: str
    terminal_entity_label: str
    path_relation_consistency: float
    endpoint_alignment: float
    length_decay: float
    score: float
    length: int
    text: Optional[str] = None


@dataclass
class ExtractedEvidence:
    query_head_id: int
    query_relation_id: int
    gold_tail_id: Optional[int]
    query_head_label: str
    query_relation_label: str
    gold_tail_label: Optional[str]
    one_hop: List[OneHopEvidence]
    paths: List[PathEvidence]

# ============================================================
# Evidence Extractor
# ============================================================

class EvidenceExtractor:
    """
    RotatE-guided evidence extractor for OD-KGC.

    Extracts one-hop and multi-hop structural evidence while
    filtering direct query triples to prevent trivial leakage.
    """

    def __init__(
        self,
        dataset,
        rotate_manager,
        use_train: bool = True,
        use_valid: bool = True,
        use_test: bool = False,
        normalize_scores: bool = True,
        lambda_relation: float = 1.0,
        lambda_triple: float = 1.0,
        alpha_path_relation: float = 1.0,
        beta_endpoint: float = 1.0,
        delta_length: float = 1.0,
    ):
        self.dataset = dataset
        self.rotate = rotate_manager
        self.use_train = use_train
        self.use_valid = use_valid
        self.use_test = use_test
        self.normalize_scores = normalize_scores
        self.lambda_relation = lambda_relation
        self.lambda_triple = lambda_triple
        self.alpha_path_relation = alpha_path_relation
        self.beta_endpoint = beta_endpoint
        self.delta_length = delta_length

        self.entities = dataset.entities
        self.relations = dataset.relations
        self.entity_embedding = self.rotate.get_entity_embedding()
        self.relation_embedding = self.rotate.get_relation_embedding()

        self.out_adj: Dict[int, List[KGEdge]] = {}
        self.hr_to_tails: Dict[Tuple[int, int], Set[int]] = {}

        self._build_graph()

    # --------------------------------------------------------
    # Graph construction
    # --------------------------------------------------------

    def _build_graph(self) -> None:
        triples = []
        if self.use_train:
            triples.extend(self.dataset.train_triples)
        if self.use_valid:
            triples.extend(self.dataset.valid_triples)
        if self.use_test:
            triples.extend(self.dataset.test_triples)

        for tri in triples:
            edge = KGEdge(
                h_id=tri.h_id,
                r_id=tri.r_id,
                t_id=tri.t_id,
                h_label=tri.h_label,
                r_label=tri.r_label,
                t_label=tri.t_label,
                h_classname=getattr(tri, "h_classname", None),
                t_classname=getattr(tri, "t_classname", None),
            )
            self.out_adj.setdefault(edge.h_id, []).append(edge)
            key = (edge.h_id, edge.r_id)
            self.hr_to_tails.setdefault(key, set()).add(edge.t_id)

        print(f"[EvidenceExtractor] Graph built with {sum(len(v) for v in self.out_adj.values())} edges.")

    # --------------------------------------------------------
    # Leakage control
    # --------------------------------------------------------

    def _is_query_relation_leakage(self, edge: KGEdge, query_head_id: int, query_relation_id: int) -> bool:
        """
        Filter edges directly matching the query pattern (h, r, *)
        """
        return edge.h_id == query_head_id and edge.r_id == query_relation_id

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def extract_for_query(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: Optional[int] = None,
        top_k_one_hop: int = 10,
        top_k_paths: int = 10,
        max_hops: int = 2,
        max_branch_per_node: int = 20,
        max_paths_before_ranking: int = 1000,
    ) -> ExtractedEvidence:

        one_hop = self.extract_one_hop(
            head_id=head_id,
            relation_id=relation_id,
            top_k=top_k_one_hop,
        )

        paths = self.extract_paths(
            head_id=head_id,
            relation_id=relation_id,
            top_k=top_k_paths,
            max_hops=max_hops,
            max_branch_per_node=max_branch_per_node,
            max_paths_before_ranking=max_paths_before_ranking,
        )

        h_label = self._entity_label(head_id)
        r_label = self._relation_label(relation_id)
        t_label = self._entity_label(gold_tail_id) if gold_tail_id is not None else None

        return ExtractedEvidence(
            query_head_id=head_id,
            query_relation_id=relation_id,
            gold_tail_id=gold_tail_id,
            query_head_label=h_label,
            query_relation_label=r_label,
            gold_tail_label=t_label,
            one_hop=one_hop,
            paths=paths,
        )

    # --------------------------------------------------------
    # One-hop evidence
    # --------------------------------------------------------

    def extract_one_hop(self, head_id: int, relation_id: int, top_k: int = 10) -> List[OneHopEvidence]:
        edges = self.out_adj.get(head_id, [])
        if not edges:
            return []

        # Only filter edges matching query relation
        valid_edges = [
            edge
            for edge in edges
            if not self._is_query_relation_leakage(edge, head_id, relation_id)
        ]
        if not valid_edges:
            return []

        relation_scores = [self._relation_cosine(edge.r_id, relation_id) for edge in valid_edges]
        triples_to_score = [(edge.h_id, edge.r_id, edge.t_id) for edge in valid_edges]
        triple_scores = self.rotate.score_triples(triples_to_score).tolist()

        if self.normalize_scores:
            relation_scores_for_rank = self._zscore(relation_scores)
            triple_scores_for_rank = self._zscore(triple_scores)
        else:
            relation_scores_for_rank = relation_scores
            triple_scores_for_rank = triple_scores

        evidence_items = []
        query_relation_label = self._relation_label(relation_id)
        for edge, rel_sim, triple_rel, rel_rank, triple_rank in zip(
            valid_edges, relation_scores, triple_scores, relation_scores_for_rank, triple_scores_for_rank
        ):
            score = self.lambda_relation * rel_rank + self.lambda_triple * triple_rank
            text = f"{edge.h_label} --[{edge.r_label}]--> {edge.t_label}"
            evidence_items.append(
                OneHopEvidence(
                    h_id=edge.h_id,
                    r_id=edge.r_id,
                    t_id=edge.t_id,
                    h_label=edge.h_label,
                    r_label=edge.r_label,
                    t_label=edge.t_label,
                    query_relation_id=relation_id,
                    query_relation_label=query_relation_label,
                    relation_similarity=float(rel_sim),
                    triple_reliability=float(triple_rel),
                    score=float(score),
                    h_classname=edge.h_classname,
                    t_classname=edge.t_classname,
                    text=text,
                )
            )
        evidence_items.sort(key=lambda x: x.score, reverse=True)
        return evidence_items[:top_k]

    # --------------------------------------------------------
    # Multi-hop path evidence
    # --------------------------------------------------------

    def extract_paths(
        self,
        head_id: int,
        relation_id: int,
        top_k: int = 10,
        max_hops: int = 2,
        max_branch_per_node: int = 20,
        max_paths_before_ranking: int = 1000,
    ) -> List[PathEvidence]:

        if max_hops < 2:
            return []

        raw_paths: List[List[KGEdge]] = []

        self._dfs_paths(
            current_entity=head_id,
            relation_id=relation_id,
            current_path=[],
            visited_entities={head_id},
            raw_paths=raw_paths,
            max_hops=max_hops,
            max_branch_per_node=max_branch_per_node,
            max_paths_before_ranking=max_paths_before_ranking,
        )
        if not raw_paths:
            return []

        path_relation_scores, endpoint_scores, length_scores, terminal_triples = [], [], [], []

        for path in raw_paths:
            terminal_entity = path[-1].t_id
            path_relation_scores.append(self._path_relation_consistency(path, relation_id))
            terminal_triples.append((head_id, relation_id, terminal_entity))
            length_scores.append(math.exp(-len(path)))

        endpoint_scores = self.rotate.score_triples(terminal_triples).tolist()

        if self.normalize_scores:
            pr_rank = self._zscore(path_relation_scores)
            ep_rank = self._zscore(endpoint_scores)
            ld_rank = self._zscore(length_scores)
        else:
            pr_rank = path_relation_scores
            ep_rank = endpoint_scores
            ld_rank = length_scores

        evidence_paths = []
        query_head_label = self._entity_label(head_id)
        query_relation_label = self._relation_label(relation_id)

        for path, pr_raw, ep_raw, ld_raw, pr_r, ep_r, ld_r in zip(
            raw_paths, path_relation_scores, endpoint_scores, length_scores, pr_rank, ep_rank, ld_rank
        ):
            terminal_id = path[-1].t_id
            steps = [
                PathStep(
                    h_id=edge.h_id,
                    r_id=edge.r_id,
                    t_id=edge.t_id,
                    h_label=edge.h_label,
                    r_label=edge.r_label,
                    t_label=edge.t_label,
                    h_classname=edge.h_classname,
                    t_classname=edge.t_classname,
                )
                for edge in path
            ]
            text = self._path_to_text(path)
            score = self.alpha_path_relation * pr_r + self.beta_endpoint * ep_r + self.delta_length * ld_r
            evidence_paths.append(
                PathEvidence(
                    path=steps,
                    query_head_id=head_id,
                    query_relation_id=relation_id,
                    terminal_entity_id=terminal_id,
                    query_head_label=query_head_label,
                    query_relation_label=query_relation_label,
                    terminal_entity_label=self._entity_label(terminal_id),
                    path_relation_consistency=float(pr_raw),
                    endpoint_alignment=float(ep_raw),
                    length_decay=float(ld_raw),
                    score=float(score),
                    length=len(path),
                    text=text,
                )
            )
        evidence_paths.sort(key=lambda x: x.score, reverse=True)
        return evidence_paths[:top_k]

    def _dfs_paths(
        self,
        current_entity: int,
        relation_id: int,
        current_path: List[KGEdge],
        visited_entities: Set[int],
        raw_paths: List[List[KGEdge]],
        max_hops: int,
        max_branch_per_node: int,
        max_paths_before_ranking: int,
    ) -> None:

        if len(raw_paths) >= max_paths_before_ranking or len(current_path) >= max_hops:
            return

        edges = self.out_adj.get(current_entity, [])
        if not edges:
            return

        candidate_edges = [
            edge
            for edge in edges
            if not self._is_query_relation_leakage(
                edge,
                query_head_id=current_path[0].h_id if current_path else current_entity,
                query_relation_id=relation_id,
            )
        ]

        candidate_edges.sort(key=lambda e: self._relation_cosine(e.r_id, relation_id), reverse=True)
        candidate_edges = candidate_edges[:max_branch_per_node]

        for edge in candidate_edges:
            if edge.t_id in visited_entities:
                continue
            new_path = current_path + [edge]
            new_visited = set(visited_entities)
            new_visited.add(edge.t_id)
            if len(new_path) >= 2:
                raw_paths.append(new_path)
                if len(raw_paths) >= max_paths_before_ranking:
                    return
            self._dfs_paths(
                current_entity=edge.t_id,
                relation_id=relation_id,
                current_path=new_path,
                visited_entities=new_visited,
                raw_paths=raw_paths,
                max_hops=max_hops,
                max_branch_per_node=max_branch_per_node,
                max_paths_before_ranking=max_paths_before_ranking,
            )

    # --------------------------------------------------------
    # RotatE scoring helpers
    # --------------------------------------------------------

    def _relation_cosine(self, r1_id: int, r2_id: int) -> float:
        if r1_id >= len(self.relation_embedding) or r2_id >= len(self.relation_embedding):
            return 0.0
        v1 = self.relation_embedding[r1_id]
        v2 = self.relation_embedding[r2_id]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / denom) if denom != 0 else 0.0

    def _path_relation_consistency(self, path: List[KGEdge], query_relation_id: int) -> float:
        if query_relation_id >= len(self.relation_embedding):
            return 0.0
        composed = np.zeros_like(self.relation_embedding[query_relation_id])
        for edge in path:
            if edge.r_id < len(self.relation_embedding):
                composed += self.relation_embedding[edge.r_id]
        distance = np.linalg.norm(composed - self.relation_embedding[query_relation_id])
        return float(-distance)

    # --------------------------------------------------------
    # Label helpers
    # --------------------------------------------------------

    def _entity_label(self, entity_id: Optional[int]) -> str:
        if entity_id is None:
            return "None"
        ent = self.entities.get(entity_id)
        return ent.label if ent and ent.label else f"[UnknownEntity:{entity_id}]"

    def _relation_label(self, relation_id: int) -> str:
        rel = self.relations.get(relation_id)
        return rel.label if rel and rel.label else f"[UnknownRelation:{relation_id}]"

    @staticmethod
    def _path_to_text(path: List[KGEdge]) -> str:
        if not path:
            return ""
        parts = [path[0].h_label]
        for edge in path:
            parts.extend([f"--[{edge.r_label}]-->", edge.t_label])
        return " ".join(parts)

    @staticmethod
    def _zscore(values: List[float]) -> List[float]:
        if not values:
            return []
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 1:
            return [0.0]
        mean = arr.mean()
        std = arr.std()
        if std < 1e-12:
            return [0.0 for _ in values]
        return ((arr - mean) / std).tolist()

# ============================================================
# Save / load helpers
# ============================================================

def evidence_to_dict(evidence: ExtractedEvidence) -> Dict[str, Any]:
    return asdict(evidence)

def save_evidence_jsonl(evidence_list: List[ExtractedEvidence], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for evidence in evidence_list:
            f.write(json.dumps(evidence_to_dict(evidence), ensure_ascii=False) + "\n")
    print(f"[EvidenceExtractor] Saved evidence to {output_path}")

# ============================================================
# Debug / visualization helpers
# ============================================================

def inspect_one_evidence(evidence: ExtractedEvidence) -> None:
    print("=" * 120)

    print(
        f"Query: ({evidence.query_head_label}, "
        f"{evidence.query_relation_label}, ?)"
    )

    if evidence.gold_tail_label is not None:
        print(f"Gold tail: {evidence.gold_tail_label}")

    print("\n[One-hop evidence]")

    if not evidence.one_hop:
        print("None")

    for idx, item in enumerate(evidence.one_hop):
        print(
            f"{idx + 1}. "
            f"score={item.score:.4f} | "
            f"rel_sim={item.relation_similarity:.4f} | "
            f"triple={item.triple_reliability:.4f}"
        )

        print(f"    {item.text}")

    print("\n[Path evidence]")

    if not evidence.paths:
        print("None")

    for idx, item in enumerate(evidence.paths):
        print(
            f"{idx + 1}. "
            f"score={item.score:.4f} | "
            f"path_rel={item.path_relation_consistency:.4f} | "
            f"endpoint={item.endpoint_alignment:.4f} | "
            f"length_decay={item.length_decay:.4f}"
        )

        print(f"    {item.text}")

    print("=" * 120)


# ============================================================
# Save helpers
# ============================================================

def evidence_to_dict(evidence: ExtractedEvidence) -> Dict[str, Any]:
    return asdict(evidence)


def save_evidence_jsonl(
    evidence_list: List[ExtractedEvidence],
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for evidence in evidence_list:
            f.write(
                json.dumps(
                    evidence_to_dict(evidence),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[EvidenceExtractor] Saved to: {output_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="RotatE-guided evidence extraction for OD-KGC."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/FB15k-237",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--import_path",
        type=str,
        default="import",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
    )

    parser.add_argument(
        "--max_queries",
        type=int,
        default=-1,
        help="-1 means all queries",
    )

    parser.add_argument(
        "--top_k_one_hop",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top_k_paths",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_hops",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max_branch_per_node",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lambda_relation",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda_triple",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--alpha_path_relation",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--beta_endpoint",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--delta_length",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--use_test_graph",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    data_path = Path(args.data_path)

    dataset_name = (
        args.dataset_name
        if args.dataset_name is not None
        else data_path.name
    )

    max_queries = (
        None
        if args.max_queries == -1
        else args.max_queries
    )

    print("=" * 120)
    print("[OD-KGC] Loading dataset...")
    print("=" * 120)

    loader = KGLoader(data_path)

    dataset = loader.load()

    print("=" * 120)
    print("[OD-KGC] Loading RotatE...")
    print("=" * 120)

    rotate = get_or_train_rotate(
        data_path=str(data_path),
        import_path=str(Path(args.import_path) / "KGE_model"),
        dataset_name=dataset_name,
        load_if_exists=True,
        force_train=False,
        cuda=args.cuda,
        gpu_id=args.gpu_id,
    )

    print("=" * 120)
    print("[OD-KGC] Building EvidenceExtractor...")
    print("=" * 120)

    extractor = EvidenceExtractor(
        dataset=dataset,
        rotate_manager=rotate,
        use_train=True,
        use_valid=True,
        use_test=args.use_test_graph,
        normalize_scores=True,
        lambda_relation=args.lambda_relation,
        lambda_triple=args.lambda_triple,
        alpha_path_relation=args.alpha_path_relation,
        beta_endpoint=args.beta_endpoint,
        delta_length=args.delta_length,
    )

    print("=" * 120)
    print("[OD-KGC] Extracting evidence...")
    print("=" * 120)

    if args.split == "train":
        triples = dataset.train_triples
    elif args.split == "valid":
        triples = dataset.valid_triples
    else:
        triples = dataset.test_triples

    if max_queries is not None:
        triples = triples[:max_queries]

    evidence_list = []

    for idx, tri in enumerate(triples):

        evidence = extractor.extract_for_query(
            head_id=tri.h_id,
            relation_id=tri.r_id,
            gold_tail_id=tri.t_id,
            top_k_one_hop=args.top_k_one_hop,
            top_k_paths=args.top_k_paths,
            max_hops=args.max_hops,
            max_branch_per_node=args.max_branch_per_node,
        )

        evidence_list.append(evidence)

        # print(
        #     f"[{idx + 1}/{len(triples)}] "
        #     f"({tri.h_label}, {tri.r_label}, ?)"
        # )

        # Show first example
        if idx == 0:
            inspect_one_evidence(evidence)

    output_path = (
        Path(args.import_path)
        / "evidence"
        / dataset_name
        / f"{args.split}_evidence.jsonl"
    )

    save_evidence_jsonl(
        evidence_list,
        output_path,
    )

    print("=" * 120)
    print("[OD-KGC] Finished.")
    print("=" * 120)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()