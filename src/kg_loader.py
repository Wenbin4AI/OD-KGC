# OD-KGC/src/kg_loader.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Entity:
    id: int
    label: str
    value: Any = None
    freebase_id: Optional[str] = None
    classname: Optional[str] = None
    class_id: Any = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class Relation:
    id: int
    label: str
    freebase: Optional[str] = None
    domain: Any = None
    range: Any = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class Triple:
    h_id: int
    r_id: int
    t_id: int

    h_label: str
    r_label: str
    t_label: str

    h_classname: Optional[str] = None
    t_classname: Optional[str] = None


@dataclass
class KGDataset:
    dataset_name: str
    dataset_dir: Path

    entities: Dict[int, Entity]
    relations: Dict[int, Relation]

    entity_label2id: Dict[str, int]
    entity_freebase2id: Dict[str, int]
    relation_label2id: Dict[str, int]
    relation_freebase2id: Dict[str, int]

    train_triples: List[Triple]
    valid_triples: List[Triple]
    test_triples: List[Triple]


class KGLoader:
    """
    Loader for OD-KGC data.

    Expected files:
        entity.json
        relation.json
        relation_new.json  optional
        train2id.txt       optional
        valid2id.txt       optional
        test2id.txt        required for testing

    Notes:
        - entity.json may use "value" as entity id.
        - relation.json usually uses "id" as relation id.
        - *_2id.txt is usually OpenKE format: head_id tail_id relation_id.
        - The loader automatically detects whether triples are h-t-r or h-r-t.
    """

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = self.dataset_dir.name

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

    def load(self) -> KGDataset:
        entities = self._load_entities()
        relations = self._load_relations()

        train_triples = self._load_split_if_exists(
            "train2id.txt", entities, relations
        )
        valid_triples = self._load_split_if_exists(
            "valid2id.txt", entities, relations
        )
        test_triples = self._load_split_if_exists(
            "test2id.txt", entities, relations
        )

        return KGDataset(
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
            entities=entities,
            relations=relations,
            entity_label2id=self._build_entity_label2id(entities),
            entity_freebase2id=self._build_entity_freebase2id(entities),
            relation_label2id=self._build_relation_label2id(relations),
            relation_freebase2id=self._build_relation_freebase2id(relations),
            train_triples=train_triples,
            valid_triples=valid_triples,
            test_triples=test_triples,
        )

    # ------------------------------------------------------------------
    # Entity loading
    # ------------------------------------------------------------------

    def _load_entities(self) -> Dict[int, Entity]:
        path = self.dataset_dir / "entity.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing entity file: {path}")

        raw_items = self._load_json_items(path)

        entities: Dict[int, Entity] = {}

        for outer_key, item in raw_items:
            if not isinstance(item, dict):
                continue

            ent_id = self._extract_entity_id(item, outer_key)
            if ent_id is None:
                raise ValueError(
                    f"Cannot find entity id in entity.json item: {item}"
                )

            label = (
                item.get("label")
                or item.get("name")
                or item.get("freebase_id")
                or item.get("wordnet_id")
                or str(ent_id)
            )

            freebase_id = (
                item.get("freebase_id")
                or item.get("freebase")
                or item.get("fb_id")
                or item.get("wn_id")
                or item.get("wordnet_id")
            )

            class_id = (
                item.get("classid")
                if "classid" in item
                else item.get("class_id")
                if "class_id" in item
                else item.get("class")
            )

            entity = Entity(
                id=ent_id,
                value=item.get("value"),
                label=str(label),
                freebase_id=str(freebase_id) if freebase_id is not None else None,
                classname=item.get("classname") or item.get("class_name"),
                class_id=class_id,
                raw=item,
            )

            entities[ent_id] = entity

        return entities

    def _extract_entity_id(
        self, item: Dict[str, Any], outer_key: Optional[str]
    ) -> Optional[int]:
        """
        OD-KGC entity.json may not have "id".
        In FB15k-237, the entity id is stored in "value".
        """

        for key in ["id", "entity_id", "idx", "index", "value"]:
            if key in item and self._is_int_like(item[key]):
                return int(item[key])

        if outer_key is not None and self._is_int_like(outer_key):
            return int(outer_key)

        return None

    # ------------------------------------------------------------------
    # Relation loading
    # ------------------------------------------------------------------

    def _load_relations(self) -> Dict[int, Relation]:
        path = self.dataset_dir / "relation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing relation file: {path}")

        raw_items = self._load_json_items(path)

        relations: Dict[int, Relation] = {}

        for outer_key, item in raw_items:
            if not isinstance(item, dict):
                continue

            rel_id = self._extract_relation_id(item, outer_key)
            if rel_id is None:
                raise ValueError(
                    f"Cannot find relation id in relation.json item: {item}"
                )

            label = (
                item.get("label")
                or item.get("name")
                or item.get("freebase")
                or item.get("freebase_id")
                or str(rel_id)
            )

            freebase = item.get("freebase") or item.get("freebase_id")

            relation = Relation(
                id=rel_id,
                label=str(label),
                freebase=str(freebase) if freebase is not None else None,
                domain=item.get("domain"),
                range=item.get("range"),
                raw=item,
            )

            relations[rel_id] = relation

        self._merge_relation_schema(relations)

        return relations

    def _extract_relation_id(
        self, item: Dict[str, Any], outer_key: Optional[str]
    ) -> Optional[int]:
        for key in ["id", "relation_id", "idx", "index", "value"]:
            if key in item and self._is_int_like(item[key]):
                return int(item[key])

        if outer_key is not None and self._is_int_like(outer_key):
            return int(outer_key)

        return None

    def _merge_relation_schema(self, relations: Dict[int, Relation]) -> None:
        """
        Merge domain/range information from relation_new.json if available.
        This function is deliberately tolerant because relation_new.json may
        use relation id, label, or freebase string as key.
        """

        schema_path = self.dataset_dir / "relation_new.json"
        if not schema_path.exists():
            return

        try:
            schema_items = self._load_json_items(schema_path)
        except Exception as e:
            print(f"[Warning] Failed to load relation_new.json: {e}")
            return

        label2id = {
            rel.label: rel_id for rel_id, rel in relations.items()
            if rel.label is not None
        }
        freebase2id = {
            rel.freebase: rel_id for rel_id, rel in relations.items()
            if rel.freebase is not None
        }

        for outer_key, item in schema_items:
            if not isinstance(item, dict):
                continue

            rel_id = self._extract_relation_id(item, outer_key)

            if rel_id is None:
                possible_names = [
                    item.get("label"),
                    item.get("relation"),
                    item.get("relation_label"),
                    item.get("freebase"),
                    item.get("freebase_id"),
                    outer_key,
                ]

                for name in possible_names:
                    if name in label2id:
                        rel_id = label2id[name]
                        break
                    if name in freebase2id:
                        rel_id = freebase2id[name]
                        break

            if rel_id is None or rel_id not in relations:
                continue

            rel = relations[rel_id]

            if rel.domain is None:
                rel.domain = (
                    item.get("domain")
                    or item.get("domains")
                    or item.get("head_type")
                    or item.get("head_class")
                )

            if rel.range is None:
                rel.range = (
                    item.get("range")
                    or item.get("ranges")
                    or item.get("tail_type")
                    or item.get("tail_class")
                )

    # ------------------------------------------------------------------
    # Triple loading
    # ------------------------------------------------------------------

    def _load_split_if_exists(
        self,
        filename: str,
        entities: Dict[int, Entity],
        relations: Dict[int, Relation],
    ) -> List[Triple]:
        path = self.dataset_dir / filename
        if not path.exists():
            print(f"[Warning] {filename} not found, skip it.")
            return []

        raw_triples = self._read_id_triples(path)
        triple_order = self._detect_triple_order(
            raw_triples, entities, relations
        )

        triples: List[Triple] = []

        for a, b, c in raw_triples:
            if triple_order == "htr":
                h_id, t_id, r_id = a, b, c
            else:
                h_id, r_id, t_id = a, b, c

            h_ent = entities.get(h_id)
            t_ent = entities.get(t_id)
            rel = relations.get(r_id)

            h_label = h_ent.label if h_ent else f"[UnknownEntity:{h_id}]"
            t_label = t_ent.label if t_ent else f"[UnknownEntity:{t_id}]"
            r_label = rel.label if rel else f"[UnknownRelation:{r_id}]"

            triples.append(
                Triple(
                    h_id=h_id,
                    r_id=r_id,
                    t_id=t_id,
                    h_label=h_label,
                    r_label=r_label,
                    t_label=t_label,
                    h_classname=h_ent.classname if h_ent else None,
                    t_classname=t_ent.classname if t_ent else None,
                )
            )

        print(
            f"[Loaded] {filename}: {len(triples)} triples, "
            f"detected order = {triple_order}"
        )

        return triples

    def _read_id_triples(self, path: Path) -> List[Tuple[int, int, int]]:
        triples: List[Tuple[int, int, int]] = []

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return triples

        # Some OpenKE files have the number of triples in the first line.
        start_idx = 1 if len(lines[0].split()) == 1 else 0

        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue

            triples.append((a, b, c))

        return triples

    def _detect_triple_order(
        self,
        triples: List[Tuple[int, int, int]],
        entities: Dict[int, Entity],
        relations: Dict[int, Relation],
        max_check: int = 1000,
    ) -> str:
        """
        Detect whether triple file is:
            h t r  -> "htr"
        or:
            h r t  -> "hrt"

        Most KGE/OpenKE files are h t r.
        """

        if not triples:
            return "htr"

        sample = triples[:max_check]

        htr_score = 0
        hrt_score = 0

        for a, b, c in sample:
            # h t r
            if a in entities and b in entities and c in relations:
                htr_score += 1

            # h r t
            if a in entities and b in relations and c in entities:
                hrt_score += 1

        if htr_score >= hrt_score:
            return "htr"
        return "hrt"

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _load_json_items(
        self, path: Path
    ) -> List[Tuple[Optional[str], Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return [(None, item) for item in data]

        if isinstance(data, dict):
            return [(str(k), v) for k, v in data.items()]

        raise ValueError(f"Unsupported JSON format in {path}")

    @staticmethod
    def _is_int_like(x: Any) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    @staticmethod
    def _build_entity_label2id(
        entities: Dict[int, Entity]
    ) -> Dict[str, int]:
        return {
            ent.label: ent_id
            for ent_id, ent in entities.items()
            if ent.label is not None
        }

    @staticmethod
    def _build_entity_freebase2id(
        entities: Dict[int, Entity]
    ) -> Dict[str, int]:
        return {
            ent.freebase_id: ent_id
            for ent_id, ent in entities.items()
            if ent.freebase_id is not None
        }

    @staticmethod
    def _build_relation_label2id(
        relations: Dict[int, Relation]
    ) -> Dict[str, int]:
        return {
            rel.label: rel_id
            for rel_id, rel in relations.items()
            if rel.label is not None
        }

    @staticmethod
    def _build_relation_freebase2id(
        relations: Dict[int, Relation]
    ) -> Dict[str, int]:
        return {
            rel.freebase: rel_id
            for rel_id, rel in relations.items()
            if rel.freebase is not None
        }


def inspect_dataset(dataset: KGDataset, num_examples: int = 5) -> None:
    print("\n" + "=" * 90)
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Path: {dataset.dataset_dir}")
    print("=" * 90)

    print(f"# Entities: {len(dataset.entities)}")
    print(f"# Relations: {len(dataset.relations)}")
    print(f"# Train triples: {len(dataset.train_triples)}")
    print(f"# Valid triples: {len(dataset.valid_triples)}")
    print(f"# Test triples: {len(dataset.test_triples)}")

    print("\n[Entity examples]")
    for ent_id, ent in list(dataset.entities.items())[:num_examples]:
        print(
            f"id={ent_id}, label={ent.label}, "
            f"freebase_id={ent.freebase_id}, "
            f"classname={ent.classname}, class_id={ent.class_id}"
        )

    print("\n[Relation examples]")
    for rel_id, rel in list(dataset.relations.items())[:num_examples]:
        print(
            f"id={rel_id}, label={rel.label}, "
            f"freebase={rel.freebase}, "
            f"domain={rel.domain}, range={rel.range}"
        )

    print("\n[Test triple examples]")
    for tri in dataset.test_triples[:num_examples]:
        print(
            f"({tri.h_id}, {tri.r_id}, {tri.t_id})  "
            f"{tri.h_label} --[{tri.r_label}]--> {tri.t_label}  "
            f"classes=({tri.h_classname}, {tri.t_classname})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=Path("dataset/WN18RR"),
        help="Dataset directory, e.g., data/WN18RR or dataset/FB15k-237",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to print.",
    )

    args = parser.parse_args()

    loader = KGLoader(args.dataset)
    dataset = loader.load()
    inspect_dataset(dataset, num_examples=args.num_examples)


if __name__ == "__main__":
    main()