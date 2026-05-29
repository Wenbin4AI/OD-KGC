# src/data/data_loader.py

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


Triple = Tuple[str, str, str]


@dataclass
class KGDataset:
    dataset_name: str
    dataset_dir: Path

    entity2id: Dict[str, int]
    id2entity: Dict[int, str]
    relation2id: Dict[str, int]
    id2relation: Dict[int, str]

    train_triples: List[Triple]
    valid_triples: List[Triple]
    test_triples: List[Triple]

    train_triples_id: List[Tuple[int, int, int]]
    valid_triples_id: List[Tuple[int, int, int]]
    test_triples_id: List[Tuple[int, int, int]]

    entity_info: Dict[str, Any]
    relation_info: Dict[str, Any]
    relation_schema: Dict[str, Any]


class DataLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = self.dataset_dir.name

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.dataset_dir}")

    def load(self) -> KGDataset:
        entity2id = self._load_dict_file("entities.dict")
        relation2id = self._load_dict_file("relations.dict")

        id2entity = {v: k for k, v in entity2id.items()}
        id2relation = {v: k for k, v in relation2id.items()}

        train_triples = self._load_triple_text("train.txt")
        valid_triples = self._load_triple_text("valid.txt")
        test_triples = self._load_triple_text("test.txt")

        train_triples_id = self._load_triple_id("train2id.txt")
        valid_triples_id = self._load_triple_id("valid2id.txt")
        test_triples_id = self._load_triple_id("test2id.txt")

        entity_info = self._load_json_optional("entity.json")
        relation_info = self._load_json_optional("relation.json")
        relation_schema = self._load_json_optional("relation_new.json")

        return KGDataset(
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
            entity2id=entity2id,
            id2entity=id2entity,
            relation2id=relation2id,
            id2relation=id2relation,
            train_triples=train_triples,
            valid_triples=valid_triples,
            test_triples=test_triples,
            train_triples_id=train_triples_id,
            valid_triples_id=valid_triples_id,
            test_triples_id=test_triples_id,
            entity_info=entity_info,
            relation_info=relation_info,
            relation_schema=relation_schema,
        )

    def _load_dict_file(self, filename: str) -> Dict[str, int]:
        path = self.dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                # 常见格式：entity id
                key = parts[0]
                idx = int(parts[1])
                mapping[key] = idx

        return mapping

    def _load_triple_text(self, filename: str) -> List[Triple]:
        path = self.dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        triples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))

        return triples

    def _load_triple_id(self, filename: str) -> List[Tuple[int, int, int]]:
        path = self.dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        triples = []
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 有些 train2id 第一行是三元组数量，需要跳过
        start_idx = 1 if len(lines[0].split()) == 1 else 0

        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) < 3:
                continue

            # 注意：很多 KGE 格式是 h t r
            h = int(parts[0])
            t = int(parts[1])
            r = int(parts[2])
            triples.append((h, r, t))

        return triples

    def _load_json_optional(self, filename: str) -> Dict[str, Any]:
        path = self.dataset_dir / filename
        if not path.exists():
            print(f"[Warning] Optional file not found: {path}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def inspect_dataset(dataset: KGDataset, num_examples: int = 5):
    print("=" * 80)
    print(f"Dataset: {dataset.dataset_name}")
    print(f"Path: {dataset.dataset_dir}")
    print("=" * 80)

    print(f"# Entities: {len(dataset.entity2id)}")
    print(f"# Relations: {len(dataset.relation2id)}")
    print(f"# Train triples: {len(dataset.train_triples)}")
    print(f"# Valid triples: {len(dataset.valid_triples)}")
    print(f"# Test triples: {len(dataset.test_triples)}")
    print(f"# Train triples id: {len(dataset.train_triples_id)}")
    print(f"# Valid triples id: {len(dataset.valid_triples_id)}")
    print(f"# Test triples id: {len(dataset.test_triples_id)}")

    print("\n[Entity examples]")
    for item in list(dataset.entity2id.items())[:num_examples]:
        print(item)

    print("\n[Relation examples]")
    for item in list(dataset.relation2id.items())[:num_examples]:
        print(item)

    print("\n[Train triple examples]")
    for triple in dataset.train_triples[:num_examples]:
        print(triple)

    print("\n[Train triple id examples]")
    for triple in dataset.train_triples_id[:num_examples]:
        print(triple)

    print("\n[entity.json keys]")
    print(list(dataset.entity_info.keys())[:num_examples])

    print("\n[relation.json keys]")
    print(list(dataset.relation_info.keys())[:num_examples])

    print("\n[relation_new.json keys]")
    print(list(dataset.relation_schema.keys())[:num_examples])


if __name__ == "__main__":
    # 按你的新结构可以改成 dataset/WN18RR 或 dataset/FB15k-237
    dataset_path = "data/WN18RR"

    loader = DataLoader(dataset_path)
    dataset = loader.load()
    inspect_dataset(dataset)