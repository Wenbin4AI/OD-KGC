# OD-KGC/model/KGE_model.py

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# Make sure this file can import src/kg_loader.py when executed directly
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader


TripleID = Tuple[int, int, int]  # (head_id, relation_id, tail_id)


# ======================================================================
# Basic utilities
# ======================================================================

def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_file: Path) -> None:
    ensure_dir(log_file.parent)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ======================================================================
# Config
# ======================================================================

@dataclass
class RotatEConfig:
    data_path: str = "data/WN18RR"
    import_path: str = "import/KGE_model"
    dataset_name: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    hidden_dim: int = 1000
    gamma: float = 24.0

    negative_sample_size: int = 256
    batch_size: int = 1024
    test_batch_size: int = 16

    learning_rate: float = 1e-4
    max_steps: int = 150000
    warm_up_steps: Optional[int] = None

    negative_adversarial_sampling: bool = True
    adversarial_temperature: float = 1.0
    regularization: float = 0.0
    uni_weight: bool = False

    cuda: bool = True
    gpu_id: int = 0
    amp: bool = False

    cpu_num: int = 8
    seed: int = 2026

    log_steps: int = 100
    valid_steps: int = 10000
    save_checkpoint_steps: int = 10000
    test_log_steps: int = 1000

    nentity: int = 0
    nrelation: int = 0


# ======================================================================
# Data wrapper based on src/kg_loader.py
# ======================================================================

class KGEData:
    """
    This class reuses src/kg_loader.py.

    kg_loader.py is responsible for reading:
        entity.json
        relation.json
        train2id.txt
        valid2id.txt
        test2id.txt

    This wrapper only converts loaded triples into RotatE format:
        (head_id, relation_id, tail_id)
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_path}")

        loader = KGLoader(self.data_path)
        self.dataset = loader.load()

        self.train_triples: List[TripleID] = self._convert_triples(
            self.dataset.train_triples
        )
        self.valid_triples: List[TripleID] = self._convert_triples(
            self.dataset.valid_triples
        )
        self.test_triples: List[TripleID] = self._convert_triples(
            self.dataset.test_triples
        )

        self.nentity = self._infer_nentity()
        self.nrelation = self._infer_nrelation()

        self.all_true_triples = (
            self.train_triples + self.valid_triples + self.test_triples
        )

    @staticmethod
    def _convert_triples(triples: List[Any]) -> List[TripleID]:
        converted = []

        for tri in triples:
            converted.append((tri.h_id, tri.r_id, tri.t_id))

        return converted

    def _infer_nentity(self) -> int:
        if hasattr(self.dataset, "entities") and self.dataset.entities:
            return max(self.dataset.entities.keys()) + 1

        max_id = -1
        for h, _, t in self.train_triples + self.valid_triples + self.test_triples:
            max_id = max(max_id, h, t)

        if max_id < 0:
            raise ValueError("Cannot infer number of entities.")

        return max_id + 1

    def _infer_nrelation(self) -> int:
        if hasattr(self.dataset, "relations") and self.dataset.relations:
            return max(self.dataset.relations.keys()) + 1

        max_id = -1
        for _, r, _ in self.train_triples + self.valid_triples + self.test_triples:
            max_id = max(max_id, r)

        if max_id < 0:
            raise ValueError("Cannot infer number of relations.")

        return max_id + 1


# ======================================================================
# Training dataset
# ======================================================================

class TrainDataset(Dataset):
    def __init__(
        self,
        triples: List[TripleID],
        nentity: int,
        negative_sample_size: int,
        mode: str,
    ):
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.mode = mode

        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[
            (tail, -relation - 1)
        ]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(
                self.nentity,
                size=self.negative_sample_size * 2,
            )

            if self.mode == "head-batch":
                mask = np.isin(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True,
                )
            elif self.mode == "tail-batch":
                mask = np.isin(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True,
                )
            else:
                raise ValueError(f"Unsupported training mode: {self.mode}")

            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[
            : self.negative_sample_size
        ]

        return (
            torch.LongTensor(positive_sample),
            torch.LongTensor(negative_sample),
            subsampling_weight,
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([item[0] for item in data], dim=0)
        negative_sample = torch.stack([item[1] for item in data], dim=0)
        subsampling_weight = torch.cat([item[2] for item in data], dim=0)
        mode = data[0][3]

        return positive_sample, negative_sample, subsampling_weight, mode

    @staticmethod
    def count_frequency(
        triples: List[TripleID],
        start: int = 4,
    ) -> Dict[Tuple[int, int], int]:
        count = {}

        for head, relation, tail in triples:
            count[(head, relation)] = count.get((head, relation), start) + 1
            count[(tail, -relation - 1)] = count.get(
                (tail, -relation - 1), start
            ) + 1

        return count

    @staticmethod
    def get_true_head_and_tail(triples: List[TripleID]):
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            true_tail.setdefault((head, relation), []).append(tail)
            true_head.setdefault((relation, tail), []).append(head)

        true_head = {
            key: np.array(list(set(value)))
            for key, value in true_head.items()
        }

        true_tail = {
            key: np.array(list(set(value)))
            for key, value in true_tail.items()
        }

        return true_head, true_tail


# ======================================================================
# Test dataset
# ======================================================================

class TestDataset(Dataset):
    def __init__(
        self,
        triples: List[TripleID],
        all_true_triples: List[TripleID],
        nentity: int,
        mode: str,
    ):
        self.triples = triples
        self.triple_set = set(all_true_triples)
        self.nentity = nentity
        self.mode = mode

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        head, relation, tail = self.triples[idx]

        if self.mode == "head-batch":
            tmp = [
                (-1000000.0, rand_head)
                if (rand_head, relation, tail) in self.triple_set
                else (0.0, rand_head)
                for rand_head in range(self.nentity)
            ]
            tmp[head] = (0.0, head)

        elif self.mode == "tail-batch":
            tmp = [
                (-1000000.0, rand_tail)
                if (head, relation, rand_tail) in self.triple_set
                else (0.0, rand_tail)
                for rand_tail in range(self.nentity)
            ]
            tmp[tail] = (0.0, tail)

        else:
            raise ValueError(f"Unsupported test mode: {self.mode}")

        tmp = torch.LongTensor([[int(x[0]), int(x[1])] for x in tmp])

        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([item[0] for item in data], dim=0)
        negative_sample = torch.stack([item[1] for item in data], dim=0)
        filter_bias = torch.stack([item[2] for item in data], dim=0)
        mode = data[0][3]

        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator:
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1

        if self.step % 2 == 0:
            return next(self.iterator_head)

        return next(self.iterator_tail)

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


# ======================================================================
# RotatE model
# ======================================================================

class RotatEModel(nn.Module):
    def __init__(
        self,
        nentity: int,
        nrelation: int,
        hidden_dim: int,
        gamma: float,
    ):
        super().__init__()

        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim

        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False,
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False,
        )

        self.entity_dim = hidden_dim * 2
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(
            torch.zeros(nentity, self.entity_dim)
        )
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim)
        )
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

    def forward(self, sample, mode: str = "single"):
        if mode == "single":
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0],
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1],
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2],
            ).unsqueeze(1)

        elif mode == "head-batch":
            positive_sample, negative_sample = sample
            batch_size, negative_sample_size = negative_sample.size()

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=negative_sample.view(-1),
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=positive_sample[:, 1],
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_sample[:, 2],
            ).unsqueeze(1)

        elif mode == "tail-batch":
            positive_sample, negative_sample = sample
            batch_size, negative_sample_size = negative_sample.size()

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_sample[:, 0],
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=positive_sample[:, 1],
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=negative_sample.view(-1),
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return self.rotate_score(head, relation, tail, mode)

    def rotate_score(self, head, relation, tail, mode: str):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == "head-batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        return self.gamma.item() - score.sum(dim=2)


# ======================================================================
# RotatE manager
# ======================================================================

class RotatEManager:
    def __init__(self, config: RotatEConfig):
        self.config = config

        if self.config.dataset_name is None:
            self.config.dataset_name = Path(self.config.data_path).name

        if self.config.checkpoint_dir is None:
            self.checkpoint_dir = (
                Path(self.config.import_path) / str(self.config.dataset_name)
            )
        else:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)

        ensure_dir(self.checkpoint_dir)
        setup_logger(self.checkpoint_dir / "KGE_model.log")
        set_seed(self.config.seed)

        if self.config.cuda and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.use_amp = (
            self.config.amp
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            logging.info("Using GPU: %s", torch.cuda.get_device_name(self.device))
        else:
            logging.info("Using CPU.")

        self.data = KGEData(self.config.data_path)

        self.config.nentity = self.data.nentity
        self.config.nrelation = self.data.nrelation

        self.model = RotatEModel(
            nentity=self.config.nentity,
            nrelation=self.config.nrelation,
            hidden_dim=self.config.hidden_dim,
            gamma=self.config.gamma,
        ).to(self.device)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        logging.info("Dataset: %s", self.config.dataset_name)
        logging.info("Data path: %s", self.config.data_path)
        logging.info("Checkpoint dir: %s", self.checkpoint_dir)
        logging.info("#Entity: %d", self.config.nentity)
        logging.info("#Relation: %d", self.config.nrelation)
        logging.info("#Train triples: %d", len(self.data.train_triples))
        logging.info("#Valid triples: %d", len(self.data.valid_triples))
        logging.info("#Test triples: %d", len(self.data.test_triples))
        logging.info("AMP enabled: %s", self.use_amp)

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint.pt"

    def has_checkpoint(self) -> bool:
        return self.checkpoint_path.exists()

    def train(self, resume: bool = False):
        if not self.data.train_triples:
            raise ValueError("No train triples found. Please check train2id.txt.")

        pin_memory = self.device.type == "cuda"

        train_dataloader_head = DataLoader(
            TrainDataset(
                triples=self.data.train_triples,
                nentity=self.config.nentity,
                negative_sample_size=self.config.negative_sample_size,
                mode="head-batch",
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.cpu_num,
            collate_fn=TrainDataset.collate_fn,
            pin_memory=pin_memory,
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(
                triples=self.data.train_triples,
                nentity=self.config.nentity,
                negative_sample_size=self.config.negative_sample_size,
                mode="tail-batch",
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.cpu_num,
            collate_fn=TrainDataset.collate_fn,
            pin_memory=pin_memory,
        )

        train_iterator = BidirectionalOneShotIterator(
            train_dataloader_head,
            train_dataloader_tail,
        )

        current_learning_rate = self.config.learning_rate

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=current_learning_rate,
        )

        warm_up_steps = (
            self.config.warm_up_steps
            if self.config.warm_up_steps is not None
            else self.config.max_steps // 2
        )

        init_step = 0

        if resume and self.has_checkpoint():
            state = self.load_checkpoint(load_optimizer=True)
            init_step = int(state.get("step", 0))
            current_learning_rate = float(
                state.get("current_learning_rate", current_learning_rate)
            )
            warm_up_steps = int(state.get("warm_up_steps", warm_up_steps))

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_learning_rate

            logging.info("Resume training from step %d", init_step)

        logging.info("Start RotatE training.")
        logging.info("Max steps: %d", self.config.max_steps)
        logging.info("Batch size: %d", self.config.batch_size)
        logging.info("Negative sample size: %d", self.config.negative_sample_size)
        logging.info("Learning rate: %.8f", current_learning_rate)

        training_logs = []

        for step in range(init_step, self.config.max_steps):
            log = self.train_step(train_iterator)
            training_logs.append(log)

            if step % self.config.log_steps == 0:
                metrics = self.average_logs(training_logs)
                training_logs = []

                logging.info(
                    "Step %d | loss %.6f | pos %.6f | neg %.6f",
                    step,
                    metrics["loss"],
                    metrics["positive_sample_loss"],
                    metrics["negative_sample_loss"],
                )

            if (
                self.config.valid_steps > 0
                and step > 0
                and step % self.config.valid_steps == 0
                and self.data.valid_triples
            ):
                metrics = self.evaluate(split="valid")
                self.log_metrics("Valid", step, metrics)

            if (
                self.config.save_checkpoint_steps > 0
                and step > 0
                and step % self.config.save_checkpoint_steps == 0
            ):
                self.save_checkpoint(
                    step=step,
                    current_learning_rate=current_learning_rate,
                    warm_up_steps=warm_up_steps,
                )

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_learning_rate

                logging.info(
                    "Change learning rate to %.8f at step %d",
                    current_learning_rate,
                    step,
                )

                warm_up_steps = warm_up_steps * 3

        self.save_checkpoint(
            step=self.config.max_steps,
            current_learning_rate=current_learning_rate,
            warm_up_steps=warm_up_steps,
        )

        logging.info("Training finished.")

        return self

    def train_step(self, train_iterator) -> Dict[str, float]:
        self.model.train()

        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")

        self.optimizer.zero_grad(set_to_none=True)

        positive_sample, negative_sample, subsampling_weight, mode = next(
            train_iterator
        )

        positive_sample = positive_sample.to(self.device, non_blocking=True)
        negative_sample = negative_sample.to(self.device, non_blocking=True)
        subsampling_weight = subsampling_weight.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            negative_score = self.model(
                (positive_sample, negative_sample),
                mode=mode,
            )

            if self.config.negative_adversarial_sampling:
                negative_score = (
                    F.softmax(
                        negative_score * self.config.adversarial_temperature,
                        dim=1,
                    ).detach()
                    * F.logsigmoid(-negative_score)
                ).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)

            positive_score = self.model(positive_sample, mode="single")
            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            if self.config.uni_weight:
                positive_sample_loss = -positive_score.mean()
                negative_sample_loss = -negative_score.mean()
            else:
                positive_sample_loss = -(
                    subsampling_weight * positive_score
                ).sum() / subsampling_weight.sum()

                negative_sample_loss = -(
                    subsampling_weight * negative_score
                ).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2

            if self.config.regularization != 0.0:
                regularization = self.config.regularization * (
                    self.model.entity_embedding.norm(p=3) ** 3
                    + self.model.relation_embedding.norm(p=3) ** 3
                )
                loss = loss + regularization

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {
            "positive_sample_loss": float(positive_sample_loss.item()),
            "negative_sample_loss": float(negative_sample_loss.item()),
            "loss": float(loss.item()),
        }

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        if split == "train":
            triples = self.data.train_triples
        elif split == "valid":
            triples = self.data.valid_triples
        elif split == "test":
            triples = self.data.test_triples
        else:
            raise ValueError("split must be train, valid, or test.")

        if not triples:
            logging.warning("No triples found for split=%s", split)
            return {}

        self.model.eval()

        pin_memory = self.device.type == "cuda"

        dataloader_head = DataLoader(
            TestDataset(
                triples=triples,
                all_true_triples=self.data.all_true_triples,
                nentity=self.config.nentity,
                mode="head-batch",
            ),
            batch_size=self.config.test_batch_size,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
            pin_memory=pin_memory,
        )

        dataloader_tail = DataLoader(
            TestDataset(
                triples=triples,
                all_true_triples=self.data.all_true_triples,
                nentity=self.config.nentity,
                mode="tail-batch",
            ),
            batch_size=self.config.test_batch_size,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
            pin_memory=pin_memory,
        )

        logs = []
        total_steps = len(dataloader_head) + len(dataloader_tail)
        step = 0

        with torch.no_grad():
            for dataloader in [dataloader_head, dataloader_tail]:
                for positive_sample, negative_sample, filter_bias, mode in dataloader:
                    positive_sample = positive_sample.to(
                        self.device, non_blocking=True
                    )
                    negative_sample = negative_sample.to(
                        self.device, non_blocking=True
                    )
                    filter_bias = filter_bias.to(self.device, non_blocking=True)

                    score = self.model(
                        (positive_sample, negative_sample),
                        mode=mode,
                    )

                    score += filter_bias

                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == "head-batch":
                        positive_arg = positive_sample[:, 0]
                    else:
                        positive_arg = positive_sample[:, 2]

                    batch_size = positive_sample.size(0)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()

                        if ranking.numel() == 0:
                            continue

                        ranking = 1 + ranking.item()

                        logs.append(
                            {
                                "MRR": 1.0 / ranking,
                                "MR": float(ranking),
                                "HITS@1": 1.0 if ranking <= 1 else 0.0,
                                "HITS@3": 1.0 if ranking <= 3 else 0.0,
                                "HITS@10": 1.0 if ranking <= 10 else 0.0,
                            }
                        )

                    if step % self.config.test_log_steps == 0:
                        logging.info(
                            "Evaluating %s... %d/%d",
                            split,
                            step,
                            total_steps,
                        )

                    step += 1

        if not logs:
            return {}

        metrics = {}
        for key in logs[0].keys():
            metrics[key] = sum(log[key] for log in logs) / len(logs)

        return metrics

    def save_checkpoint(
        self,
        step: int,
        current_learning_rate: float,
        warm_up_steps: int,
    ) -> None:
        ensure_dir(self.checkpoint_dir)

        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

        state = {
            "step": step,
            "current_learning_rate": current_learning_rate,
            "warm_up_steps": warm_up_steps,
            "model_state_dict": self.model.state_dict(),
        }

        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(state, self.checkpoint_path)

        entity_embedding = self.model.entity_embedding.detach().cpu().numpy()
        relation_embedding = self.model.relation_embedding.detach().cpu().numpy()

        np.save(self.checkpoint_dir / "entity_embedding.npy", entity_embedding)
        np.save(self.checkpoint_dir / "relation_embedding.npy", relation_embedding)

        logging.info("Checkpoint saved to %s", self.checkpoint_dir)

    def load_checkpoint(self, load_optimizer: bool = False) -> Dict[str, Any]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        state = torch.load(self.checkpoint_path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])

        if load_optimizer:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.config.learning_rate,
                )

            if "optimizer_state_dict" in state:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])

        logging.info("Checkpoint loaded from %s", self.checkpoint_path)

        return state

    def score_triples(
        self,
        triples: List[TripleID],
        batch_size: int = 4096,
    ) -> np.ndarray:
        """
        Score triples in the format:
            (head_id, relation_id, tail_id)

        Higher score means more plausible.
        """

        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for start in range(0, len(triples), batch_size):
                batch = triples[start : start + batch_size]
                batch_tensor = torch.LongTensor(batch).to(self.device)

                score = self.model(batch_tensor, mode="single").squeeze(dim=1)
                all_scores.append(score.detach().cpu())

        return torch.cat(all_scores, dim=0).numpy()

    def score_tail_candidates(
        self,
        head_id: int,
        relation_id: int,
        candidate_tail_ids: Optional[List[int]] = None,
        batch_size: int = 4096,
    ) -> List[Tuple[int, float]]:
        """
        Score candidates for query:
            (head_id, relation_id, ?)

        Return:
            [(tail_id, score), ...] sorted by score descending.
        """

        if candidate_tail_ids is None:
            candidate_tail_ids = list(range(self.config.nentity))

        triples = [
            (head_id, relation_id, tail_id)
            for tail_id in candidate_tail_ids
        ]

        scores = self.score_triples(triples, batch_size=batch_size)

        results = list(zip(candidate_tail_ids, scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def get_entity_embedding(self) -> np.ndarray:
        return self.model.entity_embedding.detach().cpu().numpy()

    def get_relation_embedding(self) -> np.ndarray:
        return self.model.relation_embedding.detach().cpu().numpy()

    @staticmethod
    def average_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
        if not logs:
            return {}

        metrics = {}
        for key in logs[0].keys():
            metrics[key] = sum(log[key] for log in logs) / len(logs)

        return metrics

    @staticmethod
    def log_metrics(mode: str, step: int, metrics: Dict[str, float]) -> None:
        if not metrics:
            logging.info("%s at step %d: no metrics.", mode, step)
            return

        for key, value in metrics.items():
            logging.info("%s %s at step %d: %.6f", mode, key, step, value)


# ======================================================================
# Public APIs for other modules
# ======================================================================

def get_or_train_rotate(
    data_path: str = "data/WN18RR",
    import_path: str = "import/KGE_model",
    dataset_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    load_if_exists: bool = True,
    force_train: bool = False,
    resume: bool = False,
    **kwargs,
) -> RotatEManager:
    """
    Recommended API for other modules.

    Example:
        from model.KGE_model import get_or_train_rotate

        rotate = get_or_train_rotate(
            data_path="data/WN18RR",
            import_path="import/KGE_model",
            dataset_name="WN18RR",
            load_if_exists=True,
            force_train=False,
            cuda=True,
            gpu_id=0,
        )
    """

    config = RotatEConfig(
        data_path=data_path,
        import_path=import_path,
        dataset_name=dataset_name,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )

    manager = RotatEManager(config)

    if force_train:
        manager.train(resume=False)
        return manager

    if load_if_exists and manager.has_checkpoint():
        manager.load_checkpoint(load_optimizer=False)
        return manager

    manager.train(resume=resume)

    return manager


def load_trained_rotate(
    data_path: str = "data/WN18RR",
    import_path: str = "import/KGE_model",
    dataset_name: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    **kwargs,
) -> RotatEManager:
    """
    Load an existing trained RotatE model only.
    If checkpoint does not exist, it raises FileNotFoundError.
    """

    config = RotatEConfig(
        data_path=data_path,
        import_path=import_path,
        dataset_name=dataset_name,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )

    manager = RotatEManager(config)
    manager.load_checkpoint(load_optimizer=False)

    return manager


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RotatE training/loading module for OD-KGC."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["auto", "train", "resume", "load", "test"],
        help=(
            "auto: load checkpoint if exists, otherwise train; "
            "train: train from scratch; "
            "resume: resume training from checkpoint; "
            "load: only load checkpoint; "
            "test: load checkpoint and evaluate."
        ),
    )

    parser.add_argument("--data_path", type=str, default="dataset/FB15k-237")
    parser.add_argument("--import_path", type=str, default="import/KGE_model")
    parser.add_argument("--dataset_name", type=str, default="FB15k-237")
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    parser.add_argument("--hidden_dim", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=24.0)

    parser.add_argument("--negative_sample_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--warm_up_steps", type=int, default=None)

    parser.add_argument(
        "--negative_adversarial_sampling",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_negative_adversarial_sampling",
        dest="negative_adversarial_sampling",
        action="store_false",
    )

    parser.add_argument("--adversarial_temperature", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--uni_weight", action="store_true", default=False)

    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)

    parser.add_argument("--cpu_num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--valid_steps", type=int, default=2000)
    parser.add_argument("--save_checkpoint_steps", type=int, default=1000)
    parser.add_argument("--test_log_steps", type=int, default=2000)

    return parser.parse_args()


def main():
    args = parse_args()

    config = RotatEConfig(
        data_path=args.data_path,
        import_path=args.import_path,
        dataset_name=args.dataset_name,
        checkpoint_dir=args.checkpoint_dir,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        negative_sample_size=args.negative_sample_size,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warm_up_steps=args.warm_up_steps,
        negative_adversarial_sampling=args.negative_adversarial_sampling,
        adversarial_temperature=args.adversarial_temperature,
        regularization=args.regularization,
        uni_weight=args.uni_weight,
        cuda=args.cuda,
        gpu_id=args.gpu_id,
        amp=args.amp,
        cpu_num=args.cpu_num,
        seed=args.seed,
        log_steps=args.log_steps,
        valid_steps=args.valid_steps,
        save_checkpoint_steps=args.save_checkpoint_steps,
        test_log_steps=args.test_log_steps,
    )

    manager = RotatEManager(config)

    if args.mode == "auto":
        if manager.has_checkpoint():
            manager.load_checkpoint(load_optimizer=False)
            logging.info("Existing RotatE checkpoint loaded.")
        else:
            manager.train(resume=False)

    elif args.mode == "train":
        manager.train(resume=False)

    elif args.mode == "resume":
        manager.train(resume=True)

    elif args.mode == "load":
        manager.load_checkpoint(load_optimizer=False)
        logging.info("RotatE checkpoint loaded successfully.")

    elif args.mode == "test":
        manager.load_checkpoint(load_optimizer=False)
        metrics = manager.evaluate(split="test")
        manager.log_metrics("Test", 0, metrics)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()