#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取已经训练好的 KGE 模型 checkpoint，并在 valid / test / train 上做评测。

使用方式示例：
1) 只评测 test：
python eval_saved_kge_model.py \
  --init_checkpoint /path/to/your/saved_model \
  --data_path /path/to/FB15k-237 \
  --do_test --cuda

2) 同时评测 valid 和 test：
python eval_saved_kge_model.py \
  --init_checkpoint /path/to/your/saved_model \
  --data_path /path/to/FB15k-237 \
  --do_valid --do_test --cuda

3) 只用 checkpoint 里的配置自动恢复：
python eval_saved_kge_model.py \
  --init_checkpoint /path/to/your/saved_model \
  --do_test --cuda

说明：
- 默认会优先从 init_checkpoint/config.json 中恢复训练时的模型结构参数。
- 如果你显式传了 --data_path，则优先使用你传入的数据路径。
- 该脚本不进行训练，只做加载与评测。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset


# =========================
# DataLoader 部分
# =========================
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [
                (0, rand_head) if (rand_head, relation, tail) not in self.triple_set else (-1, head)
                for rand_head in range(self.nentity)
            ]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [
                (0, rand_tail) if (head, relation, rand_tail) not in self.triple_set else (-1, tail)
                for rand_tail in range(self.nentity)
            ]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


# =========================
# Model 部分
# =========================
class KGEModel(nn.Module):
    def __init__(
        self,
        model_name,
        nentity,
        nrelation,
        hidden_dim,
        gamma,
        double_entity_embedding=False,
        double_relation_embedding=False,
    ):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False,
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item(),
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        if mode == 'single':
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(
                batch_size, negative_sample_size, -1
            )
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        return score.sum(dim=2)

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail
        return score.sum(dim=2)

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
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
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        model.eval()

        if args.countries:
            sample = []
            y_true = []
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)
            auc_pr = average_precision_score(y_true, y_score)
            metrics = {'auc_pr': auc_pr}
            return metrics

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch',
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch',
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        logs = []
        step = 0
        total_steps = sum(len(dataset) for dataset in test_dataset_list)

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda(non_blocking=True)
                        negative_sample = negative_sample.cuda(non_blocking=True)
                        filter_bias = filter_bias.cuda(non_blocking=True)

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    argsort = torch.argsort(score, dim=1, descending=True)
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero(as_tuple=False)
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()

                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)', step, total_steps)
                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum(log[metric] for log in logs) / len(logs)
        return metrics


# =========================
# 工具函数
# =========================
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load a trained KGE checkpoint and evaluate it.',
        usage='python eval_saved_kge_model.py [<args>] [-h | --help]',
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU for evaluation')
    parser.add_argument('--do_valid', action='store_true', help='evaluate on valid set')
    parser.add_argument('--do_test', action='store_true', help='evaluate on test set')
    parser.add_argument('--evaluate_train', action='store_true', help='evaluate on train set')

    parser.add_argument('--countries', action='store_true', default=False,
                        help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default=None,
                        help='dataset directory; if omitted, try reading it from checkpoint config.json')
    parser.add_argument('--init_checkpoint', type=str, required=True,
                        help='directory containing checkpoint and config.json')

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--double_entity_embedding', action='store_true', default=False)
    parser.add_argument('--double_relation_embedding', action='store_true', default=False)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)

    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--cpu_num', type=int, default=10)
    parser.add_argument('--test_log_steps', type=int, default=1000)

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--save_metrics_file', type=str, default=None,
                        help='optional path to save evaluation metrics as json')
    parser.add_argument('--build_subgraph', action='store_true',
                        help='whether to build and save query subgraphs after evaluation')
    parser.add_argument('--subgraph_output_dir', type=str, default=None,
                        help='directory to save constructed subgraphs')
    parser.add_argument('--subgraph_topk_tails', type=int, default=50,
                        help='Top-k candidate tails from KGE for each query')
    parser.add_argument('--subgraph_topm_paths', type=int, default=20,
                        help='Top-m paths kept in final subgraph')
    parser.add_argument('--subgraph_max_hops', type=int, default=3,
                        help='maximum hops when searching paths')
    parser.add_argument('--subgraph_max_expand_per_node', type=int, default=3,
                        help='max 1-hop neighbors kept for each key node')
    parser.add_argument('--subgraph_max_key_nodes', type=int, default=10,
                        help='max number of key nodes used for expansion')

    # ===== 新增：断点续跑 =====
    parser.add_argument('--subgraph_start_idx', type=int, default=0,
                        help='start index for subgraph generation, e.g. 20464')
    parser.add_argument('--skip_existing_subgraph', action='store_true',
                        help='skip subgraph json if the target file already exists')

    return parser.parse_args(args)


def override_config(args):
    config_path = os.path.join(args.init_checkpoint, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError('config.json not found in checkpoint directory: %s' % args.init_checkpoint)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    args.countries = cfg.get('countries', args.countries)
    if args.data_path is None:
        args.data_path = cfg.get('data_path', None)

    args.model = cfg.get('model', args.model)
    args.double_entity_embedding = cfg.get('double_entity_embedding', args.double_entity_embedding)
    args.double_relation_embedding = cfg.get('double_relation_embedding', args.double_relation_embedding)
    args.hidden_dim = cfg.get('hidden_dim', args.hidden_dim)
    args.gamma = cfg.get('gamma', args.gamma)
    args.test_batch_size = cfg.get('test_batch_size', args.test_batch_size)

    if args.model is None:
        raise ValueError('Cannot determine model name from arguments or config.json')
    if args.hidden_dim is None:
        raise ValueError('Cannot determine hidden_dim from arguments or config.json')
    if args.gamma is None:
        raise ValueError('Cannot determine gamma from arguments or config.json')
    if args.test_batch_size is None:
        args.test_batch_size = 16


def set_logger(args):
    log_file = os.path.join(args.init_checkpoint, 'eval.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode: str, step: int, metrics: Dict[str, float]):
    for metric, value in metrics.items():
        logging.info('%s %s at step %d: %f', mode, metric, step, value)


def read_id_dict(file_path: str) -> Dict[str, int]:
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            idx, name = line.strip().split('\t')
            mapping[name] = int(idx)
    return mapping


def read_triple(file_path: str, entity2id: Dict[str, int], relation2id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    triples = []
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def build_adjacency_from_triples(triples: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    构建从 head 出发的邻接表：adj[head] = [(relation, tail), ...]
    """
    adjacency = defaultdict(list)
    for h, r, t in triples:
        adjacency[h].append((r, t))
    return adjacency


def score_path_by_rotate_relation(model, target_relation_id: int, path_relation_ids: List[int], path_length_penalty: float = 0.1) -> float:
    """
    用 RotatE 的关系相位组合思想，对一条路径做一个简单打分。
    分数越大越好。
    """
    if len(path_relation_ids) == 0:
        return -1e9

    with torch.no_grad():
        rel_emb = model.relation_embedding.detach()
        target = rel_emb[target_relation_id]
        path_rel = rel_emb[path_relation_ids]

        if model.model_name in ['RotatE', 'pRotatE']:
            pi = 3.14159265358979323846
            target_phase = target / (model.embedding_range.item() / pi)
            path_phase = path_rel / (model.embedding_range.item() / pi)
            composed_phase = path_phase.sum(dim=0)
            sim = -torch.mean(torch.abs(composed_phase - target_phase)).item()
        else:
            composed = path_rel.mean(dim=0)
            sim = torch.nn.functional.cosine_similarity(
                target.unsqueeze(0), composed.unsqueeze(0), dim=1
            ).item()

    return sim - path_length_penalty * len(path_relation_ids)



def bfs_collect_paths(
    adjacency: Dict[int, List[Tuple[int, int]]],
    head_id: int,
    target_tail_id: int,
    max_hops: int = 3,
    max_paths_per_tail: int = 20,
) -> List[Dict[str, Any]]:
    """
    从 head 到 target_tail 做受限 BFS，收集长度不超过 max_hops 的路径。
    """
    results = []
    queue = deque()
    queue.append((head_id, [head_id], []))

    while queue and len(results) < max_paths_per_tail:
        current_node, node_path, rel_path = queue.popleft()
        depth = len(rel_path)

        if depth > max_hops:
            continue

        if current_node == target_tail_id and depth > 0:
            results.append({
                'nodes': node_path[:],
                'relations': rel_path[:],
                'tail': target_tail_id,
            })
            continue

        if depth == max_hops:
            continue

        for rel_id, next_node in adjacency.get(current_node, []):
            if next_node in node_path:
                continue
            queue.append((next_node, node_path + [next_node], rel_path + [rel_id]))

    return results



def expand_key_nodes(
    adjacency: Dict[int, List[Tuple[int, int]]],
    key_nodes: List[int],
    keep_neighbor_per_node: int = 3,
) -> List[Tuple[int, int, int]]:
    """
    对关键节点做受限 1-hop 扩展。
    """
    extra_edges = []
    for node in key_nodes:
        neighbors = adjacency.get(node, [])[:keep_neighbor_per_node]
        for rel_id, tail_id in neighbors:
            extra_edges.append((node, rel_id, tail_id))
    return extra_edges



def build_query_subgraph(
    model,
    query_h: int,
    query_r: int,
    candidate_tails: List[int],
    all_true_triples: List[Tuple[int, int, int]],
    top_m_paths: int = 20,
    max_hops: int = 3,
    max_expand_per_node: int = 3,
    max_key_nodes: int = 10,
) -> Dict[str, Any]:
    """
    为单个查询 (h, r, ?) 构建子图，并返回可序列化结果。
    """
    adjacency = build_adjacency_from_triples(all_true_triples)

    all_paths = []
    for tail_id in candidate_tails:
        paths = bfs_collect_paths(
            adjacency=adjacency,
            head_id=query_h,
            target_tail_id=tail_id,
            max_hops=max_hops,
            max_paths_per_tail=top_m_paths,
        )
        for item in paths:
            item['path_score'] = score_path_by_rotate_relation(
                model=model,
                target_relation_id=query_r,
                path_relation_ids=item['relations'],
            )
            all_paths.append(item)

    all_paths = sorted(all_paths, key=lambda x: x['path_score'], reverse=True)[:top_m_paths]

    node_frequency = defaultdict(int)
    edge_set = set()
    node_set = set([query_h])

    for item in all_paths:
        nodes = item['nodes']
        relations = item['relations']
        for n in nodes:
            node_frequency[n] += 1
            node_set.add(n)
        for i, rel_id in enumerate(relations):
            edge_set.add((nodes[i], rel_id, nodes[i + 1]))

    key_nodes = [node for node, _ in sorted(node_frequency.items(), key=lambda x: x[1], reverse=True)[:max_key_nodes]]
    expanded_edges = expand_key_nodes(
        adjacency=adjacency,
        key_nodes=key_nodes,
        keep_neighbor_per_node=max_expand_per_node,
    )

    for h_id, r_id, t_id in expanded_edges:
        edge_set.add((h_id, r_id, t_id))
        node_set.add(h_id)
        node_set.add(t_id)

    subgraph = {
        'query': {
            'head': query_h,
            'relation': query_r,
            'candidate_tails': candidate_tails,
        },
        'nodes': sorted(list(node_set)),
        'edges': [
            {'head': h_id, 'relation': r_id, 'tail': t_id}
            for h_id, r_id, t_id in sorted(edge_set)
        ],
        'paths': all_paths,
        'key_nodes': key_nodes,
    }
    return subgraph



def save_query_subgraph(
    subgraph: Dict[str, Any],
    output_dir: str,
    query_name: str,
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{query_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subgraph, f, indent=2, ensure_ascii=False)
    logging.info('Saved subgraph to %s', output_path)



def get_topk_candidate_tails(model, query_h: int, query_r: int, nentity: int, device: str, topk: int = 50) -> List[int]:
    with torch.no_grad():
        head_ids = torch.LongTensor([[query_h, query_r, 0]]).to(device)
        all_tails = torch.arange(nentity, dtype=torch.long, device=device).unsqueeze(0)
        score = model((head_ids, all_tails), mode='tail-batch').squeeze(0)
        topk = min(topk, nentity)
        _, indices = torch.topk(score, k=topk, dim=0)
    return indices.detach().cpu().tolist()



def load_model_from_checkpoint(args, nentity: int, nrelation: int):
    checkpoint_path = os.path.join(args.init_checkpoint, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('checkpoint file not found: %s' % checkpoint_path)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )

    map_location = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if 'model_state_dict' not in checkpoint:
        raise KeyError('checkpoint does not contain model_state_dict')

    missing_keys, unexpected_keys = kge_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing_keys:
        logging.warning('Missing keys when loading checkpoint: %s', missing_keys)
    if unexpected_keys:
        logging.warning('Unexpected keys when loading checkpoint: %s', unexpected_keys)

    step = checkpoint.get('step', 0)

    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('--cuda was set, but CUDA is not available')
        kge_model = kge_model.cuda()

    kge_model.eval()
    return kge_model, step


def save_subgraphs_for_queries(
    model,
    triples: List[Tuple[int, int, int]],
    all_true_triples: List[Tuple[int, int, int]],
    args,
    split_name: str,
):
    if not args.build_subgraph:
        return

    if args.subgraph_output_dir is None:
        raise ValueError('--build_subgraph was set, but --subgraph_output_dir is None')

    device = 'cuda' if args.cuda else 'cpu'
    split_output_dir = os.path.join(args.subgraph_output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    start_idx = max(0, args.subgraph_start_idx)

    logging.info(
        'Start saving subgraphs for %s from index %d (total=%d)',
        split_name, start_idx, len(triples)
    )

    for idx in range(start_idx, len(triples)):
        h, r, t = triples[idx]

        output_path = os.path.join(split_output_dir, f'{split_name}_{idx}.json')

        # 已存在就跳过
        if args.skip_existing_subgraph and os.path.exists(output_path):
            if idx % 100 == 0:
                logging.info('Skip existing subgraph: %s', output_path)
            continue

        candidate_tails = get_topk_candidate_tails(
            model=model,
            query_h=h,
            query_r=r,
            nentity=args.nentity,
            device=device,
            topk=args.subgraph_topk_tails,
        )

        if t not in candidate_tails:
            candidate_tails.append(t)

        subgraph = build_query_subgraph(
            model=model,
            query_h=h,
            query_r=r,
            candidate_tails=candidate_tails,
            all_true_triples=all_true_triples,
            top_m_paths=args.subgraph_topm_paths,
            max_hops=args.subgraph_max_hops,
            max_expand_per_node=args.subgraph_max_expand_per_node,
            max_key_nodes=args.subgraph_max_key_nodes,
        )
        subgraph['gold_tail'] = t

        # 直接按完整路径保存，避免重复拼接文件名
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subgraph, f, indent=2, ensure_ascii=False)

        if idx % 100 == 0:
            logging.info('Saved subgraphs for %s queries: %d/%d', split_name, idx, len(triples))



def maybe_save_metrics(save_path: str, results: Dict[str, Dict[str, float]]):
    if save_path is None:
        return
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info('Saved evaluation metrics to %s', save_path)


# =========================
# 主函数
# =========================
def main(args):
    if (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('At least one of --do_valid / --do_test / --evaluate_train must be specified.')

    override_config(args)

    if args.data_path is None:
        raise ValueError('data_path is still None. Please pass --data_path explicitly or ensure config.json contains it.')

    set_logger(args)

    logging.info('========== Evaluation Start ==========' )
    logging.info('Checkpoint: %s', args.init_checkpoint)
    logging.info('Data Path  : %s', args.data_path)

    entity2id = read_id_dict(os.path.join(args.data_path, 'entities.dict'))
    relation2id = read_id_dict(os.path.join(args.data_path, 'relations.dict'))

    if args.countries:
        regions = []
        with open(os.path.join(args.data_path, 'regions.list'), 'r', encoding='utf-8') as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s', args.model)
    logging.info('#entity: %d', nentity)
    logging.info('#relation: %d', nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)

    logging.info('#train: %d', len(train_triples))
    logging.info('#valid: %d', len(valid_triples))
    logging.info('#test : %d', len(test_triples))

    all_true_triples = train_triples + valid_triples + test_triples

    kge_model, step = load_model_from_checkpoint(args, nentity, nrelation)

    logging.info('Model loaded successfully from checkpoint.')
    logging.info('Checkpoint step = %d', step)
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s', name, str(param.size()), str(param.requires_grad))

    final_results = {}

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        valid_metrics = KGEModel.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, valid_metrics)
        final_results['valid'] = valid_metrics
        if args.build_subgraph:
            logging.info('Building and saving valid subgraphs...')
            save_subgraphs_for_queries(kge_model, valid_triples, all_true_triples, args, 'valid')

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_metrics = KGEModel.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, test_metrics)
        final_results['test'] = test_metrics
        if args.build_subgraph:
            logging.info('Building and saving test subgraphs...')
            save_subgraphs_for_queries(kge_model, test_triples, all_true_triples, args, 'test')

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        train_metrics = KGEModel.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Train', step, train_metrics)
        final_results['train'] = train_metrics
        if args.build_subgraph:
            logging.info('Building and saving train subgraphs...')
            save_subgraphs_for_queries(kge_model, train_triples, all_true_triples, args, 'train')

    maybe_save_metrics(args.save_metrics_file, final_results)

    logging.info('========== Evaluation Finished ==========' )
    print('\n===== Final Evaluation Results =====')
    print(json.dumps(final_results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main(parse_args())
