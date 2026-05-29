import os
import json
from tqdm import tqdm

from model import KGEModel
from run import parse_args, override_config
import torch
import torch.nn.functional as F
from collections import deque


class RotatESemanticRanker:
    """
    RotatE-guided ranking for one-hop neighbors and two-hop paths.
    """

    def __init__(
        self,
        model,
        device="cuda",
        one_rel_weight=0.5,
        one_tail_weight=0.5,
        path_rel_weight=0.5,
        path_tail_weight=0.5,
    ):
        self.model = model
        self.device = device

        self.one_rel_weight = one_rel_weight
        self.one_tail_weight = one_tail_weight
        self.path_rel_weight = path_rel_weight
        self.path_tail_weight = path_tail_weight

        self.entity_emb = model.entity_embedding.weight.data.to(device)
        self.relation_emb = model.relation_embedding.weight.data.to(device)

        # RotatE usually uses embedding_range / pi to transform relation into phase
        self.pi = 3.141592653589793
        self.embedding_range = getattr(model, "embedding_range", None)

        if self.embedding_range is not None:
            if torch.is_tensor(self.embedding_range):
                self.embedding_range = self.embedding_range.item()

    def get_entity_complex(self, eid):
        emb = self.entity_emb[eid]

        # RotatE entity embedding is usually double-dimensional: [real, imag]
        dim = emb.shape[-1] // 2
        re = emb[:dim]
        im = emb[dim:]
        return re, im

    def get_relation_complex(self, rid, inverse=False):
        rel = self.relation_emb[rid]

        # Standard RotatE: relation embedding is phase vector
        if self.embedding_range is not None:
            phase = rel / (self.embedding_range / self.pi)
        else:
            phase = rel

        re = torch.cos(phase)
        im = torch.sin(phase)

        # Backward edge corresponds to inverse rotation
        if inverse:
            im = -im

        return re, im

    def complex_mul(self, a_re, a_im, b_re, b_im):
        re = a_re * b_re - a_im * b_im
        im = a_re * b_im + a_im * b_re
        return re, im

    def complex_cosine(self, a_re, a_im, b_re, b_im):
        a = torch.cat([a_re, a_im], dim=-1)
        b = torch.cat([b_re, b_im], dim=-1)
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def relation_similarity(self, ri, rq):
        ri_re, ri_im = self.get_relation_complex(ri)
        rq_re, rq_im = self.get_relation_complex(rq)
        return self.complex_cosine(ri_re, ri_im, rq_re, rq_im)

    def query_tail_compatibility(self, h, rq, t):
        h_re, h_im = self.get_entity_complex(h)
        rq_re, rq_im = self.get_relation_complex(rq)
        t_re, t_im = self.get_entity_complex(t)

        pred_re, pred_im = self.complex_mul(h_re, h_im, rq_re, rq_im)

        dist = torch.norm(torch.cat([pred_re - t_re, pred_im - t_im], dim=-1), p=2)
        return (-dist).item()

    def score_one_hop(self, h, rq, ri, ti):
        s_rel = self.relation_similarity(ri, rq)
        s_tail = self.query_tail_compatibility(h, rq, ti)

        return (
            self.one_rel_weight * s_rel
            + self.one_tail_weight * s_tail
        )

    def score_path(self, h, rq, path):
        """
        path format:
        [(src, r, dst, direction), ...]
        direction: "forward" or "backward"
        """
        rq_re, rq_im = self.get_relation_complex(rq)

        # Compose path relations
        path_re, path_im = None, None

        for src, r, dst, direction in path:
            inverse = direction == "backward"
            r_re, r_im = self.get_relation_complex(r, inverse=inverse)

            if path_re is None:
                path_re, path_im = r_re, r_im
            else:
                path_re, path_im = self.complex_mul(path_re, path_im, r_re, r_im)

        # 1. path relation composition should be close to query relation
        s_path_rel = self.complex_cosine(path_re, path_im, rq_re, rq_im)

        # 2. path endpoint should be close to query-induced target position
        terminal = path[-1][2]
        s_path_tail = self.query_tail_compatibility(h, rq, terminal)

        return (
            self.path_rel_weight * s_path_rel
            + self.path_tail_weight * s_path_tail
        )


def rank_one_hop_neighbors(
    h,
    rq,
    out_graph,
    ranker,
    topk=5
):
    """
    Rank one-hop outgoing neighbors by:
    0.5 * relation similarity + 0.5 * RotatE query-tail compatibility
    """
    neighbors = out_graph.get(h, [])

    scored = []
    for ri, ti in neighbors:
        try:
            score = ranker.score_one_hop(h, rq, ri, ti)
            scored.append((score, ri, ti))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)

    return [(ri, ti, score) for score, ri, ti in scored[:topk]]


def enumerate_two_hop_paths(
    h,
    undirected_graph,
    max_depth=2,
    max_paths=200
):
    """
    Enumerate local paths starting from h up to max_depth.
    For your current setting, max_depth=2.
    """
    paths = []
    queue = deque()
    queue.append((h, []))
    visited_paths = 0

    while queue and visited_paths < max_paths:
        current, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for r, nxt, direction in undirected_graph.get(current, []):
            # avoid simple cycle
            visited_nodes = {edge[0] for edge in path}
            visited_nodes.add(current)

            if nxt in visited_nodes:
                continue

            new_path = path + [(current, r, nxt, direction)]
            paths.append(new_path)
            queue.append((nxt, new_path))

            visited_paths += 1
            if visited_paths >= max_paths:
                break

    # keep only exact two-hop paths if you want pure two-hop evidence
    two_hop_paths = [p for p in paths if len(p) == 2]
    return two_hop_paths


def rank_two_hop_paths(
    h,
    rq,
    undirected_graph,
    ranker,
    topm=5,
    max_depth=2
):
    """
    Rank two-hop paths by:
    0.5 * path relation composition similarity
    + 0.5 * endpoint compatibility
    """
    paths = enumerate_two_hop_paths(
        h=h,
        undirected_graph=undirected_graph,
        max_depth=max_depth
    )

    scored = []
    for p in paths:
        try:
            score = ranker.score_path(h, rq, p)
            scored.append((score, p))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)

    return [(p, score) for score, p in scored[:topm]]


# =========================
# ✔ 正确读取训练ID映射（关键修复）
# 如果你的数据是 entity2id.txt / relation2id.txt 格式就用这个
# =========================
def load_id_map(file_path):
    id_map = {}
    with open(file_path, "r") as f:
        for line in f:
            name, idx = line.strip().split("\t")
            id_map[name] = int(idx)
    return id_map


# =========================
# build mapping（修复版：优先用训练ID）
# =========================
def build_mapping(data_path):

    entity_file = os.path.join(data_path, "entity2id.txt")
    relation_file = os.path.join(data_path, "relation2id.txt")

    # ✔ 如果存在标准OpenKE格式，直接用（强烈推荐）
    if os.path.exists(entity_file) and os.path.exists(relation_file):
        print("[INFO] Using entity2id.txt / relation2id.txt (SAFE MODE)")
        entity2id = load_id_map(entity_file)
        relation2id = load_id_map(relation_file)
        return entity2id, relation2id

    # ❗ fallback（你原来的方式）
    print("[WARN] No id files found, fallback to rebuild mapping (RISK OF MISMATCH)")

    entity2id = {}
    relation2id = {}

    def scan(file):
        with open(file, "r") as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)

    scan(os.path.join(data_path, "train.txt"))
    scan(os.path.join(data_path, "valid.txt"))
    scan(os.path.join(data_path, "test.txt"))

    return entity2id, relation2id


# =========================
# load model（修复 + check）
# =========================
def load_model(args):

    config_path = os.path.join(args.init_checkpoint, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    args.nentity = config["nentity"]
    args.nrelation = config["nrelation"]
    args.hidden_dim = config["hidden_dim"]
    args.gamma = config["gamma"]
    args.double_entity_embedding = config["double_entity_embedding"]
    args.double_relation_embedding = config["double_relation_embedding"]

    print("\n[DEBUG]")
    print("nentity =", args.nentity)
    print("nrelation =", args.nrelation, "\n")

    model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    ckpt_path = os.path.join(args.init_checkpoint, "checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[INFO] Model loaded successfully ✔")

    return model


# =========================
# read triples（带进度）
# =========================
def read_triples(file, entity2id, relation2id):
    triples = []
    with open(file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Loading {os.path.basename(file)}"):
        h, r, t = line.strip().split("\t")
        triples.append((entity2id[h], relation2id[r], entity2id[t]))

    return triples


# =========================
# main
# =========================
def main():

    args = parse_args()
    override_config(args)

    data_path = args.data_path

    # =========================
    # mapping
    # =========================
    entity2id, relation2id = build_mapping(data_path)

    # =========================
    # triples
    # =========================
    test_triples = read_triples(
        os.path.join(data_path, "test.txt"),
        entity2id,
        relation2id
    )

    train_triples = read_triples(
        os.path.join(data_path, "train.txt"),
        entity2id,
        relation2id
    )

    valid_triples = read_triples(
        os.path.join(data_path, "valid.txt"),
        entity2id,
        relation2id
    )

    all_true_triples = train_triples + valid_triples + test_triples

    # =========================
    # model
    # =========================
    model = load_model(args)

    if args.cuda:
        model = model.cuda()

    # =========================
    # eval（带进度提示）
    # =========================
    print("\n[INFO] Start evaluation ...\n")

    metrics = model.test_step(
        model,
        test_triples,
        all_true_triples,
        args
    )

    print("\n===== FINAL RESULT =====")
    print(metrics)


if __name__ == "__main__":
    main()