import json
import random
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from openai import OpenAI


############################################
# 0. 基础配置
############################################

DATA_DIR = "/home/wenbin.guo/RAG/data/FB15k-237"

ENTITY_PATH = f"{DATA_DIR}/entity.json"
RELATION_PATH = f"{DATA_DIR}/relation.json"

TRAIN_PATH = f"{DATA_DIR}/train2id.txt"
VALID_PATH = f"{DATA_DIR}/valid2id.txt"
TEST_PATH = f"{DATA_DIR}/test2id.txt"

OUTPUT_PATH = "/home/wenbin.guo/DKGE4R/data/FB15k-237/eval_triple_classification_results.json"

MODEL_PATH = "/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

RANDOM_SEED = 42
NUM_TEST_POSITIVE = 10
# 如果只想先测试 500 条，改成 500
# 如果全量测试，保持 None

random.seed(RANDOM_SEED)


############################################
# 1. OpenAI client
############################################

client = OpenAI(
    base_url="http://localhost:22014/v1",
    api_key="EMPTY"
)


############################################
# 2. 数据读取
############################################

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_triples(path: str) -> List[Tuple[int, int, int]]:
    """
    原始格式：h t r
    返回格式：(h, r, t)
    """
    triples = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        h, t, r = line.split()
        triples.append((int(h), int(r), int(t)))

    return triples


def build_entity_dict(entity_list: List[Dict]) -> Dict[int, Dict]:
    entity_dict = {}

    for item in entity_list:
        entity_dict[int(item["value"])] = {
            "label": item["label"],
            "classname": item.get("classname", "Unknown")
        }

    return entity_dict


def build_relation_dict(relation_list: List[Dict]) -> Dict[int, Dict]:
    relation_dict = {}

    for item in relation_list:
        rid = int(item["id"])
        relation_dict[rid] = {
            "label": item["label"],
            "domain": item.get("domain", "Unknown"),
            "range": item.get("range", "Unknown")
        }

    return relation_dict


############################################
# 3. 构造负样本
############################################

def build_all_true_triple_set(*triple_lists):
    true_set = set()

    for triples in triple_lists:
        for h, r, t in triples:
            true_set.add((h, r, t))

    return true_set


def corrupt_tail_negative_sample(
    h: int,
    r: int,
    t: int,
    entity_ids: List[int],
    true_triple_set: set,
    max_try: int = 100
) -> Tuple[int, int, int]:

    for _ in range(max_try):
        neg_t = random.choice(entity_ids)

        if neg_t == t:
            continue

        neg_triple = (h, r, neg_t)

        if neg_triple not in true_triple_set:
            return neg_triple

    raise RuntimeError(f"Failed to generate negative sample for triple {(h, r, t)}")


def build_classification_samples(
    test_triples: List[Tuple[int, int, int]],
    entity_ids: List[int],
    true_triple_set: set,
    num_positive: int = None
):
    """
    每个正样本构造一个负样本。
    输出：
    [
      {
        "triple": (h, r, t),
        "label": 1
      },
      {
        "triple": (h, r, neg_t),
        "label": 0
      }
    ]
    """

    if num_positive is not None:
        selected_pos = random.sample(test_triples, num_positive)
    else:
        selected_pos = test_triples

    samples = []

    for h, r, t in selected_pos:
        samples.append({
            "triple": (h, r, t),
            "label": 1
        })

        neg_h, neg_r, neg_t = corrupt_tail_negative_sample(
            h=h,
            r=r,
            t=t,
            entity_ids=entity_ids,
            true_triple_set=true_triple_set
        )

        samples.append({
            "triple": (neg_h, neg_r, neg_t),
            "label": 0
        })

    random.shuffle(samples)

    return samples


############################################
# 4. Prompt
############################################

def build_prompt(
    h_label: str,
    r_label: str,
    t_label: str,
    h_type: str,
    t_type: str,
    domain: str,
    range_: str
) -> str:

    prompt = f"""
You are an expert knowledge graph triple classification system.

Your task is to determine whether the given triple is TRUE or FALSE.

Triple format:
(head entity, relation, tail entity)

Given triple:
({h_label}, {r_label}, {t_label})

Ontology information:
- Head entity type: {h_type}
- Tail entity type: {t_type}
- Relation domain: {domain}
- Relation range: {range_}

Important:
- Output ONLY valid JSON.
- Do NOT output explanations.
- Do NOT output entity names.
- The label must be either 1 or 0.
- Use 1 if the triple is likely TRUE.
- Use 0 if the triple is likely FALSE.

Before answering, internally:
1. Understand the semantic meaning of the relation.
2. Check whether the head type matches the relation domain.
3. Check whether the tail type matches the relation range.
4. Judge whether the triple is semantically plausible.

Output format:

{{
  "label": 1
}}
"""
    return prompt


############################################
# 5. LLM 调用与解析
############################################

def query_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return resp.choices[0].message.content


def clean_llm_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def extract_label(llm_output: str):
    cleaned = clean_llm_output(llm_output)

    try:
        data = json.loads(cleaned)
        label = int(data["label"])
        if label in [0, 1]:
            return label
    except:
        pass

    match = re.search(r'\b[01]\b', cleaned)
    if match:
        return int(match.group())

    return None


############################################
# 6. 单条预测
############################################

def predict_sample(
    sample: Dict,
    entity_dict: Dict[int, Dict],
    relation_dict: Dict[int, Dict]
):

    h, r, t = sample["triple"]
    gold_label = sample["label"]

    h_info = entity_dict[h]
    t_info = entity_dict[t]
    r_info = relation_dict[r]

    h_label = h_info["label"]
    t_label = t_info["label"]
    h_type = h_info.get("classname", "Unknown")
    t_type = t_info.get("classname", "Unknown")

    r_label = r_info["label"]
    domain = r_info.get("domain", "Unknown")
    range_ = r_info.get("range", "Unknown")

    prompt = build_prompt(
        h_label=h_label,
        r_label=r_label,
        t_label=t_label,
        h_type=h_type,
        t_type=t_type,
        domain=domain,
        range_=range_
    )

    llm_output = query_llm(prompt)
    pred_label = extract_label(llm_output)

    is_correct = int(pred_label == gold_label) if pred_label is not None else 0

    return {
        "triple": {
            "head": h_label,
            "relation": r_label,
            "tail": t_label,
            "head_type": h_type,
            "tail_type": t_type,
            "domain": domain,
            "range": range_
        },
        "gold_label": gold_label,
        "pred_label": pred_label,
        "is_correct": is_correct,
        "llm_output": llm_output
    }


############################################
# 7. 指标计算
############################################

def compute_metrics(results: List[Dict]):
    tp = fp = tn = fn = 0
    invalid = 0

    for item in results:
        gold = item["gold_label"]
        pred = item["pred_label"]

        if pred is None:
            invalid += 1
            continue

        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 1 and pred == 0:
            fn += 1

    total_valid = tp + fp + tn + fn
    total = len(results)

    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "total": total,
        "valid": total_valid,
        "invalid": invalid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


############################################
# 8. 主函数
############################################

def main():

    print("Loading data...")

    entity_list = load_json(ENTITY_PATH)
    relation_list = load_json(RELATION_PATH)

    train_triples = load_triples(TRAIN_PATH)
    valid_triples = load_triples(VALID_PATH)
    test_triples = load_triples(TEST_PATH)

    entity_dict = build_entity_dict(entity_list)
    relation_dict = build_relation_dict(relation_list)

    entity_ids = list(entity_dict.keys())

    true_triple_set = build_all_true_triple_set(
        train_triples,
        valid_triples,
        test_triples
    )

    print(f"Train triples: {len(train_triples)}")
    print(f"Valid triples: {len(valid_triples)}")
    print(f"Test triples : {len(test_triples)}")
    print(f"All true triples: {len(true_triple_set)}")

    print("Building classification samples...")

    samples = build_classification_samples(
        test_triples=test_triples,
        entity_ids=entity_ids,
        true_triple_set=true_triple_set,
        num_positive=NUM_TEST_POSITIVE
    )

    print(f"Classification samples: {len(samples)}")
    print("Positive and negative samples are balanced 1:1.")

    results = []

    for idx, sample in enumerate(
        tqdm(samples, desc="Triple Classification", ncols=100),
        start=1
    ):
        result = predict_sample(
            sample=sample,
            entity_dict=entity_dict,
            relation_dict=relation_dict
        )

        results.append(result)

        if idx % 100 == 0:
            metrics = compute_metrics(results)
            print(f"\n===== Processed {idx}/{len(samples)} samples =====")
            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")
            print(f"F1       : {metrics['f1']:.4f}")
            print(f"Invalid  : {metrics['invalid']}")
            print("Last Triple:", result["triple"])
            print("Gold Label:", result["gold_label"])
            print("Pred Label:", result["pred_label"])
            print("Correct:", result["is_correct"])
            print("=" * 60)

    final_metrics = compute_metrics(results)

    print("\n========== Final Results ==========")
    print(f"Total    : {final_metrics['total']}")
    print(f"Valid    : {final_metrics['valid']}")
    print(f"Invalid  : {final_metrics['invalid']}")
    print(f"Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall   : {final_metrics['recall']:.4f}")
    print(f"F1       : {final_metrics['f1']:.4f}")
    print(f"TP       : {final_metrics['tp']}")
    print(f"FP       : {final_metrics['fp']}")
    print(f"TN       : {final_metrics['tn']}")
    print(f"FN       : {final_metrics['fn']}")

    save_obj = {
        "metrics": final_metrics,
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()