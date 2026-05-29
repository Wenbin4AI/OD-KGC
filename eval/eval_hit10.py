import json
import random
from typing import List, Dict, Tuple
import os
import time
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError
import re
from tqdm import tqdm


def clean_llm_output(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def extract_selected_indices(llm_output: str, k: int = 10):
    cleaned_output = clean_llm_output(llm_output)

    try:
        data = json.loads(cleaned_output)
        indices = data["selected_indices"]

        if isinstance(indices, list):
            indices = [int(i) for i in indices]

            # 去重，保持顺序
            dedup = []
            for i in indices:
                if i not in dedup:
                    dedup.append(i)

            return dedup[:k]
    except:
        pass

    numbers = re.findall(r'\d+', cleaned_output)
    if len(numbers) >= k:
        dedup = []
        for x in numbers:
            i = int(x)
            if i not in dedup:
                dedup.append(i)
            if len(dedup) == k:
                break
        return dedup

    return None


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_entity_dict(entity_list: List[Dict]) -> Dict[int, Dict]:
    entity_dict = {}
    for item in entity_list:
        entity_dict[int(item["value"])] = {
            "label": item["label"],
            "classname": item.get("classname", "Unknown")
        }
    return entity_dict


def build_relation_dict(relation_list: List[Dict]) -> Dict[int, str]:
    relation_dict = {}
    for item in relation_list:
        relation_dict[int(item["id"])] = item["label"]
    return relation_dict


def load_triples(path: str) -> List[Tuple[int, int, int]]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        h, t, r = line.split()
        triples.append((int(h), int(t), int(r)))

    return triples


def build_tail_candidates(true_tail_id: int,
                          entity_dict: Dict,
                          num_candidates: int = 20) -> List[str]:

    all_entity_ids = list(entity_dict.keys())

    if true_tail_id in all_entity_ids:
        all_entity_ids.remove(true_tail_id)

    negative_ids = random.sample(all_entity_ids, num_candidates - 1)

    candidate_ids = negative_ids + [true_tail_id]
    random.shuffle(candidate_ids)

    candidates = [entity_dict[eid]["label"] for eid in candidate_ids]

    return candidates


def build_prompt(head: str,
                 relation: str,
                 candidates: List[str],
                 key_evidence: str) -> str:

    indexed_candidates = "\n".join(
        [f"{i}. {c}" for i, c in enumerate(candidates)]
    )

    prompt = f"""
You are an expert knowledge graph completion system.

Your task is to select the TOP 10 most likely tail entities from a candidate list.

Triple format:
(head entity, relation, tail entity)

Important:
- You are predicting ONLY the tail entity.
- The head entity "{head}" is NOT a valid answer.
- You MUST select exactly TEN candidate indices.
- The indices must be ordered from MOST likely to LEAST likely.
- You MUST NOT output entity names.
- You MUST NOT generate new entities.
- Output ONLY valid JSON.
- Do NOT repeat indices.
- All selected indices must come from the candidate list.

Before answering, internally:
1. Understand the semantic meaning of the relation.
2. Determine the expected type of the tail entity.
3. Use the key evidence to understand the local neighborhood.
4. Compare all candidates carefully.
5. Select the TOP 10 most consistent ones.
Do NOT output reasoning.

Key evidence:
{key_evidence}

Candidate entities:
{indexed_candidates}

Output format:

{{
  "selected_indices": [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
}}

Incomplete triple:
({head}, {relation}, ?)
"""
    return prompt


client = OpenAI(
    base_url="http://localhost:22014/v1",
    api_key="EMPTY"
)


def query_llm(prompt: str, max_retries: int = 2) -> str:
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
                messages=[{"role": "user", "content": prompt}],
                timeout=120
            )
            return resp.choices[0].message.content

        except (APITimeoutError, APIConnectionError, APIError) as e:
            last_error = repr(e)
            print(f"\n[Warning] LLM request failed, attempt {attempt + 1}/{max_retries + 1}: {last_error}")
            time.sleep(2)

        except Exception as e:
            last_error = repr(e)
            print(f"\n[Warning] Unexpected LLM error: {last_error}")
            break

    return None


def predict_for_triple(triple: Tuple[int, int, int],
                       entity_dict: Dict,
                       relation_dict: Dict,
                       key_evidence: str):

    h_id, t_id, r_id = triple

    head = entity_dict[h_id]["label"]
    relation = relation_dict[r_id]
    true_tail = entity_dict[t_id]["label"]

    candidates = build_tail_candidates(t_id, entity_dict, num_candidates=20)

    prompt = build_prompt(head, relation, candidates, key_evidence)

    result = query_llm(prompt)

    if result is None:
        return {
            "triple": (head, relation, true_tail),
            "candidates": candidates,
            "key_evidence": key_evidence,
            "llm_prediction": None,
            "pred_indices": None,
            "predicted_tails": [],
            "is_hit@1": 0,
            "is_hit@3": 0,
            "is_hit@10": 0,
            "status": "failed_timeout_or_api_error"
        }

    pred_indices = extract_selected_indices(result, k=10)

    predicted_tails = []

    if pred_indices is not None:
        for pred_idx in pred_indices:
            if 0 <= pred_idx < len(candidates):
                predicted_tails.append(candidates[pred_idx])
            else:
                predicted_tails.append(None)

    is_hit1 = int(true_tail in predicted_tails[:1])
    is_hit3 = int(true_tail in predicted_tails[:3])
    is_hit10 = int(true_tail in predicted_tails[:10])

    return {
        "triple": (head, relation, true_tail),
        "candidates": candidates,
        "key_evidence": key_evidence,
        "llm_prediction": result,
        "pred_indices": pred_indices,
        "predicted_tails": predicted_tails,
        "is_hit@1": is_hit1,
        "is_hit@3": is_hit3,
        "is_hit@10": is_hit10,
        "status": "ok"
    }

def save_checkpoint(output_path, results, hit1_count, hit3_count, hit10_count):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = len(results)
    metrics = {
        "total": total,
        "hit1_count": hit1_count,
        "hit3_count": hit3_count,
        "hit10_count": hit10_count,
        "Hit@1": hit1_count / total if total else 0.0,
        "Hit@3": hit3_count / total if total else 0.0,
        "Hit@10": hit10_count / total if total else 0.0,
        "failed_count": sum(1 for x in results if x.get("status") != "ok")
    }

    save_obj = {
        "metrics": metrics,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    print(f"\n[Checkpoint Saved] {output_path}")

def main():

    entity_path = "/home/wenbin.guo/RAG/data/FB15k-237/entity.json"
    relation_path = "/home/wenbin.guo/RAG/data/FB15k-237/relation.json"
    triple_path = "/home/wenbin.guo/RAG/data/FB15k-237/test2id.txt"

    evidence_path = "/home/wenbin.guo/DKGE4R/data/FB15k-237/test_key_evidence.json"
    output_path = "/home/wenbin.guo/DKGE4R/eval/eval_with_LP_results.json"

    entity_list = load_json(entity_path)
    relation_list = load_json(relation_path)
    triples = load_triples(triple_path)

    entity_dict = build_entity_dict(entity_list)
    relation_dict = build_relation_dict(relation_list)

    evidence_list = load_json(evidence_path)
    evidence_dict = {
        item["query_index"]: item["key_evidence"]
        for item in evidence_list
    }

    sampled_items = list(enumerate(triples))

    results = []
    hit1_count = 0
    hit3_count = 0
    hit10_count = 0

    for idx, (query_index, triple) in enumerate(
        tqdm(sampled_items, desc="Evaluating Hit@10", ncols=100),
        start=1
    ):
        key_evidence = evidence_dict.get(query_index, "")

        try:
            prediction = predict_for_triple(
                triple,
                entity_dict,
                relation_dict,
                key_evidence
            )
        except Exception as e:
            prediction = {
                "query_index": query_index,
                "triple_id": triple,
                "key_evidence": key_evidence,
                "llm_prediction": None,
                "pred_indices": None,
                "predicted_tails": [],
                "is_hit@1": 0,
                "is_hit@3": 0,
                "is_hit@10": 0,
                "status": f"failed_exception: {repr(e)}"
            }

        prediction["query_index"] = query_index
        results.append(prediction)

        hit1_count += prediction.get("is_hit@1", 0)
        hit3_count += prediction.get("is_hit@3", 0)
        hit10_count += prediction.get("is_hit@10", 0)

        if idx % 100 == 0:
            total_done = len(results)
            current_hit1 = hit1_count / total_done if total_done else 0.0
            current_hit3 = hit3_count / total_done if total_done else 0.0
            current_hit10 = hit10_count / total_done if total_done else 0.0

            metrics = {
                "total": total_done,
                "hit1_count": hit1_count,
                "hit3_count": hit3_count,
                "hit10_count": hit10_count,
                "Hit@1": current_hit1,
                "Hit@3": current_hit3,
                "Hit@10": current_hit10,
                "failed_count": sum(1 for x in results if x.get("status") != "ok")
            }

            save_obj = {
                "metrics": metrics,
                "results": results
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(save_obj, f, ensure_ascii=False, indent=2)

            print(f"\n===== Processed {idx}/{len(sampled_items)} samples =====")
            print(f"Current Hit@1 : {current_hit1:.4f}")
            print(f"Current Hit@3 : {current_hit3:.4f}")
            print(f"Current Hit@10: {current_hit10:.4f}")
            print(f"Failed Count  : {metrics['failed_count']}")
            print("Last Query Index:", query_index)
            print("Status:", prediction.get("status", "ok"))
            print("Last Triple:", prediction.get("triple"))
            print("Pred Indices:", prediction.get("pred_indices"))
            print("Pred Tails:", prediction.get("predicted_tails"))
            print("True Tail:", prediction.get("triple", [None, None, None])[2])
            print("Hit@1:", prediction.get("is_hit@1"))
            print("Hit@3:", prediction.get("is_hit@3"))
            print("Hit@10:", prediction.get("is_hit@10"))
            print(f"[Checkpoint Saved] {output_path}")
            print("=" * 60)

    final_hit1 = hit1_count / len(results) if results else 0.0
    final_hit3 = hit3_count / len(results) if results else 0.0
    final_hit10 = hit10_count / len(results) if results else 0.0

    metrics = {
        "total": len(results),
        "hit1_count": hit1_count,
        "hit3_count": hit3_count,
        "hit10_count": hit10_count,
        "Hit@1": final_hit1,
        "Hit@3": final_hit3,
        "Hit@10": final_hit10,
        "failed_count": sum(1 for x in results if x.get("status") != "ok")
    }

    save_obj = {
        "metrics": metrics,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)

    print(f"\nFinal Hit@1 : {final_hit1:.4f}")
    print(f"Final Hit@3 : {final_hit3:.4f}")
    print(f"Final Hit@10: {final_hit10:.4f}")
    print(f"Total samples: {len(results)}")
    print(f"Failed Count : {metrics['failed_count']}")
    print(f"Evaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()