# llm_evaluate_full.py
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader
from src.utils import LLM_Model, CandidateBuilder
from model.KGE_model import get_or_train_rotate

# ---------------- Utility functions ----------------
def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def set_random_seed(seed: int) -> None:
    random.seed(seed)

def clean_llm_output(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def extract_json_object(text: str) -> Optional[str]:
    if text is None:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]

def extract_selected_indices(llm_output: str, k: int, num_candidates: int) -> Optional[List[int]]:
    if llm_output is None:
        return None
    cleaned = clean_llm_output(str(llm_output))
    extracted = extract_json_object(cleaned)
    if extracted:
        cleaned = extracted
    try:
        data = json.loads(cleaned)
        indices = data.get("selected_indices") or data.get("ranked_indices") or data.get("indices")
        if isinstance(indices, list):
            parsed = []
            for item in indices:
                try:
                    idx = int(item)
                    if 0 <= idx < num_candidates and idx not in parsed:
                        parsed.append(idx)
                    if len(parsed) >= k:
                        break
                except:
                    continue
            if parsed:
                return parsed[:k]
    except:
        pass
    numbers = re.findall(r"\d+", cleaned)
    parsed = []
    for x in numbers:
        idx = int(x)
        if 0 <= idx < num_candidates and idx not in parsed:
            parsed.append(idx)
        if len(parsed) >= k:
            break
    return parsed[:k] if parsed else None

def complete_topk_indices(pred_indices: Optional[List[int]], num_candidates: int, k: int) -> List[int]:
    results = []
    if pred_indices:
        for idx in pred_indices:
            if 0 <= idx < num_candidates and idx not in results:
                results.append(idx)
            if len(results) >= k:
                return results[:k]
    for idx in range(num_candidates):
        if idx not in results:
            results.append(idx)
        if len(results) >= k:
            break
    return results[:k]

def compute_rank_metrics(ranked_entity_ids: List[int], gold_tail_id: int) -> Dict[str, Any]:
    gold_tail_id = int(gold_tail_id)
    if gold_tail_id in ranked_entity_ids:
        rank = ranked_entity_ids.index(gold_tail_id) + 1
    else:
        rank = None
    if rank is None:
        return {"rank": None, "MRR":0.0, "Hit@1":0, "Hit@3":0, "Hit@10":0}
    return {"rank": rank, "MRR":1.0/rank, "Hit@1":int(rank<=1), "Hit@3":int(rank<=3), "Hit@10":int(rank<=10)}

# ---------------- LLM Evaluator ----------------
class LLMQAEvaluator:
    def __init__(self, dataset, rotate_manager, llm: LLM_Model, candidate_builder: CandidateBuilder, top_k:int=10, max_retries:int=2, retry_sleep:float=2.0):
        self.dataset = dataset
        self.rotate = rotate_manager
        self.llm = llm
        self.top_k = top_k
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.candidate_builder = candidate_builder

    def evaluate_one_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        query = record["query"]
        answer = record["correct_answer"]
        head_id = int(query["head_id"])
        relation_id = int(query["relation_id"])
        gold_tail_id = int(answer["tail_id"])
        head_label = query["head_label"]
        relation_label = query["relation_label"]
        head_classes = query.get("head_classes", [])
        relation_range = query.get("relation_range", [])

        # 生成候选实体
        candidates, _ = self.candidate_builder.build_tail_candidates(head_id, relation_id, gold_tail_id)

        # 生成辅助证据
        prompt_aux = f"""
Generate ONE short natural-language evidence sentence for KG completion.

Query:
- Head entity: {head_label}
- Relation: {relation_label}
- Head classes: {','.join(head_classes)}
- Expected tail type/range: {','.join(relation_range)}

Return ONLY JSON: {{"evidence": "sentence"}}.
""".strip()

        aux_output = self.query_llm_raw(prompt_aux)
        aux_evidence = self.parse_auxiliary_evidence(aux_output)

        # 将辅助证据追加到 evidence_text
        evidence_text = record.get("evidence_text","")
        if aux_evidence:
            if evidence_text:
                evidence_text += "\n"
            evidence_text += f"[LLM-generated evidence] {aux_evidence}"

        # ranking prompt
        prompt_ranking = f"Rank candidates 0..{len(candidates)-1} for query ({head_label}, {relation_label}, ?). Top {self.top_k}. Evidence:\n{evidence_text}\nCandidates:\n" + "\n".join([f"{i}. {c['label']}" for i,c in enumerate(candidates)]) + "\nReturn JSON with selected_indices."
        ranking_output = self.query_llm_raw(prompt_ranking)
        pred_indices_raw = extract_selected_indices(ranking_output, self.top_k, len(candidates))
        pred_indices_final = complete_topk_indices(pred_indices_raw, len(candidates), self.top_k)

        ranked_entity_ids = [candidates[i]["entity_id"] for i in pred_indices_final]
        metrics = compute_rank_metrics(ranked_entity_ids, gold_tail_id)
        status = "ok" if pred_indices_raw else "failed_parse_filled_by_default_order"

        return {
            "query_index": record.get("query_index"),
            "pred_indices_final": pred_indices_final,
            "ranked_entity_ids": ranked_entity_ids,
            **metrics,
            "status": status,
            "llm_generated_evidence": aux_evidence,
            "evidence_text_with_llm": evidence_text
        }

    def query_llm_raw(self, prompt: str) -> str:
        for attempt in range(self.max_retries+1):
            try:
                response = self.llm.openai_client.chat.completions.create(
                    **self.llm.llm_config,
                    messages=[
                        {"role":"system","content":"Answer strictly in JSON."},
                        {"role":"user","content":prompt},
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[LLM ERROR] attempt={attempt}, error={repr(e)}", flush=True)
                time.sleep(self.retry_sleep)
        return ""

    @staticmethod
    def parse_auxiliary_evidence(raw_output:str) -> str:
        if raw_output is None:
            return ""
        cleaned = clean_llm_output(raw_output)
        extracted = extract_json_object(cleaned)
        if extracted:
            try:
                data = json.loads(extracted)
                value = data.get("evidence")
                if isinstance(value, str) and value.strip():
                    return value.strip()
            except Exception as e:
                print(f"[AUX PARSE ERROR] {repr(e)} | raw={raw_output}", flush=True)
        return ""

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/FB15k-237")
    parser.add_argument("--compressed_evidence_path", type=str, default="import/evidence/FB15k-237/test_compressed_evidence.jsonl")
    parser.add_argument("--output_path", type=str, default="import/eval/FB15k-237/test_llm_qa_eval_parallel.json")
    parser.add_argument("--llm_model", type=str, default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:22014/v1")
    parser.add_argument("--candidate_size", type=int, default=20)
    parser.add_argument("--candidate_mode", type=str, default="filtered_rotate")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--parallel_workers", type=int, default=16)
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    set_random_seed(2026)

    # load dataset
    loader = KGLoader(args.data_path)
    dataset = loader.load()

    # load compressed evidence
    records = load_jsonl(args.compressed_evidence_path)
    if args.max_items > 0:
        records = records[:args.max_items]

    # load rotate
    rotate = get_or_train_rotate(args.data_path, import_path="import/KGE_model", dataset_name=None, load_if_exists=True, force_train=False, cuda=True, gpu_id=0)

    # load llm
    llm = LLM_Model(args.llm_model, args.openai_api_key, args.openai_base_url)

    candidate_builder = CandidateBuilder(dataset, rotate, candidate_size=args.candidate_size, candidate_mode=args.candidate_mode)
    evaluator = LLMQAEvaluator(dataset, rotate, llm, candidate_builder, top_k=args.top_k)

    results = []
    save_every = args.save_every

    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_idx = {executor.submit(evaluator.evaluate_one_record, record): idx for idx, record in enumerate(records)}
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="LLM QA Evaluation", ncols=100):
            idx = future_to_idx[future]
            record = records[idx]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "query_index": record.get("query_index"),
                    "ranked_entity_ids": [],
                    "MRR": 0.0,
                    "Hit@1": 0,
                    "Hit@3": 0,
                    "Hit@10": 0,
                    "status": f"failed_exception:{repr(e)}",
                }
            results.append(result)
            if len(results) % save_every == 0:
                save_json(results, args.output_path)

    save_json(results, args.output_path)
    print(f"Saved evaluation results to {args.output_path}")

if __name__ == "__main__":
    main()