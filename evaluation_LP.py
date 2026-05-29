import os
import re
import json
import ast
import time
import random
import argparse
from typing import List, Dict, Any

from openai import OpenAI


def build_client(base_url: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_candidate_count_from_json(meta: Dict[str, Any]) -> int:
    if "selected_candidates" in meta and isinstance(meta["selected_candidates"], list):
        return len(meta["selected_candidates"])
    if "candidate_limit" in meta:
        return int(meta["candidate_limit"])
    raise ValueError("无法从 JSON 中解析候选数量。")


def parse_rank_list(text: str) -> List[int]:
    text = text.strip()

    # 尝试去掉 <think> 标签（非贪婪或未闭合情况也可）
    text_wo_think = re.sub(r"<think.*?>", "", text, flags=re.S | re.I)
    text_wo_think = re.sub(r"</think>", "", text_wo_think, flags=re.S | re.I).strip()

    for candidate_text in [text_wo_think, text]:
        if not candidate_text:
            continue

        # 尝试整体解析
        try:
            obj = ast.literal_eval(candidate_text)
            if isinstance(obj, list):
                return [int(x) for x in obj]
        except Exception:
            pass

        # 按行逐行查找列表
        for line in candidate_text.splitlines()[::-1]:  # 从后向前
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                try:
                    obj = ast.literal_eval(line)
                    if isinstance(obj, list):
                        return [int(x) for x in obj]
                except Exception:
                    continue

        # 再使用正则匹配最后一个列表
        matches = re.findall(r"\[[\s\d,]+\]", candidate_text)
        if matches:
            for candidate in reversed(matches):
                try:
                    obj = ast.literal_eval(candidate)
                    if isinstance(obj, list):
                        return [int(x) for x in obj]
                except Exception:
                    continue

    raise ValueError(f"无法解析模型输出为列表，原始输出为: {text[:500]}")


def normalize_rank_list(rank_list: List[int], expected_n: int) -> List[int]:
    valid = set(range(1, expected_n + 1))
    used = set()
    result = []

    for x in rank_list:
        if x in valid and x not in used:
            result.append(x)
            used.add(x)

    missing = [x for x in range(1, expected_n + 1) if x not in used]
    result.extend(missing)
    return result[:expected_n]


def extract_gold_rank_index(meta: Dict[str, Any]) -> int:
    if "query" not in meta:
        raise KeyError("JSON 中缺少 query 字段")
    if "gold_candidate_rank_index" not in meta["query"]:
        raise KeyError("JSON 中 query 下缺少 gold_candidate_rank_index 字段")

    gold_rank = int(meta["query"]["gold_candidate_rank_index"])
    return gold_rank


def validate_gold_with_candidates(meta: Dict[str, Any], gold_rank: int) -> None:
    """
    可选校验：
    检查 selected_candidates 中是否存在 is_gold=1 且其 rank_index 与 gold_rank 一致
    """
    selected = meta.get("selected_candidates", [])
    gold_items = [x for x in selected if int(x.get("is_gold", 0)) == 1]

    if len(gold_items) == 0:
        print("[Warning] selected_candidates 中没有发现 is_gold=1 的候选。")
        return

    if len(gold_items) > 1:
        raise ValueError("selected_candidates 中存在多个 is_gold=1，数据异常。")

    cand_gold_rank = int(gold_items[0]["rank_index"])
    if cand_gold_rank != gold_rank:
        raise ValueError(
            f"gold_candidate_rank_index={gold_rank} 与 selected_candidates 中 is_gold=1 的 rank_index={cand_gold_rank} 不一致"
        )


def query_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_sec: float = 300.0,
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        # max_tokens=max_tokens,
        # timeout=timeout_sec,
    )

    return resp.choices[0].message.content.strip()


def evaluate_one_sample(
    client: OpenAI,
    model_name: str,
    txt_path: str,
    json_path: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    prompt = load_text(txt_path)
    meta = load_json(json_path)

    candidate_count = infer_candidate_count_from_json(meta)
    gold_rank = extract_gold_rank_index(meta)
    validate_gold_with_candidates(meta, gold_rank)

    if not (1 <= gold_rank <= candidate_count):
        raise ValueError(
            f"gold_candidate_rank_index 越界: {gold_rank}, candidate_count={candidate_count}"
        )

    start = time.time()
    raw_output = query_model(
        client=client,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )
    latency = time.time() - start

    parsed_rank = parse_rank_list(raw_output)
    final_rank = normalize_rank_list(parsed_rank, candidate_count)

    rank_of_gold = final_rank.index(gold_rank) + 1

    result = {
        "txt_file": os.path.basename(txt_path),
        "json_file": os.path.basename(json_path),
        "gold_candidate_rank_index": gold_rank,
        "candidate_count": candidate_count,
        "model_output_raw": raw_output,
        "model_output_parsed": parsed_rank,
        "model_output_final": final_rank,
        "rank_of_gold": rank_of_gold,
        "hit@1": 1.0 if rank_of_gold <= 1 else 0.0,
        "hit@3": 1.0 if rank_of_gold <= 3 else 0.0,
        "hit@10": 1.0 if rank_of_gold <= 10 else 0.0,
        "mrr": 1.0 / rank_of_gold,
        "mr": float(rank_of_gold),
        "latency_sec": latency,
    }
    return result


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "count": 0,
            "Hit@1": 0.0,
            "Hit@3": 0.0,
            "Hit@10": 0.0,
            "MRR": 0.0,
            "MR": 0.0,
            "AvgLatencySec": 0.0,
        }

    return {
        "count": n,
        "Hit@1": sum(x["hit@1"] for x in results) / n,
        "Hit@3": sum(x["hit@3"] for x in results) / n,
        "Hit@10": sum(x["hit@10"] for x in results) / n,
        "MRR": sum(x["mrr"] for x in results) / n,
        "MR": sum(x["mr"] for x in results) / n,
        "AvgLatencySec": sum(x["latency_sec"] for x in results) / n,
    }


def collect_prompt_pairs(prompt_dir: str) -> List[Dict[str, str]]:
    pairs = []
    for name in os.listdir(prompt_dir):
        if not name.endswith(".txt"):
            continue

        txt_path = os.path.join(prompt_dir, name)
        json_path = os.path.join(prompt_dir, os.path.splitext(name)[0] + ".json")

        if os.path.exists(json_path):
            pairs.append({
                "txt_path": txt_path,
                "json_path": json_path
            })
        else:
            print(f"[Warning] 缺少同名 JSON，跳过: {txt_path}")

    return sorted(pairs, key=lambda x: x["txt_path"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on link prediction prompts")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v2"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:22014/v1"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256
    )
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=120.0
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="eval_results_100.json"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    pairs = collect_prompt_pairs(args.prompt_dir)
    if not pairs:
        raise RuntimeError(f"目录中没有找到可用的 txt/json 对: {args.prompt_dir}")

    sample_size = min(args.sample_size, len(pairs))
    sampled_pairs = random.sample(pairs, sample_size)

    print(f"总样本数: {len(pairs)}")
    print(f"本次抽样数: {sample_size}")

    client = build_client(args.base_url, args.api_key)

    results = []
    failed = []

    for i, pair in enumerate(sampled_pairs, 1):
        txt_path = pair["txt_path"]
        json_path = pair["json_path"]

        print(f"\n[{i}/{sample_size}] Evaluating {os.path.basename(txt_path)}")

        try:
            result = evaluate_one_sample(
                client=client,
                model_name=args.model_name,
                txt_path=txt_path,
                json_path=json_path,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
            )
            results.append(result)
            print(result)

            print(
                f"  rank_of_gold={result['rank_of_gold']}, "
                f"Hit@1={result['hit@1']}, "
                f"Hit@3={result['hit@3']}, "
                f"Hit@10={result['hit@10']}, "
                f"latency={result['latency_sec']:.2f}s"
            )

        except Exception as e:
            print(f"  [Failed] {e}")
            failed.append({
                "txt_file": os.path.basename(txt_path),
                "json_file": os.path.basename(json_path),
                "error": str(e),
            })

    summary = aggregate_metrics(results)

    output = {
        "config": vars(args),
        "summary": summary,
        "success_count": len(results),
        "failed_count": len(failed),
        "failed_samples": failed,
        "results": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n========== Final Metrics ==========")
    print(f"Evaluated Count : {summary['count']}")
    print(f"Hit@1          : {summary['Hit@1']:.4f}")
    print(f"Hit@3          : {summary['Hit@3']:.4f}")
    print(f"Hit@10         : {summary['Hit@10']:.4f}")
    print(f"MRR            : {summary['MRR']:.4f}")
    print(f"MR             : {summary['MR']:.4f}")
    print(f"AvgLatencySec  : {summary['AvgLatencySec']:.4f}")
    print(f"Failed Count   : {len(failed)}")
    print(f"Saved to       : {args.output_json}")


if __name__ == "__main__":
    main()