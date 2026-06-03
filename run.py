from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("[Run]", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def file_exists(path: str | Path) -> bool:
    return Path(path).exists() and Path(path).is_file()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full OD-KGC pipeline."
    )

    parser.add_argument("--data_path", type=str, default="dataset/WN18RR")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--import_path", type=str, default="import")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--max_queries", type=int, default=-1)
    parser.add_argument("--top_k_one_hop", type=int, default=10)
    parser.add_argument("--top_k_paths", type=int, default=10)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_branch_per_node", type=int, default=20)

    parser.add_argument("--filter_mode", type=str, default="no_llm", choices=["no_llm", "precise"])
    parser.add_argument("--parallel_workers_filter", type=int, default=8)

    parser.add_argument("--max_evidence_num", type=int, default=5)
    parser.add_argument("--token_budget", type=int, default=800)

    parser.add_argument("--candidate_mode", type=str, default="filtered_rotate", choices=["filtered_rotate", "random"])
    parser.add_argument("--candidate_size", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--parallel_workers_eval", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--max_items", type=int, default=-1)

    parser.add_argument(
        "--llm_model",
        type=str,
        default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base_url", type=str, default="http://localhost:22014/v1")

    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--force_kge", action="store_true", default=False)
    parser.add_argument("--force_extract", action="store_true", default=False)
    parser.add_argument("--force_filter", action="store_true", default=False)
    parser.add_argument("--force_compress", action="store_true", default=False)
    parser.add_argument("--force_eval", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name
    import_root = Path(args.import_path)

    kge_checkpoint = import_root / "KGE_model" / dataset_name / "checkpoint.pt"

    structural_evidence_path = (
        import_root
        / "evidence"
        / dataset_name
        / f"{args.split}_evidence.jsonl"
    )

    filtered_evidence_path = (
        import_root
        / "filtered_evidence"
        / dataset_name
        / f"{args.split}_filtered_evidence_{args.filter_mode}.jsonl"
    )

    compressed_evidence_path = (
        import_root
        / "evidence"
        / dataset_name
        / f"{args.split}_compressed_evidence.jsonl"
    )

    eval_output_path = (
        import_root
        / "eval"
        / dataset_name
        / f"{args.split}_llm_qa_eval_parallel.json"
    )

    print("=" * 100)
    print("[OD-KGC Full Pipeline]")
    print("=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    print(f"KGE checkpoint: {kge_checkpoint}")
    print(f"Structural evidence: {structural_evidence_path}")
    print(f"Filtered evidence: {filtered_evidence_path}")
    print(f"Compressed evidence: {compressed_evidence_path}")
    print(f"Eval output: {eval_output_path}")
    print("=" * 100)

    # 1. KGE checkpoint
    if args.force_kge or not file_exists(kge_checkpoint):
        print("[Run] KGE checkpoint not found. Training/loading RotatE through KGE_model.py...")

        cmd = [
            sys.executable,
            "model/KGE_model.py",
            "--data_path",
            str(data_path),
            "--dataset_name",
            dataset_name,
            "--import_path",
            str(import_root / "KGE_model"),
        ]

        if args.cuda:
            cmd.extend(["--cuda", "--gpu_id", str(args.gpu_id)])
        else:
            cmd.append("--no_cuda")

        run_command(cmd)
    else:
        print(f"[Run] Found KGE checkpoint: {kge_checkpoint}")

    # 2. Structural evidence extraction
    if args.force_extract or not file_exists(structural_evidence_path):
        print("[Run] Structural evidence not found. Running extractor.py...")

        cmd = [
            sys.executable,
            "model/extractor.py",
            "--data_path",
            str(data_path),
            "--dataset_name",
            dataset_name,
            "--import_path",
            str(import_root),
            "--split",
            args.split,
            "--max_queries",
            str(args.max_queries),
            "--top_k_one_hop",
            str(args.top_k_one_hop),
            "--top_k_paths",
            str(args.top_k_paths),
            "--max_hops",
            str(args.max_hops),
            "--max_branch_per_node",
            str(args.max_branch_per_node),
            "--gpu_id",
            str(args.gpu_id),
        ]

        if args.cuda:
            cmd.append("--cuda")

        run_command(cmd)
    else:
        print(f"[Run] Found structural evidence: {structural_evidence_path}")

    # 3. Ontology filtering
    if args.force_filter or not file_exists(filtered_evidence_path):
        print("[Run] Filtered evidence not found. Running filter.py...")

        cmd = [
            sys.executable,
            "model/filter.py",
            "--data_path",
            str(data_path),
            "--dataset_name",
            dataset_name,
            "--input_evidence_path",
            str(structural_evidence_path),
            "--output_evidence_path",
            str(filtered_evidence_path),
            "--filter_mode",
            args.filter_mode,
            "--parallel_workers",
            str(args.parallel_workers_filter),
            "--llm_model",
            args.llm_model,
            "--openai_api_key",
            args.openai_api_key,
            "--openai_base_url",
            args.openai_base_url,
        ]

        run_command(cmd)
    else:
        print(f"[Run] Found filtered evidence: {filtered_evidence_path}")

    # 4. Evidence compression
    if args.force_compress or not file_exists(compressed_evidence_path):
        print("[Run] Compressed evidence not found. Running compressor.py...")

        cmd = [
            sys.executable,
            "model/compressor.py",
            "--data_path",
            str(data_path),
            "--dataset_name",
            dataset_name,
            "--split",
            args.split,
            "--import_path",
            str(import_root),
            "--input_filtered_path",
            str(filtered_evidence_path),
            "--output_path",
            str(compressed_evidence_path),
            "--max_evidence_num",
            str(args.max_evidence_num),
            "--token_budget",
            str(args.token_budget),
        ]

        run_command(cmd)
    else:
        print(f"[Run] Found compressed evidence: {compressed_evidence_path}")

    # 5. LLM evaluation
    if args.force_eval or not file_exists(eval_output_path):
        print("[Run] Evaluation result not found. Running evaluator.py...")

        cmd = [
            sys.executable,
            "src/evaluator.py",
            "--data_path",
            str(data_path),
            "--compressed_evidence_path",
            str(compressed_evidence_path),
            "--output_path",
            str(eval_output_path),
            "--llm_model",
            args.llm_model,
            "--openai_api_key",
            args.openai_api_key,
            "--openai_base_url",
            args.openai_base_url,
            "--candidate_size",
            str(args.candidate_size),
            "--candidate_mode",
            args.candidate_mode,
            "--top_k",
            str(args.top_k),
            "--parallel_workers",
            str(args.parallel_workers_eval),
            "--save_every",
            str(args.save_every),
            "--max_items",
            str(args.max_items),
        ]

        run_command(cmd)
    else:
        print(f"[Run] Found evaluation result: {eval_output_path}")

    print("\n" + "=" * 100)
    print("[Run] OD-KGC pipeline finished.")
    print("=" * 100)
    print(f"Compressed evidence: {compressed_evidence_path}")
    print(f"Evaluation result: {eval_output_path}")


if __name__ == "__main__":
    main()