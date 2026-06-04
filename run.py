from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str | Path = "config.py") -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"[Config] No config.py found at {config_path}, using defaults/CLI args.")
        return {}

    spec = importlib.util.spec_from_file_location("odkgc_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {}
    for key in dir(module):
        if key.startswith("_"):
            continue
        value = getattr(module, key)
        if not callable(value):
            config[key.lower()] = value

    print(f"[Config] Loaded config from {config_path}")
    return config


def cfg_get(config: Dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key.lower(), default)


def run_command(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("[Run]", " ".join(map(str, cmd)))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def file_exists(path: str | Path) -> bool:
    return Path(path).exists() and Path(path).is_file()


def add_bool_cuda(cmd: list[str], cuda: bool, gpu_id: int | None = None) -> None:
    if cuda:
        cmd.append("--cuda")
        if gpu_id is not None:
            cmd.extend(["--gpu_id", str(gpu_id)])
    else:
        cmd.append("--no_cuda")


def parse_args():
    config = load_config("config.py")

    parser = argparse.ArgumentParser(description="Run full OD-KGC pipeline.")

    parser.add_argument("--data_path", type=str, default=cfg_get(config, "data_path", "dataset/FB15k-237"))
    parser.add_argument("--dataset_name", type=str, default=cfg_get(config, "dataset_name", None))
    parser.add_argument("--import_path", type=str, default=cfg_get(config, "import_path", "import"))
    parser.add_argument("--split", type=str, default=cfg_get(config, "split", "test"))

    parser.add_argument("--max_queries", type=int, default=cfg_get(config, "max_queries", -1))
    parser.add_argument("--top_k_one_hop", type=int, default=cfg_get(config, "top_k_one_hop", 10))
    parser.add_argument("--top_k_paths", type=int, default=cfg_get(config, "top_k_paths", 10))
    parser.add_argument("--max_hops", type=int, default=cfg_get(config, "max_hops", 2))
    parser.add_argument("--max_branch_per_node", type=int, default=cfg_get(config, "max_branch_per_node", 20))
    parser.add_argument("--max_paths_before_ranking", type=int, default=cfg_get(config, "max_paths_before_ranking", 1000))

    parser.add_argument("--filter_mode", type=str, default=cfg_get(config, "filter_mode", "no_llm"))
    parser.add_argument("--parallel_workers_filter", type=int, default=cfg_get(config, "parallel_workers_filter", 8))

    parser.add_argument("--max_evidence_num", type=int, default=cfg_get(config, "max_evidence_num", 5))
    parser.add_argument("--token_budget", type=int, default=cfg_get(config, "token_budget", 1500))

    parser.add_argument("--candidate_mode", type=str, default=cfg_get(config, "candidate_mode", "filtered_rotate"))
    parser.add_argument("--candidate_size", type=int, default=cfg_get(config, "candidate_size", 20))
    parser.add_argument("--top_k", type=int, default=cfg_get(config, "top_k", 10))
    parser.add_argument("--parallel_workers_eval", type=int, default=cfg_get(config, "parallel_workers_eval", 4))
    parser.add_argument("--save_every", type=int, default=cfg_get(config, "save_every", 50))
    parser.add_argument("--max_items", type=int, default=cfg_get(config, "max_items", -1))

    parser.add_argument("--llm_model", type=str, default=cfg_get(config, "llm_model", "/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B"))
    parser.add_argument("--openai_api_key", type=str, default=cfg_get(config, "openai_api_key", "EMPTY"))
    parser.add_argument("--openai_base_url", type=str, default=cfg_get(config, "openai_base_url", "http://localhost:22014/v1"))

    parser.add_argument("--cuda", action="store_true", default=cfg_get(config, "cuda", True))
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--gpu_id", type=int, default=cfg_get(config, "gpu_id", 0))

    parser.add_argument("--extractor_script", type=str, default=cfg_get(config, "extractor_script", "model/extractor.py"))
    parser.add_argument("--filter_script", type=str, default=cfg_get(config, "filter_script", "model/filter.py"))
    parser.add_argument("--compressor_script", type=str, default=cfg_get(config, "compressor_script", "model/compressor.py"))
    parser.add_argument("--evaluator_script", type=str, default=cfg_get(config, "evaluator_script", "src/evaluator.py"))
    parser.add_argument("--kge_script", type=str, default=cfg_get(config, "kge_script", "model/KGE_model.py"))

    parser.add_argument("--force_all", action="store_true", default=False)
    parser.add_argument("--force_kge", action="store_true", default=False)
    parser.add_argument("--force_extract", action="store_true", default=False)
    parser.add_argument("--force_filter", action="store_true", default=False)
    parser.add_argument("--force_compress", action="store_true", default=False)
    parser.add_argument("--force_eval", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.force_all:
        args.force_kge = True
        args.force_extract = True
        args.force_filter = True
        args.force_compress = True
        args.force_eval = True

    data_path = Path(args.data_path)
    dataset_name = args.dataset_name or data_path.name
    import_root = Path(args.import_path)

    kge_checkpoint = import_root / "KGE_model" / dataset_name / "checkpoint.pt"

    structural_evidence_path = (
        import_root / "evidence" / dataset_name / f"{args.split}_evidence.jsonl"
    )

    filtered_evidence_path = (
        import_root / "evidence" / dataset_name / f"{args.split}_filtered_evidence.jsonl"
    )

    compressed_evidence_path = (
        import_root / "evidence" / dataset_name / f"{args.split}_compressed_evidence.jsonl"
    )

    eval_output_path = (
        import_root / "eval" / dataset_name / f"{args.split}_llm_qa_eval_parallel.json"
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

    if args.force_kge or not file_exists(kge_checkpoint):
        cmd = [
            sys.executable,
            args.kge_script,
            "--data_path", str(data_path),
            "--dataset_name", dataset_name,
            "--import_path", str(import_root / "KGE_model"),
        ]
        add_bool_cuda(cmd, args.cuda, args.gpu_id)
        run_command(cmd)
    else:
        print(f"[Skip] Found KGE checkpoint: {kge_checkpoint}")

    if args.force_extract or not file_exists(structural_evidence_path):
        cmd = [
            sys.executable,
            args.extractor_script,
            "--data_path", str(data_path),
            "--dataset_name", dataset_name,
            "--import_path", str(import_root),
            "--split", args.split,
            "--max_queries", str(args.max_queries),
            "--top_k_one_hop", str(args.top_k_one_hop),
            "--top_k_paths", str(args.top_k_paths),
            "--max_hops", str(args.max_hops),
            "--max_branch_per_node", str(args.max_branch_per_node),
            "--max_paths_before_ranking", str(args.max_paths_before_ranking),
        ]
        add_bool_cuda(cmd, args.cuda, args.gpu_id)
        run_command(cmd)
    else:
        print(f"[Skip] Found structural evidence: {structural_evidence_path}")

    if args.force_filter or not file_exists(filtered_evidence_path):
        cmd = [
            sys.executable,
            args.filter_script,
            "--data_path", str(data_path),
            "--dataset_name", dataset_name,
            "--input_evidence_path", str(structural_evidence_path),
            "--output_evidence_path", str(filtered_evidence_path),
            "--filter_mode", args.filter_mode,
            "--parallel_workers", str(args.parallel_workers_filter),
            "--llm_model", args.llm_model,
            "--openai_api_key", args.openai_api_key,
            "--openai_base_url", args.openai_base_url,
        ]
        run_command(cmd)
    else:
        print(f"[Skip] Found filtered evidence: {filtered_evidence_path}")

    if args.force_compress or not file_exists(compressed_evidence_path):
        cmd = [
            sys.executable,
            args.compressor_script,
            "--data_path", str(data_path),
            "--dataset_name", dataset_name,
            "--split", args.split,
            "--import_path", str(import_root),
            "--input_filtered_path", str(filtered_evidence_path),
            "--output_path", str(compressed_evidence_path),
            "--max_evidence_num", str(args.max_evidence_num),
            "--token_budget", str(args.token_budget),
        ]
        run_command(cmd)
    else:
        print(f"[Skip] Found compressed evidence: {compressed_evidence_path}")

    if args.force_eval or not file_exists(eval_output_path):
        cmd = [
            sys.executable,
            args.evaluator_script,
            "--data_path", str(data_path),
            "--compressed_evidence_path", str(compressed_evidence_path),
            "--output_path", str(eval_output_path),
            "--llm_model", args.llm_model,
            "--openai_api_key", args.openai_api_key,
            "--openai_base_url", args.openai_base_url,
            "--candidate_size", str(args.candidate_size),
            "--candidate_mode", args.candidate_mode,
            "--top_k", str(args.top_k),
            "--parallel_workers", str(args.parallel_workers_eval),
            "--save_every", str(args.save_every),
            "--max_items", str(args.max_items),
        ]
        run_command(cmd)
    else:
        print(f"[Skip] Found evaluation result: {eval_output_path}")

    print("\n" + "=" * 100)
    print("[Run] OD-KGC pipeline finished.")
    print("=" * 100)
    print(f"Filtered evidence: {filtered_evidence_path}")
    print(f"Compressed evidence: {compressed_evidence_path}")
    print(f"Evaluation result: {eval_output_path}")


if __name__ == "__main__":
    main()

# python run.py --force_extract --force_filter --force_compress --force_eval