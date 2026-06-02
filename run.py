import os
from pathlib import Path

from config import CONFIG

def main():
    import subprocess

    kge_path = Path(CONFIG["kge_checkpoint_path"])
    if not kge_path.exists():
        print("[Run] KGE checkpoint not found. Training KGE...")
        subprocess.run([
            "python", "model/KGE_model.py",
            "--data_path", CONFIG["data_path"],
            "--dataset_name", CONFIG["dataset_name"],
            "--import_path", CONFIG["import_path"],
            "--cuda"
        ], check=True)
    else:
        print(f"[Run] Found KGE checkpoint at {kge_path}")

    compressed_path = Path(CONFIG["compressed_evidence_path"])
    if not compressed_path.exists():
        print("[Run] Compressed evidence not found. Generating...")
        subprocess.run([
            "python", "model/compressor.py",
            "--data_path", CONFIG["data_path"],
            "--import_path", CONFIG["import_path"],
            "--max_evidence_num", str(CONFIG["max_evidence_num"]),
            "--token_budget", str(CONFIG["token_budget"]),
            "--cuda"
        ], check=True)
    else:
        print(f"[Run] Found compressed evidence at {compressed_path}")

    print("[Run] Running LLM QA evaluation...")
    subprocess.run([
        "python", "src/evaluator.py",
        "--data_path", CONFIG["data_path"],
        "--compressed_evidence_path", CONFIG["compressed_evidence_path"],
        "--output_path", CONFIG["eval_output_path"],
        "--candidate_size", str(CONFIG["candidate_size"]),
        "--candidate_mode", CONFIG["candidate_mode"],
        "--top_k", str(CONFIG["top_k"]),
        "--parallel_workers", str(CONFIG["parallel_workers"]),
        "--save_every", str(CONFIG["save_every"]),
        "--llm_model", CONFIG["llm_model"],
        "--openai_api_key", CONFIG["openai_api_key"],
        "--openai_base_url", CONFIG["openai_base_url"],
        "--max_items", str(CONFIG["max_items"])
    ], check=True)

    print(f"[Run] Evaluation finished. Results saved at {CONFIG['eval_output_path']}")

if __name__ == "__main__":
    main()