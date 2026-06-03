CONFIG = {
    # Dataset & Paths
    "data_path": "dataset/WN18RR",
    "dataset_name": "WN18RR",
    "import_path": "import",
    "compressed_evidence_path": "import/evidence/WN18RR/test_compressed_evidence.jsonl",
    "kge_checkpoint_path": "import/KGE_model/WN18RR/checkpoint.pt",
    "eval_output_path": "import/eval/FB15k-237/test_llm_qa_eval_parallel.json",

    # KGE & Evidence
    "candidate_size": 20,
    "candidate_mode": "filtered_rotate",
    "top_k": 10,
    "token_budget": 800,
    "max_evidence_num": 5,
    "max_items": -1,
    "save_every": 50,

    # Parallel & Reproducibility
    "parallel_workers": 4,
    "random_seed": 2026,

    # LLM
    "llm_model": "YOUR LLM MODEL PATH OR NAME",
    "openai_api_key": "YOUR_API_KEY",
    "openai_base_url": "YOUR_OPENAI_BASE_URL",
    "max_tokens": 1024,
    "temperature": 0.0,
    "timeout": 120.0,

    # Other
    "debug": False,
}