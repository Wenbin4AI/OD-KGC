from evaluation_LP import build_client, evaluate_one_sample

client = build_client(base_url="http://localhost:22014/v1", api_key="EMPTY")

result = evaluate_one_sample(
    client=client,
    model_name="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    txt_path="/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v2/test_0_prompt.txt",
    json_path="/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v2/test_0_prompt.json",
    temperature=0.0,
    max_tokens=2048,
    timeout_sec=300.0,
)

print(result)