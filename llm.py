# from openai import OpenAI
# client = OpenAI(
#     base_url="http://localhost:22014/v1",
#     api_key="EMPTY"   # vLLM 允许随便填
# )

# resp = client.chat.completions.create(
#     model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
#     messages=[{"role": "user", "content": "Hello everone!"}]
# )

# print(resp.choices[0].message.content)


# run_llm_test.py
from openai import OpenAI

# ====== 配置 LLM 客户端 ======
client = OpenAI(
    base_url="http://localhost:22014/v1",
    api_key="EMPTY"  # vLLM 可以随便填
)

# ====== 读取 prompt 文件 ======
prompt_path = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v2/test_0_prompt.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()

# ====== 调用模型 ======
resp = client.chat.completions.create(
    model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    # max_tokens=2048,   # 增大输出长度
    # timeout=300        # 增加超时，防止长输出被截断
)

# ====== 打印原始输出 ======
print(resp.choices[0].message.content)