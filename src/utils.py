from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from openai import OpenAI

from typing import Any, Dict, List, Set, Tuple
import random

class CandidateBuilder:
    def __init__(
        self,
        dataset,
        rotate_manager,
        candidate_size: int = 20,
        candidate_mode: str = "filtered_rotate",
        exclude_head: bool = True,
        random_seed: int = 2026,
    ):
        if candidate_mode not in {"filtered_rotate", "random"}:
            raise ValueError("candidate_mode must be filtered_rotate or random.")

        self.dataset = dataset
        self.rotate = rotate_manager
        self.candidate_size = candidate_size
        self.candidate_mode = candidate_mode
        self.exclude_head = exclude_head
        self.random_seed = random_seed
        self.true_tail_map = self._build_true_tail_map(dataset)

    def _build_true_tail_map(self, dataset) -> Dict[Tuple[int,int],Set[int]]:
        true_tail_map: Dict[Tuple[int,int],Set[int]] = {}
        all_triples = []
        for attr in ["train_triples", "valid_triples", "test_triples"]:
            if hasattr(dataset, attr):
                all_triples.extend(getattr(dataset, attr))
        for tri in all_triples:
            h = int(tri.h_id)
            r = int(tri.r_id)
            t = int(tri.t_id)
            true_tail_map.setdefault((h,r),set()).add(t)
        return true_tail_map

    def build_tail_candidates(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
        query_index: int = None,
    ) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
        if self.candidate_mode == "filtered_rotate":
            return self._build_filtered_rotate_candidates(head_id, relation_id, gold_tail_id)
        else:
            return self._build_random_candidates(head_id, relation_id, gold_tail_id, query_index)

    def _prepare_pools(
        self,
        head_id: int,
        relation_id: int,
        gold_tail_id: int,
    ) -> Dict[str,Any]:
        all_entity_ids = set(int(eid) for eid in self.dataset.entities.keys())
        all_true_tails = set(self.true_tail_map.get((head_id, relation_id), set()))
        other_true_tails = all_true_tails.copy()
        other_true_tails.discard(gold_tail_id)
        candidate_pool = all_entity_ids - other_true_tails
        if self.exclude_head and head_id in candidate_pool:
            candidate_pool.remove(head_id)
        negative_pool = candidate_pool.copy()
        negative_pool.discard(gold_tail_id)
        return {"all_true_tails": all_true_tails,
                "other_true_tails": other_true_tails,
                "candidate_pool": candidate_pool,
                "negative_pool": negative_pool}

    def _build_filtered_rotate_candidates(
        self,
        head_id:int,
        relation_id:int,
        gold_tail_id:int
    ) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
        head_id = int(head_id)
        relation_id = int(relation_id)
        gold_tail_id = int(gold_tail_id)
        pools = self._prepare_pools(head_id,relation_id,gold_tail_id)
        negative_pool_list = sorted(list(pools["negative_pool"]))
        scored_negatives = self.rotate.score_tail_candidates(head_id,relation_id,negative_pool_list)
        num_negatives = max(0,self.candidate_size-1)
        selected_negatives = scored_negatives[:num_negatives]
        score_dict = {int(eid):float(score) for eid,score in selected_negatives}
        gold_score = float(self.rotate.score_triples([(head_id,relation_id,gold_tail_id)])[0])
        score_dict[gold_tail_id] = gold_score
        sorted_items = sorted(score_dict.items(),key=lambda x:x[1],reverse=True)[:self.candidate_size]
        candidates = self._format_candidates(sorted_items,gold_tail_id,pools["other_true_tails"])
        info = {"candidate_mode":"filtered_rotate",
                "candidate_size":len(candidates),
                "gold_tail_id":gold_tail_id,
                "gold_candidate_index":next(i for i,item in enumerate(candidates) if item["is_gold"]),
                "gold_rotate_score":gold_score,
                "num_all_true_tails_for_query":len(pools["all_true_tails"]),
                "num_filtered_other_true_tails":len(pools["other_true_tails"]),
                "num_negative_pool":len(pools["negative_pool"])}
        return candidates, info

    def _build_random_candidates(
        self,
        head_id:int,
        relation_id:int,
        gold_tail_id:int,
        query_index:int=None
    ) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
        head_id = int(head_id)
        relation_id = int(relation_id)
        gold_tail_id = int(gold_tail_id)
        pools = self._prepare_pools(head_id,relation_id,gold_tail_id)
        negative_pool = sorted(list(pools["negative_pool"]))
        num_negatives = max(0,self.candidate_size-1)
        seed = self.random_seed if query_index is None else self.random_seed+int(query_index)
        rng = random.Random(seed)
        if len(negative_pool)<num_negatives:
            negatives=list(negative_pool)
        else:
            negatives=rng.sample(negative_pool,num_negatives)
        candidate_ids = negatives+[gold_tail_id]
        rng = random.Random(seed+999999)
        rng.shuffle(candidate_ids)
        triples=[(head_id,relation_id,int(eid)) for eid in candidate_ids]
        scores=self.rotate.score_triples(triples)
        items=[(int(eid),float(score)) for eid,score in zip(candidate_ids,scores)]
        gold_score = next((score for eid,score in items if eid==gold_tail_id), None)
        candidates = self._format_candidates(items,gold_tail_id,pools["other_true_tails"])
        info = {"candidate_mode":"random",
                "candidate_size":len(candidates),
                "gold_tail_id":gold_tail_id,
                "gold_candidate_index":next(i for i,item in enumerate(candidates) if item["is_gold"]),
                "gold_rotate_score":gold_score,
                "num_all_true_tails_for_query":len(pools["all_true_tails"]),
                "num_filtered_other_true_tails":len(pools["other_true_tails"]),
                "num_negative_pool":len(pools["negative_pool"])}
        return candidates, info

    def _format_candidates(self,items:List[Tuple[int,float]],gold_tail_id:int,other_true_tails:Set[int])->List[Dict[str,Any]]:
        candidates=[]
        for idx,(entity_id,score) in enumerate(items):
            candidates.append({"index":idx,
                               "entity_id":int(entity_id),
                               "label":str(self.dataset.entities[int(entity_id)].label),
                               "classes":[],
                               "rotate_score":score,
                               "is_gold":int(entity_id)==int(gold_tail_id),
                               "is_other_true_tail":int(entity_id) in other_true_tails})
        return candidates


Message = Dict[str, str]
Messages = List[Message]


# ============================================================
# Text normalization
# ============================================================

def normalize_answer(text: str) -> str:
    """
    Normalize LLM output for answer matching.

    This function is mainly used for final answer/entity text.
    Do NOT use it when you need to preserve JSON punctuation.
    """

    if text is None:
        return ""

    text = str(text).strip()

    # Lowercase
    text = text.lower()

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def remove_thinking_tags(text: str) -> str:
    """
    Remove common thinking/tool tags while preserving useful output text.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text, flags=re.DOTALL)

    return text.strip()


def extract_after_answer(text: str) -> str:
    """
    Extract content after 'Answer:' if it exists.
    """

    if text is None:
        return ""

    text = str(text)

    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    if "answer:" in text:
        text = text.split("answer:")[-1]

    return text.strip()


def clean_raw_llm_output(text: str) -> str:
    text = extract_after_answer(text)
    text = remove_thinking_tags(text)
    return text.strip()


def clean_final_answer(text: str) -> str:
    text = clean_raw_llm_output(text)
    text = normalize_answer(text)
    return text



def safe_json_loads(text: str, default: Any = None) -> Any:
    if default is None:
        default = None

    if text is None:
        return default

    text = clean_raw_llm_output(text)

    # Remove markdown code fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```", "", text).strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting JSON list
    list_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if list_match:
        try:
            return json.loads(list_match.group(0))
        except Exception:
            pass

    # Try extracting JSON object
    obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    return default


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    results = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    return results


# ============================================================
# Prompt helpers
# ============================================================

def build_messages(
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> Messages:

    messages: Messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return messages


# ============================================================
# LLM Client
# ============================================================

class LLM_Model:

    def __init__(
        self,
        llm_model: str,
        openai_api_key: str,
        openai_base_url: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        timeout: float = 60.0,
        trust_env: bool = False,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        if not openai_api_key:
            raise ValueError(
                "openai_api_key is required. "
                "For local vLLM, you can pass openai_api_key='EMPTY'."
            )

        self.llm_model = llm_model
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url

        self.http_client = httpx.Client(
            timeout=timeout,
            trust_env=trust_env,
        )

        client_kwargs = {
            "api_key": openai_api_key,
            "http_client": self.http_client,
        }

        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url

        self.openai_client = OpenAI(**client_kwargs)

        self.llm_config: Dict[str, Any] = {
            "model": llm_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if top_p is not None:
            self.llm_config["top_p"] = top_p

        if frequency_penalty is not None:
            self.llm_config["frequency_penalty"] = frequency_penalty

        if presence_penalty is not None:
            self.llm_config["presence_penalty"] = presence_penalty

        if extra_body is not None:
            self.llm_config["extra_body"] = extra_body

    def close(self) -> None:
        """
        Close the underlying httpx client.
        """

        try:
            self.http_client.close()
        except Exception:
            pass

    def infer_raw(
        self,
        messages: Messages,
        **kwargs,
    ) -> str:

        request_config = dict(self.llm_config)
        request_config.update(kwargs)

        response = self.openai_client.chat.completions.create(
            **request_config,
            messages=messages,
        )

        content = response.choices[0].message.content

        if content is None:
            return ""

        return clean_raw_llm_output(content)

    def infer(
        self,
        messages: Messages,
        **kwargs,
    ) -> str:

        content = self.infer_raw(messages, **kwargs)
        return normalize_answer(content)

    def infer_text(
        self,
        messages: Messages,
        **kwargs,
    ) -> str:

        return self.infer_raw(messages, **kwargs)

    def infer_json(
        self,
        messages: Messages,
        default: Any = None,
        **kwargs,
    ) -> Any:

        content = self.infer_raw(messages, **kwargs)
        return safe_json_loads(content, default=default)

    def infer_index(
        self,
        messages: Messages,
        default: Optional[int] = None,
        **kwargs,
    ) -> Optional[int]:

        content = self.infer_raw(messages, **kwargs)

        match = re.search(r"-?\d+", content)
        if match is None:
            return default

        return int(match.group(0))

    def infer_indices(
        self,
        messages: Messages,
        **kwargs,
    ) -> List[int]:

        content = self.infer_raw(messages, **kwargs)
        return [int(x) for x in re.findall(r"-?\d+", content)]

    def __enter__(self) -> "LLM_Model":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":
    # Example for local vLLM OpenAI-compatible server.
    # Change these according to your environment.
    llm = LLM_Model(
        llm_model="Qwen/Qwen3-8B",
        openai_api_key="EMPTY",
        openai_base_url="http://localhost:22014/v1",
        max_tokens=512,
        temperature=0,
    )

    messages = build_messages(
        user_prompt="Answer with only one word: yes",
        system_prompt="You are a helpful assistant.",
    )

    print(llm.infer_raw(messages))
    llm.close()