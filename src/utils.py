# OD-KGC/src/utils.py

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from openai import OpenAI


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
    """
    Minimal cleanup for raw LLM output.

    This preserves JSON punctuation such as:
        [], {}, "", commas, colons

    Use this for:
        - entity class inference
        - JSON list output
        - candidate index output
    """

    text = extract_after_answer(text)
    text = remove_thinking_tags(text)
    return text.strip()


def clean_final_answer(text: str) -> str:
    """
    Strong cleanup for final textual answers.

    This will normalize punctuation and case, so do not use it for JSON.
    """

    text = clean_raw_llm_output(text)
    text = normalize_answer(text)
    return text


# ============================================================
# JSON helpers
# ============================================================

def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Try to parse JSON from LLM output.

    It supports outputs like:
        ```json
        [...]
        ```
    or text containing a JSON object/list.
    """

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
    """
    Build standard OpenAI chat messages.
    """

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
    """
    OpenAI-compatible LLM client.

    This class does NOT read OPENAI_API_KEY or OPENAI_BASE_URL from env.
    You must pass them explicitly.

    It supports:
        - OpenAI official API
        - vLLM OpenAI-compatible API
        - other OpenAI-compatible local servers

    Example:
        llm = LLM_Model(
            llm_model="Qwen/Qwen2.5-7B-Instruct",
            openai_api_key="EMPTY",
            openai_base_url="http://localhost:8000/v1",
        )

        messages = build_messages("Please answer yes or no.")
        answer = llm.infer(messages)
    """

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
        """
        Return the raw model output with minimal cleanup.

        This function preserves JSON punctuation such as:
            [], {}, "", commas, colons

        Use this for:
            - entity class inference
            - ontology class list generation
            - candidate index prediction
            - JSON-formatted outputs
        """

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
        """
        Return normalized final answer.

        This function removes punctuation and lowercases text.
        Do NOT use this if you need JSON or candidate index format.
        """

        content = self.infer_raw(messages, **kwargs)
        return normalize_answer(content)

    def infer_text(
        self,
        messages: Messages,
        **kwargs,
    ) -> str:
        """
        Return cleaned natural language text without answer normalization.

        Compared with infer():
            - keeps punctuation
            - keeps capitalization
            - removes only thinking tags and 'Answer:' prefix
        """

        return self.infer_raw(messages, **kwargs)

    def infer_json(
        self,
        messages: Messages,
        default: Any = None,
        **kwargs,
    ) -> Any:
        """
        Ask the LLM and parse output as JSON.

        If parsing fails, return default.
        """

        content = self.infer_raw(messages, **kwargs)
        return safe_json_loads(content, default=default)

    def infer_index(
        self,
        messages: Messages,
        default: Optional[int] = None,
        **kwargs,
    ) -> Optional[int]:
        """
        Extract a single integer index from LLM output.

        Useful for indexed candidate ranking.
        """

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
        """
        Extract multiple integer indices from LLM output.

        Useful if the model returns a ranked candidate list.
        """

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
        llm_model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
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