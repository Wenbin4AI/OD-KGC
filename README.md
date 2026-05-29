# OD-KGC

Official implementation of **OD-KGC: Ontology-Guided Evidence Reasoning for Model-Agnostic Knowledge Graph Completion**.

OD-KGC is a fine-tuning-free framework for LLM-based knowledge graph completion (KGC). Instead of directly verbalizing raw triples or injecting KG embeddings into the token space of an LLM, OD-KGC transforms knowledge graph structure into compact, query-relevant, and ontology-consistent textual evidence. The generated evidence is then used in an indexed candidate-ranking prompt, enabling frozen LLMs to perform link prediction without task-specific fine-tuning.

## Overview

Knowledge graph completion aims to infer missing facts from incomplete knowledge graphs. Existing LLM-based KGC methods often rely on raw triple prompting, model-specific adapters, token-level alignment, or fine-tuning. OD-KGC provides a lightweight and model-agnostic alternative by treating structured knowledge as external reasoning evidence.

The framework consists of four main stages:

1. **RotatE-guided structural evidence extraction**  
   Query-relevant one-hop neighbors and multi-hop paths are extracted from the local knowledge graph.

2. **Ontology-aware semantic calibration**  
   Relation schemas, entity classes, and domain/range constraints are used to refine structural evidence.

3. **Compact textual evidence construction**  
   High-scoring neighbors and paths are converted into concise textual evidence while redundant information is removed.

4. **Fine-tuning-free LLM reasoning**  
   The final prompt is formulated as an indexed candidate-ranking task, so the LLM only needs to output candidate indices.

## Repository Structure

```text
OD-KGC/
├── KGE_model/
│   ├── models/              # Saved KGE checkpoints
│   ├── build_rela.py         # Relation and ontology preprocessing utilities
│   ├── build_subkg.py        # KGE-based subgraph construction utilities
│   ├── dataloader.py         # Data loader for KGE training
│   ├── eval.py               # KGE evaluation utilities
│   ├── get_model.py          # Model loading utilities
│   ├── model.py              # KGE model definitions
│   └── run.py                # KGE training and testing entry point
├── data/
│   ├── FB15k-237/            # FB15k-237 dataset and ontology files
│   └── WN18RR/               # WN18RR dataset and ontology files
├── eval/                     # Evaluation outputs or auxiliary evaluation files
├── build_evidence.py         # Generate key evidence for test queries
├── build_onto_query.py       # Build ontology-aware prompts from extracted subgraphs
├── evaluation_LP.py          # LLM-based link prediction evaluation
├── llm.py                    # LLM calling utilities
├── predict.py                # Candidate-based LLM prediction script
└── test_one.py               # Single-example testing script
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Wenbin4AI/OD-KGC.git
cd OD-KGC
```

Create a Python environment:

```bash
conda create -n odkgc python=3.10
conda activate odkgc
```

Install the required packages:

```bash
pip install torch numpy tqdm scikit-learn openai
```

If you use a locally deployed LLM, make sure it provides an OpenAI-compatible API endpoint. The current scripts can be adapted to call either local or remote LLM services through the OpenAI Python client.

## Data Preparation

The repository supports two widely used KGC benchmarks:

- `FB15k-237`
- `WN18RR`

Each dataset directory is expected to contain standard KGC split files and ontology-related files. Typical files include:

```text
train.txt
valid.txt
test.txt
train2id.txt
valid2id.txt
test2id.txt
entities.dict
relations.dict
entity.json
relation_new.json
```

Please check and adjust the file paths in the scripts before running. Some scripts may contain absolute paths used in the original experimental environment. For example:

```python
DATA_DIR = "/home/wenbin.guo/DKGE4R/data/FB15k-237"
```

You should replace them with your local project paths:

```python
DATA_DIR = "./data/FB15k-237"
```

## Quick Start

### 1. Train a RotatE KGE Model

The RotatE model is used to provide structural relevance signals for evidence extraction.

For FB15k-237:

```bash
python KGE_model/run.py \
  --do_train \
  --do_valid \
  --do_test \
  --cuda \
  --data_path ./data/FB15k-237 \
  --model RotatE \
  --save_path ./KGE_model/models/RotatE_FB15k-237
```

For WN18RR:

```bash
python KGE_model/run.py \
  --do_train \
  --do_valid \
  --do_test \
  --cuda \
  --data_path ./data/WN18RR \
  --model RotatE \
  --save_path ./KGE_model/models/RotatE_WN18RR
```

### 2. Build Key Evidence

After obtaining the trained KGE model, run `build_evidence.py` to generate key evidence for each query. This step extracts query-relevant structural evidence and prepares it for ontology-guided LLM reasoning.

From the project root directory:

```bash
cd /home/wenbin.guo/OD-KGC
python build_evidence.py
```

Please check the path configuration in `build_evidence.py` before running, including the dataset path, model checkpoint path, and output directory. For example:

```python
DATA_DIR = "./data/FB15k-237"
MODEL_PATH = "./KGE_model/models/RotatE_FB15k-237"
OUTPUT_DIR = "./eval/evidence/FB15k-237"
```

For WN18RR, replace the corresponding paths with:

```python
DATA_DIR = "./data/WN18RR"
MODEL_PATH = "./KGE_model/models/RotatE_WN18RR"
OUTPUT_DIR = "./eval/evidence/WN18RR"
```

### 3. Run LLM-Based Prediction and Evaluation

After key evidence has been generated, use the scripts in the `eval/` directory to perform LLM-based prediction and link prediction evaluation.

For example:

```bash
cd /home/wenbin.guo/OD-KGC
python eval/<prediction_script>.py
```

Please replace `<prediction_script>.py` with the corresponding evaluation file in the `eval/` directory. These scripts use the generated evidence, candidate entities, and ontology information to construct indexed candidate-ranking prompts for LLM-based inference.

If you use a local OpenAI-compatible LLM server, configure the API endpoint in the corresponding evaluation script:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:22014/v1",
    api_key="EMPTY"
)
```

The LLM is required to output candidate indices rather than entity names, which are then used to compute standard link prediction metrics such as MRR and Hits@K.

## Evaluation Metrics

OD-KGC follows standard link prediction metrics:

- **MRR**
- **Hits@1**
- **Hits@3**
- **Hits@10**

For candidate-based LLM reasoning, the LLM outputs ranked candidate indices, which are compared with the ground-truth entity index.

## Reported Results

The following results are reported in the paper:

| Dataset | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|
| WN18RR | 0.618 | 0.579 | 0.623 | 0.710 |
| FB15k-237 | 0.473 | 0.409 | 0.506 | 0.607 |

These results show that ontology-calibrated structural evidence improves LLM-based KGC while keeping the LLM frozen.

## Key Features

- Fine-tuning-free LLM-based KGC
- RotatE-guided query-specific evidence extraction
- Ontology-aware semantic calibration
- Compact textual evidence construction
- Indexed candidate-ranking prompt
- Support for FB15k-237 and WN18RR
- Compatible with local OpenAI-style LLM APIs

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{guo2026odkgc,
  title   = {From Graph Structure to Reasoning Evidence: Ontology-Guided Distillation for Model-Agnostic Knowledge Graph Completion},
  author  = {Guo, Wenbin and Li, Zhao and Wang, Xin and Chen, Zirui},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Please refer to the license file of this repository. If no license is provided, please contact the authors before using the code for commercial purposes.