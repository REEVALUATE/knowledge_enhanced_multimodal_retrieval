# Knowledge-Enhanced Multimodal Retrieval 
This repository contains the experimental code, models, datasets, and implementation accompanying the ESWC 2026 In-Use Track submission:

**“Knowledge-Enhanced Multimodal Retrieval over Cultural Heritage Knowledge Graphs”**  

The goal of this repository is to fully disclose the algorithms, training code, evaluation pipelines, and system modules used in the paper. The deployment-specific backend implementation of the CH retrieval system (including API endpoints, servers, and authentication configuration, etc.) is not released.

---

## 🔧 System Overview

Our proposed system integrates:

1. **A domain-adaptive multimodal retrieval module (CLIP)**  
2. **A Text2SPARQL module for knowledge reasoning over a Cultural Heritage Knowledge Graph**  

These two modules are combined through a weighted fusion strategy to support diverse queries.

### 📌 System Architecture

The full system architecture is shown below, illustrating the multimodal CLIP retrieval and the LLM-based Text2SPARQL reasoning modules.  
![System Architecture](architecture.svg)

---

## 📦 Released Assets

This repository publicly releases **all reproducible components** used in the paper:

### ✔️ **1. Dataset: Image–Text Pairs**
Used for CLIP fine-tuning and benchmarking (generated from the [KG](https://loki.linksfoundation.com/reevaluate-graphdb/graphs) ). 

Dataset includes:
- Artefact image  
- Automatically generated description text  
- Synthetic user-like query text  

🔗 **https://huggingface.co/datasets/xuemduan/reevaluate-image-text-pairs**

(Contains ~43k image–description–query triplets used in experiments.)

---

### ✔️ **2. Fine-Tuned CLIP Model**

We release the CLIP ViT-L/14 domain-adapted model used in the retrieval system.

🔗 **https://huggingface.co/xuemduan/reevaluate-clip**

This model supports both:
- **Text-to-Image (T2I)** retrieval  
- **Text-to-Text (T2T)** retrieval  

and is the backbone of the multimodal component.

---

### ✔️ **3. Source Code for All Experiments**

This repository includes the full implementation of:

- CLIP fine-tuning  
- CLIP evaluation (T2I, T2T, T2I and T2T fused evaluation)  
- Text2SPARQL pipeline  
- Text2SPARQL evaluation
- Knowledge-enhanced fusion strategy 
- Knowledge-enhanced fusion evaluation   

---

## 🧱 Repository Structure

```
src/
│
├── clip/ 
│ ├── data/ # Dataset loader & preprocessing 
│ ├── models/ # CLIP model load
│ ├── training/ # Fine-tuning scripts
│ ├── eval/ # T2I/T2T evaluation, metrics
│ ├── utils/ # logging
│ └── ... #
│
├── text2sparql/
│ ├── entity_linking.py # SPARQL-based entity resolution
│ ├── json2sparql.py # Python re-implementation of Sparnatural AI logic
│ └── text2sparql_retrieval.py # KG querying utilities
│
scripts/ 
│ ├── fine-tuning/ # CLIP fine-tuning script
│ ├── baselines/ # CLIP Baselines Performance
│ ├── fusion/ # CLIP fusion evaluation
│ └── ...
├── retrieval.py # final knowledge-enhanced multimodal retrieval script
```

## Usage

See individual experiment folders in `scripts/` for specific running instructions.


### Text2SPARQL Module Notes
The Text2SPARQL component is:

- **Inspired by Sparnatural AI**  
  https://github.com/sparna-git/sparnatural-ai  
- Fully **re-implemented in Python** for compatibility with our backend  
- Uses a multi-stage pipeline (LLM → JSON → Entity Linking → SPARQL)  
- Instruction prompts are adapted to match the SHACL configuration of our CH KG  
- Our own internal **Mistral Agent** is not released, but users may deploy their own agent using the provided prompt templates.
---

## 🚀 Usage Examples

### ⚙️ **1. Baselines**

```bash
bash scripts/baselines/run_clip_base_b32.sh

bash scripts/baselines/run_clip_base_l14.sh
```

### ⚙️ **1. Fine-tune CLIP**

```bash
bash scripts/fine-tuning/train.sh

bash scripts/fine-tuning/eval.sh
```

### ⚙️ **2. Fusion evaluation**

```bash
bash scripts/fusion/eval.sh
```

### ⚙️ **3. Retrieval (CLIP+Text2SPARQL)**

```
from src.retrieval import RetrievalEngine
retrieval = RetrievalEngine()
```

## 📝 License

This repository is released under the MIT License.