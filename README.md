# Knowledge-Enhanced Multimodal Retrieval over Cultural Heritage Knowledge Graphs

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
Used for CLIP fine-tuning and benchmarking.

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
- CLIP evaluation (T2I, T2T, fused evaluation)  
- Synthetic dataset usage  
- Text2SPARQL pipeline  
- Knowledge-enhanced fusion evaluation  
- System-level evaluation scripts  

---

## 🧱 Repository Structure

```
src/
│
├── clip/
│ ├── data/ # Dataset loader & preprocessing
│ ├── models/ # CLIP wrapper, projection layers, fusion logic
│ ├── training/ # Fine-tuning scripts (InfoNCE, mixed losses)
│ ├── eval/ # T2I/T2T evaluation, metrics
│ ├── utils/ # Checkpointing, logging, config handling
│ └── ... # (Auto-discovered on local filesystem)
│
├── text2sparql/
│ ├── entity_linking/ # SPARQL-based entity resolution
│ ├── json2sparql/ # Python re-implementation of Sparnatural AI logic
│ └── text2sparql_retrieval/ # KG querying utilities
│
scripts/
│ ├── train_clip.sh # CLIP fine-tuning script
│ ├── eval_clip.sh # Batch evaluation scripts
│ ├── run_text2sparql.sh # Text2SPARQL evaluation
│ ├── run_fusion.sh # Combined evaluation
│ └── ...
│
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

### ⚙️ **1. Fine-tune CLIP**

```bash
bash scripts/train_clip.sh \
  --dataset ./data/reevaluate \
  --epochs 20 \
  --lr 5e-6 \
  --batch 64 \
  --model ViT-L-14
```

## 📝 License

This repository is released under the MIT License.