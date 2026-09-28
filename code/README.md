# DASE7506 MP1 — Small Language Model Challenge

**Author:** He Jiaxin
**Student ID:** 3036748919
**Repository:** [star4599/DASE7506-MP1-He_Jiaxin](https://github.com/star4599/DASE7506-MP1-He_Jiaxin)  
**Protocol:** `7506-mp1-wt2-v2` | **Device:** CPU (FP32)

---

## 1. Experimental Results

Evaluated on WikiText-2 with FP32 precision under strict resource limits (CPU time $\le 5\times$ baseline, Peak RAM $< 4\text{ GiB}$, Checkpoint $< 64\text{ MiB}$).

| Model Variant | Layers | Width | Heads | Params | Val BPB | Test BPB | CPU Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Baseline** | 4 | 128 | 4 | ~1.09M | 2.1027 | 2.1027 | ~5.92s |
| **Proposed Full Model** | **8** | **256** | **8** | **~2.12M** | **1.5841** | **1.6177** | **28.70s** |
| *w/o RoPE* (Absolute Pos) | 8 | 256 | 8 | ~2.12M | 1.7351 | - | 20.81s |
| *w/o SwiGLU* (GELU MLP) | 8 | 256 | 8 | ~1.85M | 1.6305 | - | 23.41s |
| *w/o Weight Tying* (Untied Head) | 8 | 256 | 8 | ~2.64M | 1.5846 | - | 24.10s |

*Compliance check: Peak RAM < 0.4 GiB, Checkpoint size ~8.5 MiB (PASS).*

---

## 2. Running & Evaluation Commands

### Environment Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### Train & Evaluate

```bash
# Train proposed model
python train.py --implementation student --device cuda --steps 3600 --eval-every 300 --seed 17 --run-dir runs/proposed-full

# Evaluate on Validation / Test splits
python evaluate.py --checkpoint runs/proposed-full/checkpoint.pt --device cpu --precision fp32 --split validation
python evaluate.py --checkpoint runs/proposed-full/checkpoint.pt --device cpu --precision fp32 --split test
```

---

## 3. Reproduction Instructions for Peer Review

To evaluate this frozen checkpoint without retraining:

```bash
python evaluate.py --checkpoint runs/proposed-full/checkpoint.pt --device cpu --precision fp32 --split test
```

---

## 4. AI Assistance Statement

AI tools (Gemini / ChatGPT) were used for brainstorming model optimizations (SwiGLU, RoPE), debugging PyTorch tensor shapes, and assisting with report formatting. All final code, training runs, and evaluations were conducted independently.
