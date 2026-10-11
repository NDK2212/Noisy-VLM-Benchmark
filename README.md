# Benchmarking Vision-Language Models for Detailed Localized Captioning Under Image Corruptions and Spatial Prompt Perturbations

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Status-Under%20Review-orange" alt="Paper Status"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/datasets/Khoale11hcmut/gpt4o_captions_1k5samples"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DAM--PACO%20Dataset-yellow" alt="HuggingFace Dataset"></a>
  <a href="https://www.kaggle.com/"><img src="https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-20BEFF?style=flat&logo=kaggle&logoColor=white" alt="Platform"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

---

## 📌 Overview

> 📝 **Publication Status**: *This work is currently submitted and under review at IEEE Access.*

**Detailed Localized Captioning (DLC)** tasks a Vision-Language Model (VLM) with generating fine-grained, comprehensive descriptions of a specific region of interest (RoI) while ignoring surrounding background context. 

Existing DLC benchmarks evaluate models exclusively on **pristine images** with **ground-truth spatial prompts**. In deployed settings, however:
1. **Sensors introduce visual degradation**: Noise, defocus/motion blur, illumination shifts, contrast drops, and lossy JPEG compression.
2. **Prompts are inherently imperfect**: Human clicks and upstream bounding-box detectors introduce spatial displacement (**shift**), context over-inclusion (**upscale**), or boundary truncation (**downscale**).

This repository hosts the official codebase and benchmark artifacts for the comprehensive robustness study evaluating **four localized VLMs** across **two benchmark datasets** under a systematic **dual-axis perturbation matrix** (7 image corruptions $\times$ 3 prompt perturbations $\times$ 3 severity levels).

---

## 🌟 Key Research Contributions

1. **Dual-Axis Robustness Benchmark**: A systematic evaluation framework assessing regional Vision-Language Models under both visual input degradations (7 ImageNet-C corruptions across 4 spectral families) and spatial prompt inaccuracies (shift, upscale, downscale) across multiple severities.
2. **Curated DAM-PACO Benchmark**: A dedicated test suite of 1,500 region–caption pairs spanning 305 object categories with verified zero training overlap, providing high-density GPT-4o reference descriptions.
3. **Diagnostic Output Metrics**: Introduction of four targeted metrics (**OIP**, **CSR**, **APR**, **HAR**) designed to dissect fine-grained failure modes—differentiating between object identity loss, contextual semantic drift, attribute decay, and hallucination.
4. **Empirical Robustness Insights**: Extensive empirical characterization uncovering model spectral sensitivity profiles, prompt geometric asymmetries, and the shielding efficacy of promptable segmentation adapters (SAM) in sequential inference pipelines.

---

## 🏛️ Benchmark Architecture

```
                                  [ DUAL-AXIS INPUT ]
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
   [ AXIS 1: IMAGE CORRUPTIONS ]                              [ AXIS 2: PROMPT PERTURBATIONS ]
   • Stochastic: Gaussian, Impulse                            • Shift:    Center displacement (±w*s, ±h*s)
   • Spatial Blur: Defocus, Motion                            • Upscale:  Context expansion (w*s, h*s; s > 1)
   • Luminance: Brightness, Contrast                          • Downscale: Boundary crop (w*s, h*s; s < 1)
   • Compression: JPEG (DCT quant.)                           (Severities: L1, L3, L5)
   (Severities: L1, L3, L5)
         │                                                                   │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                                 [ EVALUATED MODELS ]
         ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼                  ▼
    DAM (Mask)          DAM + SAM           ViP-LLaVA           Shikra            GPT4RoI
    (Mask-Based        (BBox -> SAM        (Visual Marker      (Coordinate        (RoIAlign
    Upper Bound)        -> Mask)            Red Box)            Tokens)           Pooling)
         │                  │                  │                  │                  │
         └──────────────────┴──────────────────┼──────────────────┴──────────────────┘
                                               │
                                               ▼
                                     [ EVALUATION SUITE ]
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
 [ Traditional NLP ]                   [ LLM Judges ]                     [ Proposed Metrics ]
 • BLEU-1 to BLEU-4                    • CLAIR (Azure GPT-4o)             • OIP (Identity)
 • METEOR / ROUGE-L                    • DLC-Bench Judge                  • CSR (Semantic Drift)
 • CIDEr / SPICE                         (Llama 3.1 8B: Pos/Neg/Avg)      • APR (Attribute Recall)
                                                                          • HAR (Hallucination)
```

---

## 📊 Summary of Evaluated Models & Datasets

### 1. Evaluated Models

| Model | Parameters | Region Conditioning | Prompt Interface |
| :--- | :---: | :--- | :--- |
| **DAM** (Describe Anything Model) | 3B | Mask-conditioned | Binary mask via focal prompt and localized ViT backbone |
| **DAM + SAM** (DAM BBox) | 3B + SAM | BBox $\to$ Mask adapter | Coarse box converted to mask via SAM (`vit-huge`) $\to$ DAM |
| **ViP-LLaVA** | 7B | Visual marker | Red bounding box rendered directly onto raw RGB image pixels |
| **Shikra** | 7B | Coordinate text tokens | Textual coordinate format `[ymin, xmin, ymax, xmax]` in prompt |
| **GPT4RoI** | 7B | RoI feature pooling | Spatial RoIAlign pooling over visual tokens referenced by `<region1>` |

### 2. Benchmark Datasets

| Dataset | Size | Categories | Reference / Annotation Type | Access |
| :--- | :---: | :---: | :--- | :--- |
| **DAM-PACO** | 1,500 regions | 305 | Detail-level GPT-4o captions (mean: 361.1 chars) | [HuggingFace Hub](https://huggingface.co/datasets/Khoale11hcmut/gpt4o_captions_1k5samples) |
| **DLC-Bench** | 100 images | 892 questions | Multiple-choice attribute QA pairs ($Q^+$ and $Q^-$) | Local Objects365 subset |

---

## 📂 Repository Structure

```
Noisy-VLM-Benchmark/
├── README.md                                  <- Comprehensive benchmark documentation
├── LICENSE                                    <- MIT License
├── requirements.txt                           <- Dependency specifications
│
├── Benchmark_Notebooks/                       <- Kaggle inference pipelines (Dual Tesla T4)
│   ├── Benchmark_DAM.ipynb                    <- DAM (Mask) & DAM + SAM (BBox) inference
│   ├── Benchmark_GPT4ROI.ipynb                <- GPT4RoI-7B RoIAlign pooling inference
│   ├── Benchmark_Shikra.ipynb                 <- Shikra-7B coordinate token inference
│   └── Benchmark_VIP-LLAVA.ipynb              <- ViP-LLaVA-7B visual marker inference
│
├── Benchmark_Evaluation/                      <- Metric calculation & LLM judge pipelines
│   ├── Benchmark-CLAIR.ipynb                  <- Azure OpenAI GPT-4o batch judging & lexical scores
│   ├── Benchmark-Traditional-Metrics.ipynb    <- BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE
│   └── Output_Level_Metrics/                  <- OIP, CSR, APR, HAR under spatial prompt perturbations
│       ├── extract.py / metrics.py            <- GPT-4o-mini extraction (cached) and metric computation
│       ├── calibrate.py / plots.py            <- Threshold calibration and figures
│       ├── data/ cache/ outputs/              <- Inputs, raw GPT responses, results (no API key needed)
│       └── README.md                          <- Full documentation
│
└── Benchmark_Visualisation/                   <- Publication-ready curves & benchmark matrices
    ├── DAM_PACO/                              <- Curves for CLAIR, CIDEr, SPICE, BLEU-4, METEOR, ROUGE-L
    │   ├── CLAIR/                             <- SAM margin, relative decay, spatial decay, etc.
    │   └── CIDEr/
    └── DLC_Bench/                             <- Curves for Avg, Pos, Neg, Recog
        ├── Avg/
        └── Recog/
```

### Reproducing the output-level metrics

The proposed metrics (OIP, CSR, APR, HAR), their threshold calibration, and the corresponding figures
can be reproduced on a CPU in a few minutes without an API key, because the GPT-4o-mini extractions are
cached in the repository:

```bash
cd Benchmark_Evaluation/Output_Level_Metrics
pip install -r requirements.txt
python extract.py && python metrics.py && python calibrate.py && python plots.py
```

See [`Benchmark_Evaluation/Output_Level_Metrics/README.md`](Benchmark_Evaluation/Output_Level_Metrics/README.md)
for the metric definitions, the extraction prompt, and the file formats.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
