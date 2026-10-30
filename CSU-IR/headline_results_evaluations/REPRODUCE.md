# Step-by-Step Reproduction Guide for Headline Results

This document provides a comprehensive guide and command-line entry points to reproduce the core results and downstream analyses reported in our manuscript using the unified evaluation script: `run_headline_evaluations.py`.

---

## 1. Tasks Introduction

| Task Name | Paper Reference | Checkpoint Availability | Data Availability & Access |
| :--- | :---: | :---: | :--- |
| **100M library Retrieval** | Fig. 4 | **Public** | **Public** (automatic downloadable encoded index & test queries) |
| **Functional groups detection** | Fig. 6a | **Public** | **Public** (automatic downloadable test spectra & structures) |
| **OOD performance** | Fig. 6b | **Public** | *User-prepared* (Requires Thermo data, which can be accessed through standard procurement procedures. The specific OOD data needs to be processed firstly to extract the TF data from the test set in EB_TF. Then, using OOD building script (CSU-IR/OOD_data_building/build_OOD_data.py) to obtain the corresponding OOD data.) |
| **NPS retrieval** | Fig. 6c | **Public** | *Partially restricted* (Existing library public and automatic downloadable; derived sets can be accessed upon reasonable request) |

---

## 2 Execution Commands

Prior to execution, please note that commands for other tasks will automatically download the publicly available data and model weights, except for Option B of the 100M library retrieval, which entails a time-consuming local data download. For the OOD benchmark and NPS derivative identification tests, please refer to the table above to prepare the data beforehand to obtain the corresponding results.

### Task 1: 100-Million-Scale Compound Retrieval

#### Option A: Quick smoke test (100K compounds, finishes in ~10 seconds to verify environment)

```bash
python run_headline_evaluations.py \
    --task retrieval_100m \
    --gpu_ids 0 \
    --test_descr ./data/EB/test_smiles_augmented.txt \
    --test_ir ./data/EB/test_ir.pt \
    --db_dir ./encoded_features_54_EB_TF_seed_4200
```

#### Option B: Full headline benchmark (100 Million compounds, single-GPU supported)

> Powered by precomputed hash/MW binary caches, full 100M retrieval runs efficiently on a single consumer GPU.

```bash
python run_headline_evaluations.py \
    --task retrieval_100m \
    --eval_full_scale \
    --gpu_ids 0 \
    --test_descr ./data/EB/test_smiles_augmented.txt \
    --test_ir ./data/EB/test_ir.pt \
    --db_dir ./encoded_features_54_EB_TF_seed_4200
```

---

### Task 2: Functional Group Detection 

```bash
python run_headline_evaluations.py \
    --task functional_groups \
    --test_descr ./data/EB/test_smiles_augmented.txt \
    --test_ir ./data/EB/test_ir.pt \
    --ir_weights ./weights/overall_best_top1_ir.pth \
    --smiles_weights ./weights/overall_best_top1_smiles.pth
```

---

### Task 3: Out-of-Distribution (OOD) Benchmark

```bash
python run_headline_evaluations.py \
    --task ood \
    --ood_dir ./data/OOD_with_EB/
```

---

### Task 4: New Psychoactive Substances (NPS) Identification

```bash
python run_headline_evaluations.py \
    --task ps \
    --ps_query_smiles ./data/PS/query_smiles.txt \
    --ps_query_ir ./data/PS/query_ir.pt
```

---

### Run All Tasks in Sequence

```bash
python run_headline_evaluations.py --task all
```
