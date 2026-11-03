# Step-by-Step Guide for Headline Results Reproduction

This document provides a comprehensive guide and command-line entry points to reproduce the core results and downstream analyses reported in our manuscript using the unified evaluation script: `run_headline_evaluations.py`.

---

## 1. Tasks Introduction

| Task Name | Paper Reference | Checkpoint Availability | Data Availability & Access |
| :--- | :---: | :---: | :--- |
| **100M library Retrieval** | Fig. 4 | **Public**, https://huggingface.co/Hsqcsu/CSU-IR  | **Public**. The EB test set needs to undergo the "2.1 EB dataset Construction" process as described in the readme file to obtain the corresponding data. |
| **Functional groups detection** | Fig. 6a | **Public**, https://huggingface.co/Hsqcsu/CSU-IR | **Public**. The EB test set needs to undergo the "2.1 EB dataset Construction" process as described in the readme file to obtain the corresponding data. |
| **OOD performance** | Fig. 6b | **Public**, https://huggingface.co/Hsqcsu/CSU-IR | *User-prepared* (Requires Thermo data, which can be accessed through standard procurement procedures. The specific OOD data needs to be processed firstly to extract the TF data from the test set in EB_TF. Then, using OOD building script (CSU-IR/OOD_data_building/build_OOD_data.py) to obtain the corresponding OOD data.) |
| **NPS retrieval** | Fig. 6c | **Public**, https://huggingface.co/Hsqcsu/CSU-IR | *Partially restricted* (Existing library is public in https://huggingface.co/Hsqcsu/CSU-IR; derived sets can be accessed upon reasonable request) |

---

## 2 Execution Commands

Prior to execution, please download the required resources from the following Hugging Face repositories and place them in the directories specified by the code(run_headline_evaluations.py)'s default paths. It is particularly important to note that the 100 million library of Task 1 are stored at https://huggingface.co/datasets/Hsqcsu/CSU-IR_100_Million_library, and the entire dataset needs to be downloaded to the folder named (data/100-Million-library-Retrieval). The execution time for all commands, including the retrieval from the 100-Million-Scale library, will be completed within a few minutes.


### Task 1: 100-Million-Scale Compound Retrieval

```bash
python run_headline_evaluations.py --task retrieval_100m --gpu_ids 0
```
---

### Task 2: Functional Group Detection 

```bash
python run_headline_evaluations.py --task functional_groups --gpu_ids 0
```
---

### Task 3: Out-of-Distribution (OOD) Benchmark

```bash
python run_headline_evaluations.py --task ood --gpu_ids 0
```
---

### Task 4: New Psychoactive Substances (NPS) Identification

```bash
python run_headline_evaluations.py --task ps --gpu_ids 0
```
---

### Run All Tasks in Sequence

```bash
python run_headline_evaluations.py --task all --gpu_ids 0
```
