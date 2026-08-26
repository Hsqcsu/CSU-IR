# **From spectra to structures with interpretable retrieval at 100-million scale**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Demo-blue.svg)](https://huggingface.co/spaces/Hsqcsu/CSU-IR-Web)

This is the official code repository for our paper, **"From spectra to structures with interpretable retrieval at 100-million scale"**.

We introduce **CSU-IR**, a novel deep learning framework designed for high-precision unknown-compound identification by unifying infrared (IR) spectra and molecular structures. Key strengths of our work include:

*   **Exceptional 100-Million-Scale Library Retrieval Performance**: Maintains high accuracy (Recall@1 of 63.11% and a Recall@10 of 90.74%) when retrieving against libraries containing 100 million compounds.
*   **Specialized for Psychoactive Substances Identification**: Purpose-built models and libraries for the accurate retrieval of Psychoactive Substances.
*   **Multi-perspective Interpretability**: CSU-IR achieves accurate and trustworthy spectral retrieval by grounding its predictions in interpretable chemical property and specific spectral-structural correspondence.
*   **Comprehensive Functional Groups Detection**: The model successfully detected 48 functional groups with an average recall@1 of 93.80%.

<img width="4575" height="3950" alt="Fig1_github" src="https://github.com/user-attachments/assets/b8e2f632-3a61-4840-aecb-c7bbe9200f06" />

## 🚀 Quick Start with Google Colab

Experience our models instantly, without any local setup. Our Colab notebooks handle all dependencies and data downloads automatically, allowing you to explore our models directly in your browser.

---

### **Testing**

| Notebook                               | Description                                                                       | Link                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CSU-IR Tesing**               | Explore the specialized retrieval results in CSU-IR.  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/test_CSU_IR_in_colab.ipynb)       |

---

### **Training**

| Notebook                               | Description                                                                       | Link                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Train CSU-IR**                 | Train the CSU-IR model with DFT data.         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/train_CSU_IR_in_colab.ipynb)                              |
> **Note:** 
> For full training, please see the `train` instructions in our `README`'s `Local Installation and Usage` section and download the complete dataset from our [Hugging Face Repository](https://huggingface.co/Hsqcsu/CSU-IR/tree/main).

## 💻 Local Installation and Usage

For users who wish to run the project locally, please follow these steps.

### 1. Using pip to set up environment 

#### ①. Clone the repository

```bash
git clone https://github.com/Hsqcsu/CSU-IR.git
```
#### ②. create CSU-IR environment

```bash
conda create --name CSU-IR python=3.11.9
conda activate CSU-IR
```

#### ③. Install the required packages using pip

```bash
cd CSU-IR
pip install -r requirements/requirements_local.txt
```
### 2. Data & Checkpoints Preparation 
#### 2.1 EB dataset Construction

The complete dataset with identical splits can be fully reproduced locally. Please follow the steps.

**Step 1:** Download the EB dataset (including the smiles txt file, compound labels txt file and ir pt file) without NIST IR data from **[Download Hub: Hugging Face Repository (Stage III- EXP / EB Folder)](https://huggingface.co/Hsqcsu/CSU-IR/tree/main/Multi-stage%20training%20data/Stage%20III-%20EXP/EB)**.

**Step 2:** Put all the downloaded files into our filefolder[`CSU-IR/data/EB_dataset/data_without_NIST_IR/`](https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/data/EB_dataset/data_without_NIST_IR).

**Step 3:** Run the data_reconstruction script [`CSU-IR/EB_dataset_construction/EB_data_reconstruction.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/EB_dataset_construction/EB_data_reconstruction.py). 

> [!NOTE]
>
>The whole construction will take ~8 hours，brief hangs are expected during the process.
>
>The EB_CHONF, PS and E2S subset can be constructed from our script in the folder [`CSU-IR/EB_dataset_construction`](https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/EB_dataset_construction). The experimental data for the E2S subset are based on EB_CHONF, which needs to be obtained from EB first. The DFT spectra used to construct the E2S subset via the corresponding script should be downloaded from the **[Download Hub: Hugging Face Repository (Stage II- DFT Folder)](https://huggingface.co/Hsqcsu/CSU-IR/tree/main/Multi-stage%20training%20data/Stage%20II-%20DFT)**.

> [!WARNING]
> 
> These scripts are provided solely for provenance tracking and reproducibility. Users remain responsible for complying with the NIST Chemistry WebBook (SRD 69) terms of use, institutional policies, and relevant legal requirements.
> 
> Such generated files are not covered by the license of our repository. This repository does not grant permission to redistribute, publish, mirror, sublicense, or commercially reuse generated WebBook-derived data files.

#### 2.2 EB-TF dataset Construction (Optional)

The EB-TF dataset incorporates Thermo Fisher (TF) data into the existing EB dataset. The Thermo Fisher database used to build the EB_TF dataset is a commercial proprietary resource, which can be accessed through standard procurement procedures from the vendor. To ensure reproducibility and verifiability, the molecule identifiers and source labels can be downloaded from  **[Download Hub: Hugging Face Repository](https://huggingface.co/Hsqcsu/CSU-IR)**. The preprocessing scripts of the Thermo Fisher data are given in [`CSU-IR/data_process/ir_process.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/data_process/ir_process.py), specifically utilizing the CSV-related processing functions.

#### 2.3 Other Data & Checkpoints Preparation

The other data used in our study (For example, the multi-stage pretraining data), candidate libraries and trained model weights are hosted on **[Download Hub: Hugging Face Repository](https://huggingface.co/Hsqcsu/CSU-IR)** for easy access. One must download these assets manually and place them into the corresponding directories as structured in this project to run the local scripts. 

### 3. Training

You can initiate training scripts using a configuration file. All configs are located in the configs/ directory. The training config of CSU-IR for SOTA comparison is also privided. Below we provide the main training process of CSU-IR (Multi-stage training, MD DFT EXP). The PS fine-tuning for downstream task is also provided, with training scripts located in the local_training/ directory and corresponding config files in the configs/ directory.

> Ensure you are in the project's root directory and your environment is activated.
> 
> Put the corresponding training data into the corresponding folder according to the config file.
> 
> It is strongly recommended to run this script in an IDE terminal (like PyCharm's) instead of the standard system terminal to avoid potential environment-related issues.

#### Train CSU-IR in Stage-I with Molecular Dynamics (MD) data.

```bash
python local_training/Multi-stage_training_CSU-IR_in_local.py --config configs/config_CSU-IR_Multi-stage_training_I_MD.yaml
```

#### Train CSU-IR in Stage-II with Density Functional Theory (DFT) data.

```bash
python local_training/Multi-stage_training_CSU-IR_in_local.py --config configs/config_CSU-IR_Multi-stage_training_II_DFT.yaml
```

#### Train CSU-IR in Stage-III with the EB data. 

```bash
python local_training/Multi-stage_training_CSU-IR_in_local.py --config configs/config_CSU-IR_Multi-stage_training_III_EB.yaml
```

### 4. Testing and Inference

Scripts for testing and inference are available in the respective project folders. These code snippets do not require terminal operations; users can run them directly within the code interface.

- **CSU-IR Retrieval**: [`CSU-IR/test_and_infer/`](https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/test_and_infer)

> **✨ Use custom Libraries!**  
> To perform retrieval against your own custom library, you can make it in our web server.
---

## 🌐 Web Service

We have developed an open-access retrieval platform for PS retrieval or general compound retrieval in small libraries or custom libraries.

**[➡️ Try the Live Demo Here!](https://huggingface.co/spaces/Hsqcsu/CSU-IR-Web)**

The demo includes:
- **General Retrieval**: Search against the NIST library (~1W).
- **PS Retrieval**: Specialized search against psychoactive substance libraries.
---
## 🌐 Local GUI of 100 million compounds retrieval

For 100-Million-Scale retrieval，We have provided a GUI for local usage.
Users need to download the processed 100-million-library-Retrieval library from **[Hugging Face](https://huggingface.co/datasets/Hsqcsu/CSU-IR_100_Million_library/tree/main)** and place it in the data/100-Million-library-Retrieval folder. Then, simply run [`Retrieval_GUI.py`](https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/100-Million-library-Retrieval/Retrieval_GUI.py) and click the link generated in the terminal to perform a 100-million-library retrieval.

The GUI includes:
- **IR Only**: Search against the 100-Million library using IR spectral signals alone.
- **IR + Molecular Weight**: Search against the 100-Million library using IR combined with molecular weight filtering.
- **IR + Molecular Formula**: : Search against the 100-Million library using IR combined with molecular formula filtering.
---

## 📦 Hardware Requirements

The experiments during Stage-I and Stage-II were conducted on a single NVIDIA GPU (RTX 4090). 
- **Stage-I (MD Data)**: ~33 hours
- **Stage-II (DFT Data)**: ~3 hours
The Stage-III experiment was optimized using three NVIDIA RTX 6000 Ada GPUs. The time required for a single complete experiment is
- **Stage-III (EXP Data)**: ~10 minutes

## 📬 Contact

We welcome any questions, suggestions, or collaboration opportunities. Please feel free to open an issue on GitHub or contact us via email.
- **Email**: `232307004@csu.edu.cn`

