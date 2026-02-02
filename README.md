# **CSU-IR: An Interpretable Deep Learning Framework for 100-Million-Scale Infrared Spectral Retrieval**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/YOUR_ARXIV_ID)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_all_in_colab.ipynb)  [![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Demo-blue.svg)](https://huggingface.co/spaces/Hsqcsu/CSU-IR-Web)

This is the official code repository for our paper, **"CSU-IR: An Interpretable Deep Learning Framework for 100-Million-Scale Infrared Spectral Retrieval"**.

We introduce **CSU-IR**, a novel deep learning framework designed for high-precision unknown-compound identification by unifying infrared (IR) spectra and molecular structures. Key strengths of our work include:

*   🚀 **Exceptional 100-Million-Scale Performance**: Maintains high accuracy (Recall@1 of 68.13% and a Recall@10 of 93.97%) when retrieving against libraries containing 100 million compounds.
*   🎯 **Specialized for Psychoactive Substances Identification**: Purpose-built models and libraries for the accurate retrieval and SMILES classification of Psychoactive Substances.
*   🔬 **Strong Interpretability**: The learned representations are highly interpretable, revealing a direct mapping between spectral features and molecular features.
*   🧩 **Chemical Substructures Detection**: The model successfully detected 83 chemical substructures, including common functional groups, with an average detection rate of 95.46%.

![Fig1](https://github.com/user-attachments/assets/163af346-e49d-494c-8889-fa1620fe0988)

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

### 2. Data & Checkpoints Download

All data files, processed libraries, and trained model weights are hosted on Hugging Face for easy access. You must download these assets manually and place them into the corresponding directories as structured in this project to run the local scripts. 
 **[Download Hub: Hugging Face Repository](https://huggingface.co/Hsqcsu/CSU-IR)**

### 3. Training

You can initiate training scripts using a configuration file. All configs are located in the configs/ directory.

> Ensure you are in the project's root directory and your environment is activated.
> Put the corresponding training data into the corresponding folder according to the config file.
> It is strongly recommended to run this script in an IDE terminal (like PyCharm's) instead of the standard system terminal to avoid potential environment-related issues.

#### Train CSU-IR in Stage-I with Molecular Dynamics (MD) data

```bash
python local_training/Multi-stage_training_CSU-IR_in_local.py --config configs/config_CSU-IR_Multi-stage_training_I_MD.yaml
```

#### Train CSU-IR in Stage-II with Density Functional Theory (DFT) data

```bash
python local_training/Multi-stage_training_CSU-IR_in_local.py --config configs/config_CSU-IR_Multi-stage_training_II_DFT.yaml
```

> **Note on Stage-III Training:**  
> The datasets used for Stage-III Training are subject to copyright and are not publicly released. However, the code and logic are provided for transparency. The Stage-III Training process mirrors the Stage-I or II Training scripts.
> We also provide more intuitive training scripts that do not require terminal operations [`CSU-IR/train_and_val/`](https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/train_and_val); users can run them directly in the code interface.
---

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
- **PS-SMILES-Classifier**: Classify substances using SMILES strings.

The performance and reliability of these tools are backed by extensive benchmarking. You can review the detailed results in the section below.

# 📊 Performance & Results

Here is a summary of our model's performance benchmarks.

| General Retrieval  | Top-1 Accuracy | Top-10 Accuracy |
| :-------------------------------: | :------------: | :-------------: |
|         Library Size: 1W          |     0.8000     |     0.9440     |

| NPS Retrieval                      | Top-1 Accuracy | Top-10 Accuracy |
| :------------------------------------------------: | :------------: | :-------------: |
|                 Existed PS Library                 |     0.7692     |     1.0000      |
| Derivative PS Library|     0.5000     |     0.9688      |

| PS-Classifier Accuracy (NPS Set) |  Accuracy  |
| :-----------------------------------: | :--------: |
|           SMILES-Classifier           |   92.31%   |

---
## 🌐 Local GUI of 100 million compounds retrieval

For 100-Million-Scale retrieval，We have provided a GUI for local usage.
Users need to download the processed 100-million-library-Retrieval library from **[Hugging Face](https://huggingface.co/datasets/Hsqcsu/CSU-IR)** and place it in the data/100-Million-library-Retrieval folder. Then, simply run [`Retrieval_GUI.py`](https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/100-Million-library-Retrieval/Retrieval_GUI.py) and click the link generated in the terminal to perform a 100-million-library retrieval.

The GUI includes:
- **IR Only**: Search against the 100-Million library using IR spectral signals alone.
- **IR + Molecular Weight**: Search against the 100-Million library using IR combined with molecular weight filtering.
- **IR + Molecular Formula**: : Search against the 100-Million library using IR combined with molecular formula filtering.

# 📊 Performance & Results

Here is a summary of our model's performance in Large-Scale library retrieval.

| IR Only  | Top-1 Accuracy | Top-10 Accuracy |
| :-------------------------------: | :------------: | :-------------: |
|          Library Size: 1M         |     0.5170     |     0.8110      |
|          Library Size: 10M         |     0.3860     |     0.6940      |
|          Library Size: 100M         |     0.1600     |     0.4380      |

| IR + Molecular Weight  | Top-1 Accuracy | Top-10 Accuracy |
| :-------------------------------: | :------------: | :-------------: |
|          Library Size: 100M         |     0.6110     |     0.8700     |

| IR + Molecular Formula  | Top-1 Accuracy | Top-10 Accuracy |
| :-------------------------------: | :------------: | :-------------: |
|          Library Size: 100M         |     0.6813     |     0.9397      |
---

## 📦 Hardware Requirements

All experiments were conducted on a single NVIDIA GPU (RTX 4090). The full training pipeline for CSU-IR requires approximately **37 hours**:
- **Stage-I (MD Data)**: ~33 hours
- **Stage-II (DFT Data)**: ~3 hours
- **Stage-III (ET Data)**: ~1 hours

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{YourLastName2024CSUIR,
  title   = {CSU-IR: An Interpretable Deep Learning Framework for 100-Million-Scale Infrared Spectral Retrieval},
  author  = {First Author and Second Author and ...},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```

## 🙏 Acknowledgements

*(This space is reserved for any acknowledgements you wish to add, e.g., funding sources, computational resources, or helpful discussions.)*

## 📬 Contact

We welcome any questions, suggestions, or collaboration opportunities. Please feel free to open an issue on GitHub or contact us via email.
- **Email**: `232307004@csu.edu.cn`

