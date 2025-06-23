# **CSU-IR: Interpretable Unification of Infrared Spectra and Molecular Structures for Enhanced Chemical Identification**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/YOUR_ARXIV_ID)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_all_in_colab.ipynb)  [![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Demo-blue.svg)](https://huggingface.co/spaces/Skylight666/CSU-IR-normal-compound-identification)

This is the official code repository for our paper, **"Interpretable Contrastively Spectral-structural Unification between infrared Spectra and Molecular Structures assisting Novel Psychoactive Substances identification"**.

We introduce **CSU-IR**, a novel deep learning framework designed for high-precision compound identification by unifying infrared (IR) spectra and molecular structures. Key strengths of our work include:

*   🚀 **Exceptional Million-Scale Performance**: Maintains high accuracy even when retrieving against libraries containing millions of compounds.
*   🎯 **Specialized for Psychoactive Substances**: Purpose-built models and libraries for the accurate retrieval and classification of Novel Psychoactive Substances (NPS).
*   🔬 **Strong Interpretability**: The model architecture is designed to provide insights into the spectral-structural correlations it learns.
*   🧩 **Versatile Applications**: The framework offers broad utility, featuring not only high-performance retrieval for general compounds but also a specialized search for psychoactive substances. Additionally, it includes dedicated classifiers for both IR spectra and SMILES strings, all of which are deployable through a user-friendly web interface.
  
![image](https://github.com/user-attachments/assets/9fa914cf-4c32-42a5-b594-ab4cd67941f4)

## 🚀 Quick Start with Google Colab

Experience the full power of our models instantly, without any local setup. Our Colab notebooks handle all dependencies and data downloads automatically.

| Notebook                               | Description                                                                       | Link                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interactive Demo & All Tests**       | Explore all test results interactively. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_all_in_colab.ipynb)                               |
| **Train CSU-IR Model**                 | Pre-train the core CSU-IR model from scratch on simulation datasets.            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/train_CSU-IR_in_colab.ipynb)                               |
| **Train PS-SMILES-Classifier**         | Fine-tune the specialized classifier for psychoactive substances using SMILES.      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Hsqcsu/CSU-IR/blob/main/train_SMILE_Classifier_in_colab.ipynb)                               |

## 💻 Local Installation and Usage

For users who wish to run the project locally, please follow these steps.

### 1. Environment Setup

Setting up the environment is straightforward using pip.

#### 1. Clone the repository

```bash
git clone https://github.com/Hsqcsu/CSU-IR.git
cd CSU-IR
```

#### 2. Install the required packages using pip

```bash
pip install -r requirements.txt
```

### 2. Data & Checkpoints Download

All large data files, processed libraries, and pre-trained model weights are hosted on Hugging Face for easy access. You must download these assets manually and place them into the corresponding directories as structured in this project to run the local scripts. 
Download Hub: Hugging Face Repository

### 3. Training

You can initiate training scripts using a configuration file. All configs are located in the configs/ directory.

> Ensure you are in the project's root directory and your environment is activated.
> Put the corresponding training data into the corresponding folder according to the config file. 

#### Pre-train CSU-IR with Molecular Dynamics (MD) data

```bash
python -m CSU-IR.train_and_val.pretrain_MD --config configs/config_CSU-IR_pretrain_MD.yaml
```

#### Pre-train CSU-IR with Density Functional Theory (DFT) data

```bash
python -m CSU-IR.train_and_val.pretrain_DFT --config configs/config_CSU-IR_pretrain_DFT.yaml
```

#### Train the SMILES-based Psychoactive Substance Classifier

```bash
python -m PS-Classifier.train_and_val.trian_SMILES_Classifier --config configs/config_SMILES_Classifer_train.yaml
```

> **Note on Fine-tuning and IR-Classifier Training:**  
> The datasets used for fine-tuning the CSU-IR model and training the IR-Classifier are subject to copyright and are not publicly released. However, the code and logic are provided for transparency. The fine-tuning process mirrors the pre-training scripts. The IR-Classifier training script can be found at [`PS-Classifier/train_and_val/train_IR_Classifier.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/PS-Classifier/train_and_val/train_IR_Classifier.py).

---

### 4. Testing and Inference

Scripts for testing and inference are available in their respective project folders. Please ensure you have downloaded the required data and model weights, placing them in their corresponding directories as structured in this project, before running the scripts.

- **CSU-IR Retrieval**: [`CSU-IR/test_and_infer/`](https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/test_and_infer)
- **PS-Classifier**: [`PS-Classifier/test_and_infer/`](https://github.com/Hsqcsu/CSU-IR/tree/main/PS-Classifier/test_and_infer)

> **✨ Use Your Own Library!**  
> To perform retrieval against your own custom library, please refer to this script:  
> [`single_test_in_user_defined_library.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/test_and_infer/single_test_in_user_defined_library.py)

---

## 🌐 Web Service & Live Demo

We have integrated all functionalities into a public Hugging Face Space for easy access and live demonstration.

**[➡️ Try the Live Demo Here!](https://huggingface.co/spaces/Skylight666/CSU-IR-normal-compound-identification)**

The demo includes:
- **General Retrieval**: Search against multi-scale general-purpose libraries.
- **PS Retrieval**: Specialized search against psychoactive substance libraries.
- **PS-IR-Classifier**: Classify substances using IR spectra.
- **PS-SMILES-Classifier**: Classify substances using SMILES strings.

The performance and reliability of these tools are backed by extensive benchmarking. You can review the detailed results in the section below.

## 📊 Performance & Results

Here is a summary of our model's performance benchmarks.

| General Retrieval (Internal Test) | Top-1 Accuracy | Top-10 Accuracy |
| :-------------------------------: | :------------: | :-------------: |
|         Library Size: 4k          |     0.8849     |     0.9745      |
|        Library Size: 200k         |     0.6210     |     0.9004      |
|          Library Size: 1M         |     0.5202     |     0.8245      |
|          Library Size: 2M         |     0.4501     |     0.7755      |

| General Retrieval (External NIST Test) | Top-1 Accuracy | Top-10 Accuracy |
| :------------------------------------: | :------------: | :-------------: |
|          Library Size: NIST          |     0.8250     |     1.0000      |
|           Library Size: 4k           |     0.7750     |     0.9500      |
|          Library Size: 200k          |     0.4750     |     0.8500      |
|           Library Size: 1M           |     0.4750     |     0.8250      |
|           Library Size: 2M           |     0.3250     |     0.7250      |

| NPS Retrieval (Internal Test)                      | Top-1 Accuracy | Top-10 Accuracy |
| :------------------------------------------------: | :------------: | :-------------: |
|                 Existed PS Library                 |     0.6190     |     0.9524      |
| Derivative PS Library* <br> <small>(6 NPS filtered)</small> |     0.3889     |     0.8333      |

| PS-Classifier Accuracy (NPS Test Set) |  Accuracy  |
| :-----------------------------------: | :--------: |
|             IR-Classifier             |    100%    |
|           SMILES-Classifier           |   95.24%   |

---

## 📦 Hardware Requirements

All experiments were conducted on a single NVIDIA GPU (RTX 4090). The full pre-training pipeline for CSU-IR requires approximately **48 hours**:
- **Stage 1 (MD Data)**: ~40 hours
- **Stage 2 (DFT Data)**: ~6 hours
- **Stage 3 (Experimental Data Fine-tuning)**: ~2 hours

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{YourLastName2024CSUIR,
  title   = {Interpretable Contrastively Spectral-structural Unification between infrared Spectra and Molecular Structures assisting Novel Psychoactive Substances identification},
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

