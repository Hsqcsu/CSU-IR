# **CSU-IR: An Interpretable Deep Learning Framework for 100-Million-Scale Infrared Spectral Retrieval**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/YOUR_ARXIV_ID)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_all_in_colab.ipynb)  [![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Demo-blue.svg)](https://huggingface.co/spaces/Hsqcsu/CSU-IR-Web)

This is the official code repository for our paper, **"CSU-IR: An Interpretable Deep Learning Framework for 100-Million-Scale Infrared Spectral Retrieval"**.

We introduce **CSU-IR**, a novel deep learning framework designed for high-precision unknown-compound identification by unifying infrared (IR) spectra and molecular structures. Key strengths of our work include:

*   🚀 **Exceptional 100-Million-Scale Performance**: Maintains high accuracy (Recall@1 of 68.13% and a Recall@10 of 93.97%) when retrieving against libraries containing 100 million compounds.
*   🎯 **Specialized for Psychoactive Substances Identification**: Purpose-built models and libraries for the accurate retrieval and SMILES classification of Psychoactive Substances.
*   🔬 **Strong Interpretability**: The learned representations are highly interpretable, revealing a direct mapping between spectral features and molecular features.
*   🧩 **Chemical Substructures Detection**: The model successfully detected 83 chemical substructures, including common functional groups, with an average detection rate of 95.46%.

![Fig1](https://github.com/user-attachments/assets/7b3fffd1-6c0d-41c9-99a3-dd9988be6d8e)


## 🚀 Quick Start with Google Colab

Experience the full power of our models instantly, without any local setup. Our Colab notebooks handle all dependencies and data downloads automatically, allowing you to explore our models directly in your browser.

---

### **Demonstration & Testing Notebooks**

| Notebook                               | Description                                                                       | Link                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CSU-IR NPS Retrieval**               | Explore the specialized retrieval results for Novel Psychoactive Substances (NPS).  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/test_CSU-IR_NPS_retrieval_in_colab.ipynb)       |
| **PS-SMILES-Classifier Demo**          | View the performance and interactive demo of the SMILES-based PS classifier.      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/test_PS-SMILES-Classifier_in_colab.ipynb)      |

---

### **Model Training Notebooks**

| Notebook                               | Description                                                                       | Link                                                                                                                                                                                            |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Train CSU-IR Model**                 | Train the CSU-IR model using the simulation datasets.         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/train_CSU-IR_in_colab.ipynb)                              |
| **Train PS-SMILES-Classifier**         | Train the specialized classifier for psychoactive substances using SMILES data.     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/train_SMILE_Classifier_in_colab.ipynb)     

> **Note on Dataset Size:** 
> The **Train CSU-IR Model** notebook uses a 1/5 sample of the data for demonstration to avoid exceeding Google Drive storage limits. For full training, please see the `train` instructions in our `README`'s `Local Installation and Usage` section and download the complete dataset from our [Hugging Face Repository](https://huggingface.co/Skylight666/CSU-IR/tree/main).

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
pip install -r requirements_local.txt
```

### 2. Data & Checkpoints Download

All large data files, processed libraries, and pre-trained model weights are hosted on Hugging Face for easy access. You must download these assets manually and place them into the corresponding directories as structured in this project to run the local scripts. 
 **[Download Hub: Hugging Face Repository](https://huggingface.co/Skylight666/CSU-IR)**

### 3. Training

You can initiate training scripts using a configuration file. All configs are located in the configs/ directory.

> Ensure you are in the project's root directory and your environment is activated.
> Put the corresponding training data into the corresponding folder according to the config file.
> It is strongly recommended to run this script in an IDE terminal (like PyCharm's) instead of the standard system terminal to avoid potential environment-related issues.

#### Pre-train CSU-IR with Molecular Dynamics (MD) data

```bash
python -m pretrain_CSU-IR_in_local --config configs/config_CSU-IR_pretrain_MD.yaml
```

#### Pre-train CSU-IR with Density Functional Theory (DFT) data

```bash
python -m pretrain_CSU-IR_in_local --config configs/config_CSU-IR_pretrain_DFT.yaml
```

#### Train the SMILES-based Psychoactive Substance Classifier

```bash
python -m train_SMILES_Classifier_in_local --config configs/config_SMILES_Classifer_train.yaml
```

> **Note on Experimental Fine-tuning and IR-Classifier Training:**  
> The datasets used for experimental fine-tuning the CSU-IR model and training the IR-Classifier are subject to copyright and are not publicly released. However, the code and logic are provided for transparency. The fine-tuning process mirrors the pre-training scripts. The IR-Classifier training script can be found at [`PS-Classifier/train_and_val/train_IR_Classifier.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/PS-Classifier/train_and_val/train_IR_Classifier.py).

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
|                 Existed PS Library                 |     0.7143     |     0.9762      |
| Derivative PS Library* <br> <small>(6 NPS filtered)</small> |     0.3235     |     0.8824      |

| PS-Classifier Accuracy (NPS Test Set) |  Accuracy  |
| :-----------------------------------: | :--------: |
|             IR-Classifier             |    100%    |
|           SMILES-Classifier           |   90.48%   |

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

