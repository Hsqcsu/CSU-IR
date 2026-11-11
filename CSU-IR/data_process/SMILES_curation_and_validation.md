# Detailed description of the processing procedure for SMILES of the experimental data used in CSU-IR.
---

## 1. SMILES quering and label verification

To ensure one-to-one fidelity between IR and its chemical structure:

1. **SMILES label quering**:
   - Raw substance names or CAS numbers from raw IR files were queried via PubChem and IUPAC search tools to obtain SMILES labels.
   - Code implementation: [`CSU-IR/data_process/SMILES_extraction.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/data_process/SMILES_extraction.py).

2. **SMILES verification**:
   - An independent validation suite ([`CSU-IR/data_process/SMILES_verification.py`](https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/data_process/SMILES_verification.py)) was executed to let readers verify our data. 
   - For complete reproducibility, full local verification logs are hosted on our [Hugging Face Repository](https://huggingface.co/Hsqcsu/CSU-IR):
     - `CSU-IR/Multi-stage training data/Stage III- EXP/EB/verification_log.txt` (EB dataset)
     - `CSU-IR/Multi-stage training data/Stage III- EXP/EB_TF/tf_data_verification_log.txt` (EB_TF dataset)
---

## 2. The treatment of Salt Forms in SMILES

In standard chemoinformatics applications, salts are often stripped (desalted) to map compounds to a parent neutral molecule. However, **infrared spectroscopy directly measures molecular vibrational modes, which are drastically altered by ionic interactions and protonation states**. Figure 1 illustrate the difference. Artificially stripping salts from the target SMILES introduces an irreconcilable physical discrepancy between the experimental spectrum and the structural representation.  

<img width="865" height="685" alt="image" src="https://github.com/user-attachments/assets/1eab7721-de91-4fa4-91e3-67f9473efea2" />

*(Figure 1: Distinct infrared spectral profiles between free base and hydrochloride salt forms of representative compounds (a 4'-Methyl-alpha-pyrrolidinobutyrophenone MPBP; b 4-CIC).)*

To quantify this discrepancy, we analyzed the spectral similarity and latent embedding space (extracted from the trained model using the desalted SMILES) of representative free bases and their hydrochloride (HCl) salts (e.g., MPBP and 4-CIC):

| Compound | Raw IR Similarity (HCl vs. Base) | IR Latent Embeddings Similarity (HCl vs. Base) | Cross-Modal Similarity (Base IR vs. Base SMILES) | Cross-Modal Similarity (HCl IR vs. Base SMILES) |
| :--- | :---: | :---: | :---: | :---: |
| **4'-Methyl-alpha-pyrrolidinobutyrophenone (MPBP)** | 0.6791 | 0.5136 | 0.1735 | **0.0656** |
| **4-CIC** | 0.5870 | 0.4150 | 0.1341 | **0.0955** |

*(Table 1: Quantitative similarity evaluations between base and HCl forms across IR representations and cross-modal retrieval.)*

As shown above, the cosine similarity between the raw spectra of the free base and HCl salt is inherently low (0.6791 for MPBP; 0.5870 for 4-CIC). When forced to map to an identical desalted SMILES representation, the neural IR encoder accentuates the intrinsic vibrational differences rather than closing them (embedding similarity drops to 0.5136 and 0.4150). Salt-form IR spectra show an abrupt drop in cross-modal retrieval scores against base SMILES (e.g., falling from 0.1735 to 0.0656 for MPBP).

**Protocol**: All compounds originally characterized as salt forms strictly retain their corresponding salt-containing SMILES strings and are evaluated as distinct chemical entities.
---
