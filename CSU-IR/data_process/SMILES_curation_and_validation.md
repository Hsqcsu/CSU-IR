# Comprehensive Data Curation, Label Verification, and Quality Control Protocol

To ensure the highest standard of physical consistency and chemical fidelity in infrared (IR) spectral cross-modal alignment, all datasets in CSU-IR undergo a multi-stage data curation and verification pipeline.

---

## 1. Chemical Entity Identification & SMILES Label Verification

To ensure one-to-one fidelity between spectral records and chemical structures:

1. **Label Filtering and Retrieval**:
   - Raw records with missing, ambiguous, or invalid SMILES strings were filtered out.
   - For datasets lacking canonical SMILES (e.g., NIST, SWGDRUG, ENFSI, and Thermo Fisher), IUPAC query tools, substance names, and CAS numbers were queried via PubChem and automated IUPAC search tools to obtain canonicalized, high-fidelity SMILES labels.
   - Code implementation: [`CSU-IR/data_process/SMILES_extraction.py`](./data_process/SMILES_extraction.py).

2. **Automated Verification**:
   - An independent validation suite ([`CSU-IR/data_process/SMILES_verification.py`](./data_process/SMILES_verification.py)) was executed to verify structural validity and chemical validity (valence checks, isotope specifications).
   - Both input queries and the final verified SMILES labels are made fully accessible.
   - For complete reproducibility, full local verification logs are hosted on our [Hugging Face Repository](https://huggingface.co/Hsqcsu/CSU-IR):
     - `CSU-IR/Multi-stage training data/Stage III- EXP/EB/verification_log.txt` (EB dataset)
     - `CSU-IR/Multi-stage training data/Stage III- EXP/EB_TF/tf_data_verification_log.txt` (Thermo Fisher dataset)

---

## 2. Preservation of Salt Forms in Infrared Cross-Modal Matching

A critical domain-specific consideration in vibrational spectroscopy is the **treatment of salt forms**. 

### 2.1 Theoretical Rationale
In standard chemoinformatics applications, salts are often stripped (desalted) to map compounds to a parent neutral molecule. However, **infrared spectroscopy directly measures molecular vibrational modes, which are drastically altered by ionic interactions and protonation states** (e.g., counterion bands, amine salt hydrogen bonding networks, and lattice vibrations). Artificially stripping salts from the target SMILES introduces an irreconcilable physical discrepancy between the experimental spectrum and the structural representation.

### 2.2 Empirical Evidence: Divergence of Free Bases vs. Salt Forms
To quantify this discrepancy, we analyzed the spectral similarity and latent embedding space of representative free bases and their hydrochloride (HCl) salts (e.g., MPBP and 4-CIC):

| Compound | Intra-Modal IR Similarity (HCl vs. Base) - Raw Spectra | Intra-Modal IR Similarity (HCl vs. Base) - IR Latent Embeddings | Cross-Modal Similarity (IR vs. Base SMILES) - Base IR | Cross-Modal Similarity (IR vs. Base SMILES) - HCl IR |
| :--- | :---: | :---: | :---: | :---: |
| **4'-Methyl-alpha-pyrrolidinobutyrophenone (MPBP)** | 0.6791 | 0.5136 | 0.1735 | **0.0656** |
| **4-CIC** | 0.5870 | 0.4150 | 0.1341 | **0.0955** |

*(Table 1: Quantitative similarity evaluations between base and HCl forms across IR representations and cross-modal retrieval.)*

- **Raw Spectral Similarity**: As shown above, the cosine similarity between the raw spectra of the free base and HCl salt is inherently low (0.6791 for MPBP; 0.5870 for 4-CIC).
- **Latent Divergence**: When forced to map to an identical desalted SMILES representation, the neural IR encoder accentuates the intrinsic vibrational differences rather than closing them (embedding similarity drops to 0.5136 and 0.4150).
- **Retrieval Performance Degradation**: Salt-form IR spectra show an abrupt drop in cross-modal retrieval scores against base SMILES (e.g., falling from 0.1735 to 0.0656 for MPBP).

**Protocol**: All compounds originally characterized as salt forms strictly retain their corresponding salt-containing SMILES strings and are evaluated as distinct chemical entities.

---

## 3. Spectral Anomaly & Artifact Removal

Raw spectra from public repositories often contain instrument-level artifacts. Prior to dataset partitioning and model training:
- Severe non-physical noise (e.g., baseline jumps, saturations, and high-frequency detector noise) was systematically inspected and filtered.
- Spectra displaying extreme baseline distortion or corrupted channels were excluded.

---

## 4. Reproducibility & Pipeline Usage

All scripts for spectral preprocessing, filtering, and validation are centralized under:
- `CSU-IR/data_process/unified_preprocess_spectrum.py`
- `CSU-IR/data_process/SMILES_extraction.py`
- `CSU-IR/data_process/SMILES_verification.py`

Refer to the main [README.md](./README.md) for direct reproduction commands.
