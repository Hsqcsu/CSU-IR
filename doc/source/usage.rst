================
   Usage Guide
================

This guide provides several ways to use and explore the `CSU-IR project <https://github.com/Hsqcsu/CSU-IR/tree/main>`_, from quick online demos to full local setup, training, evaluation, and 100-Million compound retrieval.

.. contents::
   :local:
   :depth: 2

Step 1: Quick Start with our Web Service 
----------------------------------------

For a quick and easy way to use the model without any local installation, you can use our deployed open-access web service.
This service supports:

* **General Retrieval**: Search against the NIST library (~10k compounds).
* **PS Retrieval**: Specialized search against Psychoactive Substance libraries.
* **Custom Libraries**: Perform retrieval directly against your own user-provided spectral libraries.

* **Try the Web Demo**: `CSU-IR Web Service on Hugging Face Spaces <https://huggingface.co/spaces/Hsqcsu/CSU-IR-Web>`_

Step 2: Interactive Exploration with Google Colab
-------------------------------------------------

Use our pre-configured Google Colab notebooks to experience our models instantly in the cloud with free GPU resources. Dependencies and data downloads are handled automatically.

Testing Notebooks
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 45 20
   :header-rows: 1

   * - Notebook
     - Description
     - Link
   * - **CSU-IR Testing**
     - Explore specialized retrieval results in CSU-IR.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/test_CSU_IR_in_colab.ipynb
          :alt: Open In Colab

Training Notebooks
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 45 20
   :header-rows: 1

   * - Notebook
     - Description
     - Link
   * - **Train CSU-IR**
     - Train the CSU-IR model with DFT data.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/train_CSU_IR_in_colab.ipynb
          :alt: Open In Colab

.. note::
   For full multi-stage training, please refer to : "Step 3: Full Local Setup and Training <step-3-local-setup>" and download the complete datasets from our `Hugging Face Repository <https://huggingface.co/Hsqcsu/CSU-IR/tree/main>`_.

.. _step-3-local-setup:

Step 3: Full Local Setup and Training
-------------------------------------

For advanced use, such as custom multi-stage training, benchmark evaluation, or local testing, follow these steps to set up the project on your local machine.

Prerequisites & Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the Repository**

.. code-block:: bash

   git clone https://github.com/Hsqcsu/CSU-IR.git
   cd CSU-IR

2. **Create and Activate the Conda Environment**

.. code-block:: bash

   conda create --name CSU-IR python=3.11.9
   conda activate CSU-IR

3. **Install Required Packages**

.. code-block:: bash

   pip install -r requirements/requirements_local.txt

Data & Checkpoints Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EB Dataset Construction
^^^^^^^^^^^^^^^^^^^^^^^

The complete Experimental Benching (EB) dataset with identical splits can be reproduced locally:

1. **Download Baseline Files**: Download the EB dataset (including SMILES text files, compound labels, and IR ``.pt`` files without NIST IR data) from `Hugging Face Hub (Stage III- EXP / EB Folder) <https://huggingface.co/Hsqcsu/CSU-IR/tree/main/Multi-stage%20training%20data/Stage%20III-%20EXP/EB>`_.
2. **Place Files**: Put all downloaded files into ``CSU-IR/data/Multi-staged_training_data/Experimental_Benching_data/EB/data_without_NIST_IR/``.
3. **Execute Reconstruction**: Run the data reconstruction script:

.. code-block:: bash

   python CSU-IR/EB_dataset_construction/EB_data_reconstruction.py

Reconstructed files will be saved in ``CSU-IR/data/Multi-staged_training_data/Experimental_Benching_data/EB/data_with_NIST_IR/``.

.. note::
   * **Runtime**: Reconstruction takes approximately 11 hours. A stalled progress bar is normal—do not interrupt the execution. The script supports auto-resume if terminated.
   * **Subsets**: The ``EB_CHONF`` and ``PS`` subsets can be built using scripts in ``CSU-IR/EB_dataset_construction/``. SMILES and DFT data for ``EB_CHONF_E2S`` are hosted on `Hugging Face <https://huggingface.co/Hsqcsu/CSU-IR/tree/main/EB_CHONF_E2S>`_.

.. warning::
   These reconstruction scripts are provided solely for reproducibility and provenance tracking. Users are responsible for complying with the NIST Chemistry WebBook (SRD 69) terms of use.

EB_TF Dataset Construction (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To incorporate proprietary Thermo Fisher (TF) commercial database data:

* Obtain the identifier and source labels from `Hugging Face (Stage III- EXP / EB_TF Folder) <https://huggingface.co/Hsqcsu/CSU-IR/tree/main/Multi-stage%20training%20data/Stage%20III-%20EXP/EB_TF>`_.
* Preprocess the data using the provided pipeline script in ``CSU-IR/data_process/ir_process.py``.

Pretrained Weights & Candidate Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All pretraining data, candidate libraries, and model checkpoints are available on the `Hugging Face Hub <https://huggingface.co/Hsqcsu/CSU-IR>`_. Please place downloaded assets into their respective folders as specified in the configuration files under ``configs/``.

Local Training
~~~~~~~~~~~~~~

All training scripts are located in ``local_training/`` and configured via files in ``configs/``.

.. important::
   * Always run training scripts from the project root in an IDE terminal (PyCharm / VS Code) to avoid path and environment mismatches.
   * The ``batch_size`` in the configuration files is on a **per-GPU basis**. In DDP mode, the total batch size scales with device count. If you observe performance drops, consider reducing per-GPU batch size or tuning the learning rate.

**1. Single-GPU Training**

.. code-block:: bash

   python local_training/Multi-stage_training_CSU-IR_in_local.py --config <CONFIG_PATH>

**2. Multi-GPU Training (DDP, e.g., 3 GPUs)**

.. code-block:: bash

   torchrun --nproc_per_node=3 local_training/Multi-stage_training_CSU-IR_in_local_DDP.py --config <CONFIG_PATH>

**Multi-Stage Configurations:**

* **Stage-I (MD)**: ``configs/config_CSU-IR_Multi-stage_training_I_MD.yaml``
* **Stage-II (DFT)**: ``configs/config_CSU-IR_Multi-stage_training_II_DFT.yaml``
* **Stage-III (EXP)**: ``configs/config_CSU-IR_Multi-stage_training_III_EXP.yaml``

.. tip::
   **Multi-Seed Trials for Stage-III (EXP):**
   Stage-III scripts natively support repeated trials across multiple seeds. You can modify ``training_params.seeds`` in the YAML file:
   
   * ``seeds: [42]``: Default single-seed run.
   * ``seeds: [42, 420, 4200]``: Automatically performs triplicate trials. Checkpoints are logged to ``seed_<SEED>/`` subdirectories along with an aggregated ``multi_seed_summary.json``.

**Additional Training Scenarios:**

* **SOTA Comparison**: Use the corresponding training configurations in ``configs/``. For QM9S multi-modal training, set the task parameter to ``QM9S_ir_raman`` and adjust the batch size accordingly.
* **Downstream PS Fine-Tuning**: Available in ``local_training/`` with matching config files (access to the private PS validation set can be requested via contact).

**Hardware Requirements & Training Time Reference:**

* **Stage-I (MD Data)**: ~33 hours on a single NVIDIA RTX 4090.
* **Stage-II (DFT Data)**: ~8 hours on a single NVIDIA RTX 4090.
* **Stage-III (EXP Data)**: ~10 minutes for ~20,000 data items on 3x NVIDIA RTX 6000 Ada GPUs.

Testing, Inference, and Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inference & Local Testing
^^^^^^^^^^^^^^^^^^^^^^^^^

Testing scripts are located in `CSU-IR/test_and_infer/ <https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/test_and_infer>`_. These scripts do not require command-line arguments and can be run directly inside your IDE.

Reproducing Headline Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A unified evaluation script is provided to reproduce all headline metrics and downstream analyses from pre-trained checkpoints:

.. code-block:: bash

   cd CSU-IR/CSU-IR/headline_results_evaluations
   python run_headline_evaluations.py --task retrieval_100m --gpu_ids 0

For a comprehensive evaluation walkthrough, please consult ``REPRODUCE.md <https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/headline_results_evaluations/REPRODUCE.md>`` in the repository.

Step 4: Local 100-Million-library-Retrieval
------------------------------------------

For high-throughput 100-Million-Scale retrieval, CSU-IR provides an intuitive local GUI:

1. **Download Library**: Download the processed 100-million library from `Hugging Face Dataset Hub <https://huggingface.co/datasets/Hsqcsu/CSU-IR_100_Million_library/tree/main/100_Million_library>`_.
2. **Organize Directory**: Place the downloaded dataset files inside ``data/100-Million-library-Retrieval/ <https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/data/100-Million-library-Retrieval>``.
3. **Launch Retrieval GUI**:

.. code-block:: bash

   python CSU-IR/100-Million-library-Retrieval/Retrieval_GUI.py

Open the local URL displayed in your terminal. The interface supports:

* **IR Only**: Search against 100 million compounds using IR spectra alone.
* **IR + Molecular Weight**: Narrow search spaces by combining spectral features with molecular weight filtering.
* **IR + Molecular Formula**: Search against the library with exact chemical formula constraints combined with spectral similarity.
