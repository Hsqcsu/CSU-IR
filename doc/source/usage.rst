================
   Usage Guide
================

This guide provides several ways to use and explore the `CSU-IR project <https://github.com/Hsqcsu/CSU-IR/tree/main>`_, from quick online demos to full local setup, training, and 100-Million compound retrieval.

.. contents::
   :local:
   :depth: 2

Step 1: Quick Start with our Web Service 
----------------------------------------

For a quick and easy way to use the model without any local installation, you can use our deployed open-access web service.
This service supports PS retrieval, general compound retrieval in small libraries (e.g., NIST ~10k), and custom user-provided libraries.

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
   * - **CSU-IR Tesing**
     - Explore the specialized retrieval results in CSU-IR.
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


Step 3: Full Local Setup and Training
-------------------------------------

For advanced use, such as custom multi-stage training or local testing, follow these steps to set up the project on your local machine.

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

.. note::
   The multi-stage pretraining data, candidate libraries, and trained model weights are hosted on `Hugging Face Repository <https://huggingface.co/Hsqcsu/CSU-IR>`_.

   * Ensure training data and checkpoints are downloaded and placed into the corresponding directories as specified in the configuration files (in ``configs/``) before running local scripts.
   * For the **EB dataset construction**, please refer to the detailed reconstruction pipeline and scripts in ``CSU-IR/EB_dataset_construction/``.

Local Training
~~~~~~~~~~~~~~

All stages support both **Single-GPU** and **Multi-GPU (DDP)** training. 

.. important::
   * It is strongly recommended to run training scripts in an IDE terminal (like PyCharm / VS Code) to avoid potential environment-related issues.
   * The ``batch_size`` specified in the configuration files is configured on a **per-GPU basis**.

**1. Single GPU Training**

.. code-block:: bash

   python local_training/Multi-stage_training_CSU-IR_in_local.py --config <CONFIG_PATH>

**2. Multi-GPU Training (DDP, e.g., 3 GPUs)**

.. code-block:: bash

   torchrun --nproc_per_node=3 local_training/Multi-stage_training_CSU-IR_in_local.py --config <CONFIG_PATH>

**Configurations by Stage:**

* **Stage-I (MD)**: ``configs/config_CSU-IR_Multi-stage_training_I_MD.yaml``
* **Stage-II (DFT)**: ``configs/config_CSU-IR_Multi-stage_training_II_DFT.yaml``
* **Stage-III (EXP)**: ``configs/config_CSU-IR_Multi-stage_training_III_EXP.yaml``

Testing and Inference
~~~~~~~~~~~~~~~~~~~~~

Scripts for testing and inference are available in the `CSU-IR/test_and_infer/ <https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/test_and_infer>`_ directory. 

These scripts do not require terminal arguments and can be executed directly within your IDE/code interface.

Step 4: Local 100-Million-library-Retrieval
------------------------------------------

For 100-Million-Scale retrieval, we provide a dedicated local GUI.

1. Download the processed 100-million library from `Hugging Face Dataset Hub <https://huggingface.co/datasets/Hsqcsu/CSU-IR_100_Million_library/tree/main>`_.
2. Place the downloaded library into the ``data/100-Million-library-Retrieval`` folder.
3. Run the GUI script:

.. code-block:: bash

   python CSU-IR/100-Million-library-Retrieval/Retrieval_GUI.py

Click the local URL generated in your terminal to access the retrieval interface, which supports:

* **IR Only**: Search against the 100-Million library using IR spectra alone.
* **IR + Molecular Weight**: Search with combined IR signals and MW filtering.
* **IR + Molecular Formula**: Search with combined IR signals and molecular formula filtering.
