================
   Usage Guide
================

This guide provides several ways to use and explore the CSU-IR project, from quick online demos to full local setup and training.

.. contents::
   :local:
   :depth: 2

Step 1: Quick Start with the Web Service
----------------------------------------

For a quick and easy way to use the model without any installation, you can use our deployed web service.

* **Try the Web Demo**: `CSU-IR Web Service <https://huggingface.co/spaces/HSQC/CSU-IR>`_

Step 2: Interactive Exploration with Google Colab
-------------------------------------------------

Use our pre-configured Google Colab notebooks to check results or perform training in the cloud with free GPU resources. Click on the badges to open the notebooks directly in Google Colab.

Demonstration & Testing Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 45 20
   :header-rows: 1

   * - Notebook
     - Description
     - Link
   * - **CSU-IR General Retrieval**
     - See the results and visualizations for general compound retrieval tests.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_CSU-IR_General_retrieval_in_colab.ipynb
          :alt: Open In Colab
   * - **CSU-IR NPS Retrieval**
     - Explore the specialized retrieval results for Novel Psychoactive Substances (NPS).
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_CSU-IR_NPS_retrieval_in_colab.ipynb
          :alt: Open In Colab
   * - **PS-IR-Classifier Demo**
     - View the performance and interactive demo of the IR-based PS classifier.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_PS-IR-Classifier_in_colab.ipynb
          :alt: Open In Colab
   * - **PS-SMILES-Classifier Demo**
     - View the performance and interactive demo of the SMILES-based PS classifier.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/test_PS-SMILES-Classifier_in_colab.ipynb
          :alt: Open In Colab

Model Training Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 45 20
   :header-rows: 1

   * - Notebook
     - Description
     - Link
   * - **Train CSU-IR Model**
     - Pre-train the core CSU-IR model from scratch using our simulation datasets.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/train_CSU-IR_in_colab.ipynb
          :alt: Open In Colab
   * - **Train PS-SMILES-Classifier**
     - Train the specialized classifier for psychoactive substances using SMILES data.
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/train_SMILE_Classifier_in_colab.ipynb
          :alt: Open In Colab


Step 3: Full Local Setup and Training
-------------------------------------

For advanced use, such as custom training or in-depth analysis, follow these steps to set up the project on your local machine.

Prerequisites
~~~~~~~~~~~~~

1.  **Install Anaconda**:
    We recommend using Conda to manage Python environments. If you don't have it, download and install it from the `Anaconda official website <https://www.anaconda.com/products/distribution>`_.

2.  **Prepare an IDE**:
    An Integrated Development Environment (IDE) like `PyCharm <https://www.jetbrains.com/pycharm/>`_ or `Visual Studio Code <https://code.visualstudio.com/>`_ is highly recommended for a better development experience.

Installation and Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, open your terminal (Anaconda Prompt on Windows).

**1. Clone the Repository**

.. code-block:: bash

   git clone https://github.com/Hsqcsu/CSU-IR.git
   cd CSU-IR

**2. Create and Activate the Conda Environment**

.. code-block:: bash

   conda create --name CSU-IR python=3.11.9
   conda activate CSU-IR

**3. Install Required Packages**

.. code-block:: bash

   pip install -r requirements_local.txt

Local Training
~~~~~~~~~~~~~~

.. important::
   **Download Datasets First!**

   Before you can start training, you must download the required datasets from our Hugging Face repository: `CSU-IR Datasets on Hugging Face <https://huggingface.co/datasets/HSQC/CSU-IR_DATA>`_.

   Please check the corresponding ``.yaml`` file in the ``configs/`` directory to identify which specific dataset is needed for each training script. Ensure the data is placed in the correct directory as specified in the configuration (e.g., ``data/pretrain_data/``).

Once the environment and data are correctly set up, you can start training the models from the terminal.

**Pre-train CSU-IR with Molecular Dynamics (MD) data**

.. code-block:: bash

   python -m CSU-IR.train_and_val.pretrain_MD --config configs/config_CSU-IR_pretrain_MD.yaml

**Pre-train CSU-IR with Density Functional Theory (DFT) data**

.. code-block:: bash

   python -m CSU-IR.train_and_val.pretrain_DFT --config configs/config_CSU-IR_pretrain_DFT.yaml

**Train the SMILES-based Psychoactive Substance Classifier**

.. code-block:: bash

   python -m CSU-IR.train_and_val.train_SMILES_Classifier --config configs/config_SMILES_Classifer_train.yaml

Reproducing Original Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   To view and reproduce the results from the original paper, you must first download the pre-trained model weights and the required test data.

   * **Download From**: `CSU-IR Model Hub <https://huggingface.co/HSQC/CSU-IR>`_
   * **Placement**: Please place all downloaded files into their respective directories as expected by the analysis scripts (e.g., model weights into a ``checkpoints/`` folder and test data into the ``data/`` folder).

After setting up the necessary files, you can run the provided analysis scripts to reproduce the results.

*   **Analysis Script**: Please refer to the ``reproduce_results.py`` script for detailed instructions on execution.