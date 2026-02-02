================
   Usage Guide
================

This guide provides several ways to use and explore the `CSU-IR project <https://github.com/Hsqcsu/CSU-IR/tree/main>`_. , from quick online demos to full local setup and training.

.. contents::
   :local:
   :depth: 2

Step 1: Quick Start with our Web Service 
----------------------------------------

For a quick and easy way to use the model without any installation, you can use our deployed web service.
This is for PS retrieval or general compound retrieval in small libraries or custom libraries.

* **Try the Web Demo**: `CSU-IR Web Service <https://huggingface.co/spaces/Hsqcsu/CSU-IR-Web>`_

Step 2: Interactive Exploration with Google Colab
-------------------------------------------------

Use our pre-configured Google Colab notebooks to check results or perform training in the cloud with free GPU resources. Click on the badges to open the notebooks directly in Google Colab.

Testing Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 45 20
   :header-rows: 1

   * - Notebook
     - Description
     - Link
   * - **Train CSU-IR Model**
     - Train the CSU-IR model with DFT data
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/Hsqcsu/CSU-IR/blob/main/colab/train_CSU_IR_in_colab.ipynb
          :alt: Open In Colab


Step 3: Full Local Setup and Training
-------------------------------------

For advanced use, such as custom training or 100-Million-library-Retrieval, follow these steps to set up the project on your local machine.

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

**2. Create and Activate the Conda Environment**

.. code-block:: bash

   conda create --name CSU-IR python=3.11.9
   conda activate CSU-IR

**3. Install Required Packages**

.. code-block:: bash

   cd CSU-IR
   pip install -r requirements/requirements_local.txt

Local Training
~~~~~~~~~~~~~~

.. important::
   **Download Datasets First!**

   Before you can start training, you must download the required datasets from our Hugging Face repository: `CSU-IR training Datasets on Hugging Face <https://huggingface.co/Hsqcsu/CSU-IR/tree/main>`_.

   Please check the corresponding ``.yaml`` file in the ``configs/`` directory to identify which specific dataset is needed for each training script. Ensure the data is placed in the correct directory as specified in the configuration.

Once the environment and data are correctly set up, you can start training. It is strongly recommended to run this script in an IDE terminal (like PyCharm's) instead of the standard system terminal to avoid potential environment-related issues.

**train CSU-IR with Molecular Dynamics (MD) data**

.. code-block:: bash

   python -m local_training/Multi-stage_training_CSU-IR_in_local --config configs/config_CSU-IR_Multi-stage_training_I_MD.yaml

**train CSU-IR with Density Functional Theory (DFT) data**

.. code-block:: bash

   python -m CSU-IR.train_and_val.pretrain_DFT --config configs/config_CSU-IR_pretrain_DFT.yaml

**Train the SMILES-based Psychoactive Substance Classifier**

.. code-block:: bash

   python -m local_training/Multi-stage_training_CSU-IR_in_local --config configs/config_CSU-IR_Multi-stage_training_II_DFT.yaml

Reproducing Original Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   To view and reproduce the results from the original paper, you must first download the trained model weights and the required test data.

   * **Download From**: `CSU-IR Model Hub <https://huggingface.co/Hsqcsu/CSU-IR/tree/main>`_
   * **Placement**: Please place all downloaded files into their respective directories as expected by the analysis scripts (e.g., model weights into a ``checkpoints/`` folder and test data into the ``data/`` folder).

After setting up the necessary files, you can run the provided analysis scripts to reproduce the results.

*   **Analysis Script**: Please refer to the `batch_test_and_infer.py <https://github.com/Hsqcsu/CSU-IR/blob/main/CSU-IR/test_and_infer/batch_test_and_infer.py>`_ script for detailed instructions on execution.

Step 4: Local 100-Million-library-Retrieval
----------------------------------------
For 100-Million-Scale retrieval，We have provided a GUI for local usage.

Users need to download the processed 100-million-library-Retrieval library from `Hugging Face <https://huggingface.co/datasets/Hsqcsu/CSU-IR>`_ and place it in the data/100-Million-library-Retrieval folder. Then, simply run `Retrieval_GUI.py <https://github.com/Hsqcsu/CSU-IR/tree/main/CSU-IR/100-Million-library-Retrieval/Retrieval_GUI.py>`_ and click the link generated in the terminal to perform a 100-million-library retrieval.
