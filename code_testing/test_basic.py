import os
import sys
import shutil
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from unittest.mock import patch

# ==============================================================================
# 1. Path Setup: Inject CSU-IR directory into sys.path
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
CSU_IR_DIR = os.path.join(REPO_ROOT, 'CSU-IR')

# Prioritize CSU-IR directory to resolve internal module imports
if CSU_IR_DIR not in sys.path:
    sys.path.insert(0, CSU_IR_DIR)

# Import components under test
from data_process.ir_process import unified_preprocess_spectrum
from train_and_val.train_and_val import (
    IRSmilesDataset,
    validate_model,
    train_model
)
from test_and_infer.batch_test_and_infer import (
    normalize_smiles,
    embed_ir_tensors,
    embed_smiles_list,
    evaluate_retrieval
)
from model.IR_encoder import IRModel
from model.SMILES_encoder import SmilesModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER_PATH = os.path.join(CSU_IR_DIR, "model", "tokenizer-smiles-roberta-1e_new")


# ==============================================================================
# 2. Component ①: IR Preprocessing Pipeline Test
# ==============================================================================
class TestIRProcess:
    def test_preprocess_transmittance_spectrum(self):
        """Test the standard preprocessing pipeline for transmittance spectra."""
        # Synthesize noisy transmittance data between 400.0 and 4200.0 cm^-1
        wn = np.linspace(400.0, 4200.0, 500)
        # Simulate an absorption peak around 1700 cm^-1
        spec = 80.0 - 50.0 * np.exp(-((wn - 1700) ** 2) / (2 * 50 ** 2))

        # Run preprocessing to uniform 3500 points
        processed = unified_preprocess_spectrum(
            wavenumbers=wn,
            intensities=spec,
            declared_yunits="%T",
            target_start=500.0,
            target_end=4000.0,
            num_points=3500
        )

        assert processed is not None, "Preprocessing failed and returned None."
        assert isinstance(processed, np.ndarray), "Output must be a numpy ndarray."
        assert processed.shape == (3500,), f"Expected shape (3500,), got {processed.shape}"
        assert 0.0 <= np.min(processed) and np.max(processed) <= 1.0, "Values must be scaled to [0, 1]."

    def test_preprocess_invalid_data(self):
        """Test robustness against insufficient data points (< 10 valid points)."""
        wn = np.array([500.0, 600.0, 700.0])
        spec = np.array([0.1, 0.2, 0.3])
        result = unified_preprocess_spectrum(wn, spec)
        assert result is None, "Function should return None for insufficient data points without raising exceptions."


# ==============================================================================
# 3. Component ②: Model Training and Validation Pipeline Test
# ==============================================================================
class TestTrainAndVal:
    @pytest.fixture
    def dummy_data(self):
        """Generate a minimal toy dataset for fast pipeline verification."""
        # Increased to 8 samples (4 batches) to ensure GradScaler completes adaptive scaling and actually updates the optimizer within 1 epoch
        ir_spectra = torch.rand(8, 3500).float()
        smiles = [
            "CCO", "CC(=O)O", "CCN", "c1ccccc1",
            "CCCC", "CC(=O)C", "CCNCC", "c1ccc(O)cc1"
        ]
        return ir_spectra, smiles
    
    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create a temporary directory for model weights and guarantee deletion upon completion."""
        temp_dir = tmp_path / "test_checkpoints"
        temp_dir.mkdir(parents=True, exist_ok=True)
        yield str(temp_dir)
        # Teardown: ensure complete removal of temporary weights and logs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_dataset_and_loader(self, dummy_data):
        """Verify IRSmilesDataset indexing and slicing."""
        ir_spectra, smiles = dummy_data
        dataset = IRSmilesDataset(ir_spectra, smiles)
        assert len(dataset) == len(smiles)
        item_ir, item_s = dataset[0]
        assert item_ir.shape == (3500,)
        assert item_s == "CCO"

    def test_fast_train_and_val_loop(self, dummy_data, temp_checkpoint_dir):
        """Run an end-to-end smoke test on GPU and verify weight generation."""
        ir_spectra, smiles = dummy_data
        dataset = IRSmilesDataset(ir_spectra, smiles)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        ir_model = IRModel().to(DEVICE)
        smiles_model = SmilesModel(
            roberta_model_path=None,
            roberta_tokenizer_path=TOKENIZER_PATH,
            smiles_maxlen=64,
            feature_dim=768
        ).to(DEVICE)

        val_loss, top1_acc = validate_model(smiles_model, ir_model, loader, device=DEVICE)
        assert isinstance(val_loss, float)
        assert 0.0 <= top1_acc <= 1.0

        optimizer = torch.optim.AdamW(
            list(smiles_model.parameters()) + list(ir_model.parameters()), 
            lr=1e-4
        )

        with patch('train_and_val.train_and_val.GradScaler', lambda: torch.cuda.amp.GradScaler(init_scale=128.0)):
            train_model(
                smiles_model=smiles_model,
                ir_model=ir_model,
                train_loader=loader,
                val_loader=loader,
                optimizer=optimizer,
                num_epochs=1,
                warmup_epochs=0,
                patience=2,
                output_dir=temp_checkpoint_dir,
                device=DEVICE
            )

        best_smiles_weight = os.path.join(temp_checkpoint_dir, 'best_smiles_model.pth')
        best_ir_weight = os.path.join(temp_checkpoint_dir, 'best_ir_model.pth')
        history_file = os.path.join(temp_checkpoint_dir, 'history.json')

        assert os.path.exists(best_smiles_weight)
        assert os.path.exists(best_ir_weight)
        assert os.path.exists(history_file)

# ==============================================================================
# 4. Component ③: Batch Inference and Matrix Retrieval Test
# ==============================================================================
class TestTestAndInfer:
    def test_smiles_normalization(self):
        """Test RDKit SMILES canonicalization and fault tolerance."""
        # Canonicalization check
        valid_s = "OCC"
        norm_s = normalize_smiles(valid_s)
        assert norm_s == "CCO", "SMILES normalization must produce canonical form."

        # Edge cases check
        assert normalize_smiles("") is None
        assert normalize_smiles("invalid_smiles_xyz") == "invalid_smiles_xyz"

    def test_embedding_and_retrieval(self):
        """Test batch feature extraction and vectorized top-k matrix retrieval."""
        ir_model = IRModel().to(DEVICE)
        smiles_model = SmilesModel(
            roberta_model_path=None,
            roberta_tokenizer_path=TOKENIZER_PATH,
            smiles_maxlen=64,
            feature_dim=768
        ).to(DEVICE)
        tokenizer = smiles_model.smiles_tokenizer

        dummy_ir = torch.randn(3, 3500).float()
        dummy_smiles = ["CCO", "CC", "CCC"]

        # Feature embeddings
        ir_feats = embed_ir_tensors(dummy_ir, ir_model, DEVICE, batch_size=2)
        smiles_feats = embed_smiles_list(dummy_smiles, smiles_model, tokenizer, DEVICE, batch_size=2)

        assert ir_feats.shape[0] == 3 and ir_feats.ndim == 2
        assert smiles_feats.shape[0] == 3 and smiles_feats.ndim == 2

        # Retrieval evaluation
        scores, indices = evaluate_retrieval(
            query_smiles=["CCO", "CC"],
            query_ir_feats=ir_feats[:2],
            gallery_smiles=dummy_smiles,
            gallery_smiles_feats=smiles_feats,
            loss_type="sigmoid"
        )

        assert scores.shape == (2, 3), "Scores matrix shape mismatch."
        assert indices.shape == (2, 3), "Indices matrix shape mismatch."
