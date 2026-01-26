import sys
import os
import yaml
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'PS-Classifier'))
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

# If there is a red underline below, don't worry, it will not affect the code running
from model.SMILES_encoder import SmilesModel
from model.Classifier_model import classifymodel


# --- 2. Helper Functions & Classes ---
def load_smiles_labels(smiles_path, labels_path):
    with open(smiles_path, 'r', encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    labels_tensor = torch.load(labels_path)
    return smiles_list, labels_tensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels_tensor):
        self.smiles = smiles_list
        self.labels = labels_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]


# --- 3. Core Training & Validation Logic (with performance optimization) ---
def validate_model(smiles_encoder, classifier, val_loader, criterion, device):
    smiles_encoder.eval()
    classifier.eval()
    running_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc="Validating", unit="batch", leave=False)

    with torch.no_grad():
        for smiles_batch, labels_batch in val_loader_tqdm:
            labels = labels_batch.to(device)

            tokenizer = smiles_encoder.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in smiles_batch]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)


            with autocast():
                smiles_features = smiles_encoder.encode((input_ids, attention_mask), lengths)
                logits = classifier(smiles_features)
                loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            val_loader_tqdm.set_postfix(loss=loss.item())

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss


def train_model(config, smiles_encoder, classifier, train_loader, val_loader, criterion, optimizer, device):
    scaler = GradScaler()
    output_dir = config['paths']['output_dir']
    num_epochs = config['training_params']['num_epochs']
    num_best_models = config['model_save_params']['num_best_models']
    os.makedirs(output_dir, exist_ok=True)
    best_models_tracker, training_losses, validation_losses = [], [], []

    for epoch in range(num_epochs):
        smiles_encoder.eval()  # Feature extractor is frozen
        classifier.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for smiles_batch, labels_batch in train_loader_tqdm:
            labels = labels_batch.to(device)

            tokenizer = smiles_encoder.smiles_tokenizer
            encoded_smiles = [tokenizer.encode_plus(
                text=smiles,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for smiles in smiles_batch]

            input_ids = torch.cat([item['input_ids'] for item in encoded_smiles], dim=0).to(device)
            attention_mask = torch.cat([item['attention_mask'] for item in encoded_smiles], dim=0).to(device)
            lengths = attention_mask.sum(dim=1)

            optimizer.zero_grad()
            with autocast():
                with torch.no_grad():  # Ensure feature extractor is not updated
                    smiles_features = smiles_encoder.encode((input_ids, attention_mask), lengths)
                logits = classifier(smiles_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        training_losses.append(epoch_loss)
        val_loss = validate_model(smiles_encoder, classifier, val_loader, criterion, device)
        validation_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save top N best models based on validation loss
        current_model_info = (val_loss, epoch + 1, None)
        if len(best_models_tracker) < num_best_models:
            best_models_tracker.append(current_model_info)
        elif val_loss < max(best_models_tracker, key=lambda x: x[0])[0]:
            worst_model_info = max(best_models_tracker, key=lambda x: x[0])
            if worst_model_info[2] and os.path.exists(worst_model_info[2]): os.remove(worst_model_info[2])
            best_models_tracker.remove(worst_model_info)
            best_models_tracker.append(current_model_info)

        for i, (loss, ep, _) in enumerate(best_models_tracker):
            if ep == epoch + 1:
                model_path = os.path.join(output_dir, f'classifier_epoch_{ep}_valloss_{loss:.4f}.pth')
                torch.save(classifier.state_dict(), model_path)
                best_models_tracker[i] = (loss, ep, model_path)

    print('\nTraining complete.')
    print('Best validation models saved:')
    for loss, epoch, path, in sorted(best_models_tracker, key=lambda x: x[0]):
        print(f"  - Epoch {epoch}: Val Loss = {loss:.4f}, Path: {path}")

    loss_data = {'training_losses': training_losses, 'validation_losses': validation_losses}
    loss_file_path = os.path.join(output_dir, config['model_save_params']['loss_log_file'])
    with open(loss_file_path, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"\nLoss data saved to {loss_file_path}")


# --- 4. Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train SMILES Classifier model.")
    parser.add_argument('--config', type=str, required=False, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        default_config_path = "configs/config_SMILES_Classifer_train.yaml"
        config_path = os.path.join(PROJECT_ROOT,'..', default_config_path)
        print(f"No config file provided via command line. Using default: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to the project root
    for key, path in config['paths'].items():
        if not os.path.isabs(path):
            config['paths'][key] = os.path.join(PROJECT_ROOT, path)

    # Setup device
    device = torch.device(config['training_params']['device'] if config['training_params']['device'] != 'auto' else (
        'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Initialize models
    print("Initializing models...")
    smiles_encoder_config = config['model_params']['smiles_encoder']
    smiles_encoder = SmilesModel(roberta_model_path=None, roberta_tokenizer_path=config['paths']['tokenizer'],
                                 **smiles_encoder_config)

    classifier_config = config['model_params']['classifier']
    classifier = classifymodel(**classifier_config)

    # Load pre-trained weights for the feature extractor and freeze it
    print(f"Loading pre-trained SMILES encoder from: {config['paths']['pretrained_smiles_encoder']}")
    smiles_encoder.load_weights(config['paths']['pretrained_smiles_encoder'])
    for param in smiles_encoder.parameters():
        param.requires_grad = False

    print(f"SMILES Encoder parameters (frozen): {count_parameters(smiles_encoder)}")
    print(f"Classifier parameters (trainable): {count_parameters(classifier)}")

    smiles_encoder.to(device)
    classifier.to(device)

    # Load data
    print("Loading data...")
    train_smiles, train_labels = load_smiles_labels(config['paths']['train_smiles'], config['paths']['train_labels'])
    val_smiles, val_labels = load_smiles_labels(config['paths']['val_smiles'], config['paths']['val_labels'])
    train_dataset = SmilesDataset(train_smiles, train_labels)
    val_dataset = SmilesDataset(val_smiles, val_labels)

    dl_params = config['dataloader_params']
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=dl_params['batch_size'], shuffle=True,
                              num_workers=dl_params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=dl_params['batch_size'], shuffle=False,
                            num_workers=dl_params['num_workers'])

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Setup loss, optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    opt_params = config['optimizer_params']
    # Only optimize the classifier's parameters
    optimizer = AdamW(classifier.parameters(), lr=opt_params['learning_rate'], weight_decay=opt_params['weight_decay'])

    # Start training
    train_model(config, smiles_encoder, classifier, train_loader, val_loader, criterion, optimizer, device)


# --- 5. Script Entry Point ---
if __name__ == '__main__':
    main()
