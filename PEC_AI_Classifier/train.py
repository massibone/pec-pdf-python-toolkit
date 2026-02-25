"""Training loop completo."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List
import pickle
from tqdm import tqdm
import numpy as np

from .config import config
from .model import PECClassifierLSTM
from .dataset import PECDataset
from .utils import (
    preprocess_text, build_vocab, load_pec_csv, setup_logging,
    classify_pec_rule_based
)

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer per il modello PEC."""
    
    def __init__(
        self,
        model: nn.Module,
        vocab: dict,
        label_map: dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        self.device = device
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train per un epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for input_ids, labels in tqdm(train_loader, desc="Training"):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits = self.model(input_ids)
            loss = criterion(logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metriche
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validazione."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, labels in tqdm(val_loader, desc="Validation"):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(input_ids)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 0.001,
        patience: int = 5
    ):
        """Training completo con early stopping."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {epochs}, LR: {lr}, Patience: {patience}")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint()
                logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
    
    def save_checkpoint(self):
        """Salva modello + vocab + label_map."""
        config.MODELS_DIR.mkdir(exist_ok=True)
        
        torch.save(self.model.state_dict(), config.MODELS_DIR / 'model.pth')
        
        with open(config.MODELS_DIR / 'vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
        
        with open(config.MODELS_DIR / 'label_map.pkl', 'wb') as f:
            pickle.dump(self.label_map, f)
        
        logger.info(f"Checkpoint salvato in {config.MODELS_DIR}")

def train_classifier(csv_path: Path):
    """
    Allenamento completo da CSV.
    
    CSV columns: subject, body, category (opzionale)
    """
    setup_logging()
    logger.info(f"Caricamento dati da {csv_path}")
    
    # Load data
    pecs_data = load_pec_csv(csv_path)
    logger.info(f"Loaded {len(pecs_data)} PEC")
    
    # Prepara testi
    texts = [f"{pec['subject']} {pec['body']}" for pec in pecs_data]
    
    # Label
    label_map = {cat: i for i, cat in enumerate(config.CATEGORIES)}
    labels = []
    for text in texts:
        if 'category' in pecs_data[0]:
            # Se hai categoria nel CSV
            cat = next((p['category'] for p in pecs_data if f"{p['subject']} {p['body']}" == text), None)
            label = label_map.get(cat, label_map['ALTRO'])
        else:
            # Usa rule-based per bootstrap
            cat = classify_pec_rule_based(text)
            label = label_map[cat]
        labels.append(label)
    
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Build vocab
    vocab = build_vocab(texts, max_vocab=config.VOCAB_SIZE)
    
    # Create dataset
    dataset = PECDataset(texts, labels, vocab, max_len=config.MAX_LEN)
    
    # Split: train (80%) -> train (80%) + val (20%), test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_STATE)
    )
    
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_STATE)
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # Model
    model = PECClassifierLSTM(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        pad_idx=vocab['<PAD>']
    )
    
    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(model, vocab, label_map)
    trainer.train(
        train_loader,
        val_loader,
        epochs=config.EPOCHS,
        lr=config.LR,
        patience=5
    )
    
    # Test
    logger.info("Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.validate(test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    logger.info(f"✅ Training completato!")
    return trainer

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Path to CSV')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LR)
    
    args = parser.parse_args()
    
    train_classifier(args.data)
