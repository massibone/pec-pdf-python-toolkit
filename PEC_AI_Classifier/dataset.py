"""Dataset PyTorch per PEC."""
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .utils import preprocess_text

class PECDataset(Dataset):
    """
    Dataset PyTorch per classificazione PEC.
    
    Args:
        texts: Lista di testi (subject + body)
        labels: Lista di label indices
        vocab: Dict {token: id}
        max_len: Lunghezza padding
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_len: int = 150
    ):
        assert len(texts) == len(labels), "Testi e labels mismatched"
        
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab['<PAD>']
        self.unk_idx = vocab['<UNK>']
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> tuple:
        """Ritorna (token_ids_tensor, label_tensor)"""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Preprocess
        tokens = preprocess_text(text)
        
        # Converti a IDs
        token_ids = [
            self.vocab.get(token, self.unk_idx)
            for token in tokens
        ][:self.max_len]
        
        # Padding
        token_ids += [self.pad_idx] * (self.max_len - len(token_ids))
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

class PECDatasetWithMask(PECDataset):
    """
    PECDataset con attention mask per gestire padding.
    """
    
    def __getitem__(self, idx: int) -> tuple:
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = preprocess_text(text)
        token_ids = [
            self.vocab.get(token, self.unk_idx)
            for token in tokens
        ][:self.max_len]
        
        # Crea mask: 1 per token veri, 0 per padding
        mask = [1] * len(token_ids) + [0] * (self.max_len - len(token_ids))
        
        token_ids += [self.pad_idx] * (self.max_len - len(token_ids))
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
            torch.tensor(label, dtype=torch.long)
        )
5️⃣ pec_ai_classifier/model.py
Copy
"""Modello LSTM per classificazione PEC."""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PECClassifierLSTM(nn.Module):
    """
    Classificatore LSTM per PEC.
    
    Architettura:
    - Embedding layer
    - BiLSTM
    - Fully connected + output
    
    Args:
        vocab_size: Dimensione vocabolario
        embed_dim: Dimensione embedding
        hidden_dim: Dimensione hidden LSTM
        num_classes: Numero categorie
        num_layers: Numero layer LSTM (default 2)
        dropout: Dropout rate
        pad_idx: Indice token PAD
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Embedding con pad_idx per non trainare i pad
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )
        
        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Embedding: (batch, seq) -> (batch, seq, embed_dim)
        embedded = self.embedding(input_ids)
        
        # LSTM: (batch, seq, embed_dim) -> (batch, seq, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Usa ultimo hidden state
        # Se bidirectional: concatena forward + backward
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # FC
        logits = self.fc(last_hidden)  # (batch, num_classes)
        
        return logits

class PECClassifierWithAttention(nn.Module):
    """
    Classificatore LSTM con attention mechanism.
    Più sofisticato: l'attention impara quali token sono importanti.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # Attention
        self.attention_weights = nn.Linear(lstm_output_dim, 1)
        
        # Classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq)
            mask: (batch, seq) - 1 per token veri, 0 per padding
        
        Returns:
            logits: (batch, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq, embed_dim)
        
        lstm_out, _ = self.lstm(embedded)  # (batch, seq, hidden_dim*2)
        
        # Attention scores
        attention_logits = self.attention_weights(lstm_out)  # (batch, seq, 1)
        
        # Applica mask se fornito
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq, 1)
            attention_logits = attention_logits.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(attention_logits, dim=1)  # (batch, seq, 1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        
        logits = self.fc(context)
        
        return logits
