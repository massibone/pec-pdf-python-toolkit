# pec_ai_classifier/classifier_pytorch.py
"""
Classificatore PEC con PyTorch (LSTM). Addestra su dataset custom.
Integra con pec_automation.py: classifica dopo download.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

class PECDataset(Dataset):
    """Dataset PyTorch per testi PEC."""
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = [self.vocab.get(w, 0) for w in re.findall(r'\w+', text.lower())][:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))  # Padding
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

class PECClassifier(nn.Module):
    """LSTM per classificazione PEC."""
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)
        _, (hn, _) = self.lstm(embed)
        out = self.fc(hn[-1])
        return self.softmax(out)

def build_vocab(texts: List[str]) -> Dict[str, int]:
    """Vocabolario da testi."""
    words = [w for text in texts for w in re.findall(r'\w+', text.lower())]
    vocab = {w: i+1 for i, (w, _) in enumerate(Counter(words).most_common(5000))}  # Top 5k
    vocab['<PAD>'] = 0
    return vocab

def train_classifier(pecs_data: List[Dict], epochs=10, lr=0.01):
    """Addestra modello su sample PEC."""
    categories = ['FATTURA', 'GARA', 'COMUNICAZIONE', 'ALTRO']
    label_map = {cat: i for i, cat in enumerate(categories)}
    
    texts = [f"{p['subject']} {p['body']}" for p in pecs_data]
    labels = [label_map.get(classify_pec_rule_based(t), 3) for t in texts]  # Bootstrap con regole
    
    vocab = build_vocab(texts)
    dataset = PECDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PECClassifier(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts_batch, labels_batch in loader:
            optimizer.zero_grad()
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), 'pec_classifier_pytorch.pth')
    return model, vocab, label_map

def classify_pec_pytorch(model, vocab, label_map, text: str):
    """Predizione."""
    model.eval()
    tokens = [vocab.get(w, 0) for w in re.findall(r'\w+', text.lower())][:100]
    tokens += [0] * (100 - len(tokens))
    input_tensor = torch.tensor([tokens])
    with torch.no_grad():
        probs = model(input_tensor)[0]
    pred = torch.argmax(probs).item()
    return list(label_map.keys())[pred], probs[pred].item()

# Fallback rule-based (per bootstrap)
def classify_pec_rule_based(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in {'fattura', 'iva', 'pagamento'}): return 'FATTURA'
    if any(kw in text_lower for kw in {'gara', 'bando'}): return 'GARA'
    if any(kw in text_lower for kw in {'comunicazione', 'protocoll'}): return 'COMUNICAZIONE'
    return 'ALTRO'

def generate_pec_report_pytorch(model, vocab, label_map, pecs: List[Dict], output: Path = Path('report_pec_pytorch.xlsx')):
    """Genera report con PyTorch."""
    results = []
    for pec in pecs:
        text = f"{pec['subject']} {pec['body']}"
        category, confidence = classify_pec_pytorch(model, vocab, label_map, text)
        results.append({'subject': pec['subject'][:50], 'category': category, 'confidence': f"{confidence:.2f}", 'timestamp': pd.Timestamp.now()})
    
    df = pd.DataFrame(results)
    df.to_excel(output, index=False)
    logger.info(f"Report PyTorch: {output}")
    return df

# Esempio uso
if __name__ == "__main__":
    sample_pecs = [  # I tuoi sample reali qui
        {'subject': 'Fattura n.123', 'body': 'Pagamento IVA €1000.'},
        {'subject': 'Bando Gara', 'body': 'Offerte entro 30gg.'}
    ]
    
    # Train (usa tuoi dati reali)
    model, vocab, label_map = train_classifier(sample_pecs, epochs=5)
    
    # Report
    report = generate_pec_report_pytorch(model, vocab, label_map, sample_pecs)
    print(report)
