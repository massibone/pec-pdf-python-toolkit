"""Utilità preprocessing e helper."""
import spacy
import logging
import re
from typing import List, Tuple
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# Carica spacy una sola volta
try:
    nlp = spacy.load('it_core_news_sm')
except OSError:
    logger.error("Spacy model non trovato. Esegui: python -m spacy download it_core_news_sm")
    nlp = None

def preprocess_text(text: str) -> List[str]:
    """
    Preprocessing testo PEC:
    - Lowercase
    - Lemmatizzazione
    - Rimuovi stopwords + punteggiatura
    - Mantieni numeri come token speciale
    """
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    if not text:
        return []
    
    # Riconosci numeri
    text = re.sub(r'\d+', '<NUM>', text)
    
    if nlp is None:
        # Fallback: simple tokenization
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ 
        for token in doc 
        if not token.is_stop and (token.is_alpha or token.text == '<NUM>')
    ]
    return tokens

def build_vocab(texts: List[str], min_freq: int = 2, max_vocab: int = 5000) -> dict:
    """
    Costruisci vocabolario da testi.
    
    Args:
        texts: Lista di testi
        min_freq: Frequenza minima per includere token
        max_vocab: Taglia max vocabolario
    
    Returns:
        {token: id} mapping
    """
    tokens_all = []
    for text in texts:
        tokens_all.extend(preprocess_text(text))
    
    # Conta frequenze
    freq_counter = Counter(tokens_all)
    
    # Filtra per frequenza e taglia
    vocab_words = [
        word for word, freq in freq_counter.most_common(max_vocab)
        if freq >= min_freq
    ]
    
    # Crea mapping: indice 0 = PAD, 1 = UNK
    vocab = {word: idx + 2 for idx, word in enumerate(vocab_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    
    logger.info(f"Vocabolario: {len(vocab)} token")
    return vocab

def classify_pec_rule_based(text: str) -> str:
    """
    Fallback rule-based per bootstrap dataset piccoli.
    Usa quando confidence < 0.6.
    """
    text_lower = text.lower()
    
    fattura_keywords = {'fattura', 'iva', 'pagamento', 'importo', 'euro', '€', 'cig', 'cassa'}
    gara_keywords = {'gara', 'bando', 'offerta', 'appalto', 'preventivo', 'licitazione'}
    comunicazione_keywords = {'comunicazione', 'protocollo', 'avviso', 'notifica', 'ricezione'}
    
    if any(kw in text_lower for kw in fattura_keywords):
        return 'FATTURA'
    if any(kw in text_lower for kw in gara_keywords):
        return 'GARA'
    if any(kw in text_lower for kw in comunicazione_keywords):
        return 'COMUNICAZIONE'
    
    return 'ALTRO'

def confidence_to_label(confidence: float) -> str:
    """Converti confidence float a label."""
    if confidence >= 0.85:
        return 'HIGH'
    elif confidence >= 0.65:
        return 'MEDIUM'
    return 'LOW'

def load_pec_csv(csv_path: Path) -> List[dict]:
    """
    Carica dati PEC da CSV.
    
    Expected columns: subject, body, category (opzionale)
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path, dtype={'subject': str, 'body': str})
    
    # Pulisci
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    
    return df.to_dict('records')

# Setup logging
def setup_logging(log_file: Path = None):
    """Configura logging."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler (opzionale)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )
