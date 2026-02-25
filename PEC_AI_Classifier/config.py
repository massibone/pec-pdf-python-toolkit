"""Configurazione globale."""
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODELS_DIR = PROJECT_ROOT / 'models'
    REPORTS_DIR = PROJECT_ROOT / 'reports'
    
    # Model
    VOCAB_SIZE = 5000
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_CLASSES = 4
    MAX_LEN = 150
    
    # Training
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 0.001
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    RANDOM_STATE = 42
    
    # Categories
    CATEGORIES = ['FATTURA', 'GARA', 'COMUNICAZIONE', 'ALTRO']
    
    # Preprocessing
    SPACY_MODEL = 'it_core_news_sm'
    
    def __post_init__(self):
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.REPORTS_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)

config = Config()
