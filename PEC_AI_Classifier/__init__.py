"""PEC AI Classifier."""

__version__ = '1.0.0'

from .config import config
from .model import PECClassifierLSTM
from .inference import PECPredictor, generate_report
from .train import train_classifier

__all__ = [
    'config',
    'PECClassifierLSTM',
    'PECPredictor',
    'generate_report',
    'train_classifier'
]
