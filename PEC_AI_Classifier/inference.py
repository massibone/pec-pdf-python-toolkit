"""Inference e generazione report."""
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime

from .config import config
from .model import PECClassifierLSTM
from .utils import preprocess_text, confidence_to_label, classify_pec_rule_based

logger = logging.getLogger(__name__)

class PECPredictor:
    """Predittore per PEC."""
    
    def __init__(
        self,
        model_dir: Path = config.MODELS_DIR,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model_dir = model_dir
        
        # Load vocab e label_map
        with open(model_dir / 'vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)
        
        with open(model_dir / 'label_map.pkl', 'rb') as f:
            self.label_map = pickle.load(f)
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load model
        model = PECClassifierLSTM(
            vocab_size=len(self.vocab),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            pad_idx=self.vocab['<PAD>']
        )
        model.load_state_dict(torch.load(model_dir / 'model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        
        self.model = model
        logger.info(f"Model loaded from {model_dir}")
    
    def predict(
        self,
        text: str,
        use_rule_based_fallback: bool = True,
        fallback_threshold: float = 0.6
    ) -> Tuple[str, float, str]:
        """
        Predici categoria per un testo PEC.
        
        Args:
            text: Subject + body
            use_rule_based_fallback: Se confidence < fallback_threshold, usa rule-based
            fallback_threshold: Soglia per fallback
        
        Returns:
            (category, confidence, confidence_label)
        """
        if not text or not text.strip():
            logger.warning("Testo vuoto")
            return 'ALTRO', 0.0, 'LOW'
        
        try:
            # Preprocess
            tokens = preprocess_text(text)
            if not tokens:
                logger.warning("Nessun token dopo preprocessing")
                return 'ALTRO', 0.0, 'LOW'
            
            # Tokenize
            token_ids = [
                self.vocab.get(token, self.vocab['<UNK>'])
                for token in tokens
            ][:config.MAX_LEN]
            
            # Padding
            token_ids += [self.vocab['<PAD>']] * (config.MAX_LEN - len(token_ids))
            
            # Predict
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
            
            confidence, pred_idx = torch.max(probs, dim=0)
            confidence = confidence.item()
            category = self.reverse_label_map[pred_idx.item()]
            
            # Fallback rule-based se bassa confidence
            if use_rule_based_fallback and confidence < fallback_threshold:
                logger.info(f"Low confidence ({confidence:.2f}), using rule-based fallback")
                category_fallback = classify_pec_rule_based(text)
                category = category_fallback
            
            conf_label = confidence_to_label(confidence)
            
            return category, confidence, conf_label
        
        except Exception as e:
            logger.error(f"Errore nella predizione: {e}")
            return 'ALTRO', 0.0, 'LOW'
    
    def predict_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Tuple[str, float, str]]:
        """Predici per batch di testi."""
        return [self.predict(text, **kwargs) for text in texts]

def generate_report(
    pecs_data: List[Dict],
    output_path: Path = config.REPORTS_DIR / 'report_pec_ai.xlsx',
    model_dir: Path = config.MODELS_DIR
) -> pd.DataFrame:
    """
    Genera report Excel da PEC.
    
    Args:
        pecs_data: Lista di {'subject': ..., 'body': ..., ...}
        output_path: Path per xlsx
        model_dir: Path al modello
    
    Returns:
        DataFrame con risultati
    """
    logger.info(f"Generando report per {len(pecs_data)} PEC...")
    
    predictor = PECPredictor(model_dir)
    
    results = []
    for i, pec in enumerate(pecs_data):
        subject = str(pec.get('subject', ''))
        body = str(pec.get('body', ''))
        text = f"{subject} {body}"
        
        category, confidence, conf_label = predictor.predict(text)
        
        results.append({
            'ID': i + 1,
            'Subject': subject[:80],
            'Category': category,
            'Confidence': f"{confidence:.2%}",
            'Confidence_Label': conf_label,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(results)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    
    logger.info(f"Report salvato: {output_path}")
    logger.info(f"\nSummary:\n{df['Category'].value_counts()}")
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, help='Input CSV con PEC')
    parser.add_argument('--output', type=Path, default=config.REPORTS_DIR / 'report.xlsx')
    parser.add_argument('--text', type=str, help='Singolo testo da classificare')
    
    args = parser.parse_args()
    
    if args.text:
        predictor = PECPredictor()
        category, conf, conf_label = predictor.predict(args.text)
        print(f"Category: {category}, Confidence: {conf:.2%}, Label: {conf_label}")
    
    elif args.input:
        from .utils import load_pec_csv
        pecs = load_pec_csv(args.input)
        generate_report(pecs, output_path=args.output)
