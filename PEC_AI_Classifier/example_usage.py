"""Esempi di utilizzo."""
import logging
from pathlib import Path
import pandas as pd
from config import config
from utils import setup_logging, load_pec_csv
from train import train_classifier
from inference import PECPredictor, generate_report

setup_logging()
logger = logging.getLogger(__name__)

def example_1_train_from_csv():
    """Esempio 1: Allenamento da CSV."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training da CSV")
    print("="*60)
    
    csv_path = config.DATA_DIR / 'pec_training_data.csv'
    
    if not csv_path.exists():
        logger.error(f"CSV non trovato: {csv_path}")
        logger.info("Crea un CSV con colonne: subject, body, category (opzionale)")
        return
    
    train_classifier(csv_path)

def example_2_single_prediction():
    """Esempio 2: Predizione singola."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Single Prediction")
    print("="*60)
    
    predictor = PECPredictor()
    
    texts = [
        "Fattura n.123 - Pagamento IVA €1500",
        "Bando Gara Pubblica - Offerte aperte fino al 31/12",
        "Comunicazione ricezione - Protocollo 2026-123",
    ]
    
    for text in texts:
        category, confidence, conf_label = predictor.predict(text)
        print(f"Text: {text[:50]}...")
        print(f"  → Category: {category}, Confidence: {confidence:.2%}, Label: {conf_label}\n")

def example_3_batch_prediction():
    """Esempio 3: Predizione batch."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Prediction & Report")
    print("="*60)
    
    # Dati di esempio
    sample_pecs = [
        {
            'subject': 'Fattura Elettronica n.001/2026',
            'body': 'Importo: €5.000,00. Data scadenza: 31/03/2026. CIG: ABC123'
        },
        {
            'subject': 'Bando di Gara',
            'body': 'Procedura aperta per fornitura servizi. Scadenza offerte: 30/06/2026'
        },
        {
            'subject': 'Comunicazione ricezione documento',
            'body': 'Protocollo n.2026-456. Data ricezione: 20/02/2026'
        },
        {
            'subject': 'Fattura numero 789',
            'body': 'Pagamento entro 60 gg. Cassa: 15000. IVA 22%'
        }
    ]
    
    # Genera report
    df = generate_report(sample_pecs)
    print("\nReport:\n", df.to_string())

def example_4_custom_data():
    """Esempio 4: Usa tuoi dati da CSV."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom CSV Data")
    print("="*60)
    
    csv_path = Path('my_pecs.csv')
    
    if not csv_path.exists():
        logger.warning(f"File non trovato: {csv_path}")
        logger.info("Crea un CSV con colonne: subject, body")
        return
    
    pecs = load_pec_csv(csv_path)
    df = generate_report(pecs)
    print(df)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            example_1_train_from_csv()
        elif example_num == '2':
            example_2_single_prediction()
        elif example_num == '3':
            example_3_batch_prediction()
        elif example_num == '4':
            example_4_custom_data()
        else:
            print("Usage: python example_usage.py [1|2|3|4]")
    else:
        # Run all
        try:
            example_2_single_prediction()
            example_3_batch_prediction()
        except Exception as e:
            logger.error(f"Errore: {e}")
