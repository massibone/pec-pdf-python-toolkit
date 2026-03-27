# pec_ai_classifier.py - Classificatore AI per PEC (snippet settimanale Python-Office-PA-Toolkit)
"""
Classificatore NLP per categorizzare PEC usando spaCy.
Categorie: FATTURA, GARA, COMUNICAZIONE, ALTRO.
Integra con tool PEC esistenti (es. pec-pdf-python-toolkit).
"""

import spacy
import pandas as pd
from typing import List, Dict
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("it_core_news_sm")  # pip install spacy && python -m spacy download it_core_news_sm
except OSError:
    logger.error("Scarica it_core_news_sm: python -m spacy download it_core_news_sm")
    raise

def classify_pec(text: str) -> str:
    """
    Classifica testo PEC con NLP (keyword + entità).

    Args:
        text: Oggetto + corpo PEC.

    Returns:
        Categoria PEC.
    """
    doc = nlp(text.lower())
    
    # Keyword + entità recognition
    fattura_kws = {'fattura', 'bonifico', 'iva', 'totale', 'pagamento'}
    gara_kws = {'gara', 'bando', 'appalto', 'offerta'}
    comm_kws = {'comunicazione', 'protocoll', 'notifica', 'aggiornamento'}
    
    if any(kw in text.lower() for kw in fattura_kws) or any(ent.label_ == 'MONEY' for ent in doc.ents):
        return 'FATTURA'
    elif any(kw in text.lower() for kw in gara_kws):
        return 'GARA'
    elif any(kw in text.lower() for kw in comm_kws):
        return 'COMUNICAZIONE'
    return 'ALTRO'

def generate_pec_report(pecs: List[Dict[str, str]], output: Path = Path('report_pec_ai.xlsx')) -> pd.DataFrame:
    """
    Genera report Excel da lista PEC.

    Args:
        pecs: [{'subject': str, 'body': str}]
        output: File Excel.

    Returns:
        DataFrame report.
    """
    results = []
    for pec in pecs:
        category = classify_pec(f"{pec['subject']} {pec['body']}")
        results.append({
            'subject': pec['subject'][:50],  # Truncate
            'category': category,
            'timestamp': pd.Timestamp.now(),
            'confidence': 'high' if category != 'ALTRO' else 'low'  # Placeholder ML
        })
    
    df = pd.DataFrame(results)
    df.to_excel(output, index=False)
    logger.info(f"Report salvato: {output}")
    return df

# Esempio
if __name__ == "__main__":
    sample_pecs = [
        {'subject': 'Fattura Elettronica n.123', 'body': 'Pagamento entro 30gg, totale €1.200 IVA incl.'},
        {'subject': 'Bando Gara Pubblica', 'body': 'Scadenza offerte 15/03/2026.'},
        {'subject': 'Comunicazione protocollata', 'body': 'Aggiornamento pratica n.456.'}
    ]
    report = generate_pec_report(sample_pecs)
    print(report)
----
output.md

**Output esempio**:
| subject                  | category     | timestamp           | confidence |
|--------------------------|--------------|---------------------|------------|
| Fattura Elettronica...  | FATTURA     | 2026-02-20 09:00   | high      |
| Bando Gara Pubblica     | GARA        | 2026-02-20 09:00   | high      |

