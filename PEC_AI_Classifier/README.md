#1: Classificatore LSTM PyTorch per PEC, addestrabile.
**Problema**: Smistare manualmente PEC per categoria (fatture da contabilità, gare da acquisti, ecc.) è tempo perso.

**Soluzione**: Classificatore NLP con LSTM Pytorch che analizza oggetto+corpo, categorizza e genera `report_pec_ai.xlsx` pronto per Excel/PowerBI.

**Installa**:
pip install -r requirements.txt
python -m spacy download it_core_news_sm
python pec_ai_classifier.py

text

**Output esempio**:
| subject                  | category     | timestamp           | confidence |
|--------------------------|--------------|---------------------|------------|
| Fattura Elettronica...   | FATTURA      | 2026-02-20 09:00    | high       |
| Bando Gara Pubblica      | GARA         | 2026-02-20 09:00    | high       |

**Integrazioni**:
- Collega al tuo [pec-pdf-python-toolkit](https://github.com/massibone/pec-pdf-python-toolkit) per full pipeline (download → classify → report).
- Espandi: addestra modello scikit-learn su dataset PEC reali.


