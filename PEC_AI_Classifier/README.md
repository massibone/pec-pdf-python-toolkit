# Classificatore LSTM PyTorch per PEC

## Installazione
```bash
pip install -r requirements.txt
python -m spacy download it_core_news_sm
Training (su dataset reale)
Copy
python pec_ai_classifier/train.py --data pec_training_data.csv --epochs 20
Inference
Copy
python pec_ai_classifier/classifier_pytorch.py --input pecs.xlsx
Output
subject	category	confidence	confidence_label
Fattura 123	        FATTURA    	0.95	HIGH


**Integrazioni**:
- Collega al tuo [pec-pdf-python-toolkit](https://github.com/massibone/pec-pdf-python-toolkit) per full pipeline (download → classify → report).
- Espandi: addestra modello scikit-learn su dataset PEC reali.


