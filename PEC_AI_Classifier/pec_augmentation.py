"""
pec_augmentation.py
Genera dataset augmented a partire da liste di PEC (subject+body).
Output: DataFrame pandas e salvataggio CSV pronto per training PyTorch.
Requisiti: nlpaug, pandas
"""

from typing import List, Dict, Optional
import logging
import math

import pandas as pd

# nlpaug augmenters (char + word)
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def augment_pec_dataset(
    pecs: List[Dict[str, str]],
    num_aug: int = 5,
    augmenters: Optional[List] = None,
    out_csv: Optional[str] = "pec_dataset_augmented.csv"
) -> pd.DataFrame:
    """
    Genera dataset augmented.
    - pecs: lista di dict con chiavi almeno 'subject' e 'body'. 'category' opzionale.
    - num_aug: numero totale di augmentazioni per esempio originale (>=0).
    - augmenters: lista opzionale di oggetti nlpaug. Se None, si usano quelli di default.
    - out_csv: percorso file di output CSV (None per non salvare).
    Restituisce DataFrame con colonne ['text','label','is_augmented'].
    """

    if not pecs:
        logger.warning("Nessuna PEC fornita.")
        return pd.DataFrame(columns=['text', 'label', 'is_augmented'])

    # Default augmenters (moderati per testi formali)
    if augmenters is None:
        augmenters = [
            nac.KeyboardAug(action="substitute", aug_p=0.03),  # errori di battitura
            nac.OcrAug(aug_p=0.03),                           # errori OCR/sostituzioni
            naw.SynonymAug(lang='ita')                        # sinonimi (NLPAug)
        ]

    texts = [f"{p.get('subject','')} {p.get('body','')}".strip() for p in pecs]
    labels = [p.get('category', 'ALTRO') for p in pecs]

    augmented_texts = []
    augmented_labels = []

    # Calcola quante augment per augmenter per esempio (distribuzione uniforme)
    k = max(1, len(augmenters))
    per_aug = num_aug // k
    remainder = num_aug % k

    for idx, (text, label) in enumerate(zip(texts, labels)):
        # Keep original
        augmented_texts.append(text)
        augmented_labels.append(label)

        # Apply each augmenter
        for i, aug in enumerate(augmenters):
            times = per_aug + (1 if i < remainder else 0)
            for _ in range(times):
                try:
                    aug_out = aug.augment(text)
                    aug_out_list = _ensure_list(aug_out)
                    # scegli primo elemento valido non vuoto
                    aug_text = next((s for s in aug_out_list if isinstance(s, str) and s.strip()), None)
                    if aug_text:
                        augmented_texts.append(aug_text)
                        augmented_labels.append(label)
                    else:
                        logger.debug("Augmenter %s ha restituito output vuoto per index %d", type(aug).__name__, idx)
                except Exception as e:
                    logger.exception("Errore durante augmentazione con %s per index %d: %s", type(aug).__name__, idx, e)

    df = pd.DataFrame({
        'text': augmented_texts,
        'label': augmented_labels,
        'is_augmented': [False] * len(texts) + [True] * (len(augmented_texts) - len(texts))
    })

    if out_csv:
        try:
            df.to_csv(out_csv, index=False)
            logger.info("Salvato CSV: %s (samples=%d)", out_csv, len(df))
        except Exception:
            logger.exception("Impossibile salvare CSV su %s", out_csv)

    logger.info("Dataset generato: originali=%d, tot_augmented=%d, fattore=%.2f",
                len(texts), len(df) - len(texts), len(df) / max(1, len(texts)))
    return df


if __name__ == "__main__":
    # Esempio d'uso
    pecs_data = [
        {'subject': 'Fattura n.123', 'body': 'Pagamento IVA €1000.', 'category': 'FATTURA'},
        {'subject': 'Bando Gara', 'body': 'Offerte entro 30gg.', 'category': 'GARA'}
    ]
    df_aug = augment_pec_dataset(pecs_data, num_aug=6, out_csv='pec_dataset_augmented.csv')
    print(df_aug.head(10))

