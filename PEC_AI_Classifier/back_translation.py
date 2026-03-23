"""
back_translation.py
Esempio: back-translation (IT -> EN -> IT) per generare parafrasi.
Requisiti: nlpaug, torch, transformers
"""

from typing import List
import logging

try:
    import nlpaug.augmenter.word as naw_word  # possibilità futuri
    import nlpaug.augmenter.sentence as naw
except Exception as e:
    raise ImportError("nlpaug non trovato. Installa con: pip install nlpaug") from e

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = False) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_backtranslation_augmenter(device: str = "cpu",
                                     from_model_name: str = "Helsinki-NLP/opus-mt-it-en",
                                     to_model_name: str = "Helsinki-NLP/opus-mt-en-it"):
    """
    Crea l'augmenter di back-translation.
    device: "cpu" o "cuda"
    """
    try:
        bt = naw.BackTranslationAug(
            from_model_name=from_model_name,
            to_model_name=to_model_name,
            device=device
        )
    except Exception as e:
        logger.exception("Errore nella creazione del BackTranslationAug.")
        raise
    return bt


def augment_texts(texts: List[str], augmenter, n: int = 1) -> List[List[str]]:
    """
    Applica l'augmenter a una lista di testi.
    Restituisce lista di liste: per ciascun input, n output (o meno se errore).
    """
    results = []
    for idx, t in enumerate(texts):
        try:
            # nlpaug restituisce lista di stringhe se n>1
            augmented = augmenter.augment(t, n=n)
            # Normalize: sempre lista
            if isinstance(augmented, str):
                augmented = [augmented]
            results.append(augmented)
        except Exception:
            logger.exception("Fallita l'augmentazione del testo index=%s", idx)
            results.append([])
    return results


def main():
    # Esempio d'uso
    prefer_gpu = False  # Imposta True se vuoi usare GPU e torch.cuda disponibile
    device = get_device(prefer_gpu)

    bt_aug = create_backtranslation_augmenter(device=device)

    # Esempio di input (PEC sample di esempio)
    pec_samples = [
        "Fattura Elettronica n.123/2026 - Pagamento da effettuare entro 30 giorni, importo totale €1.200 IVA 22%."
    ]

    # Genera 2 parafrasi per ogni testo (lento: carica i modelli)
    augmented = augment_texts(pec_samples, bt_aug, n=2)

    for orig, aug_list in zip(pec_samples, augmented):
        print("Orig:", orig)
        for i, a in enumerate(aug_list, start=1):
            print(f"BackTranslation {i}:", a)


if __name__ == "__main__":
    main()

