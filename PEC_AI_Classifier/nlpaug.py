# augmentation character-level (errori OCR/keyboard)

import nlpaug.augmenter.char as nac
import pandas as pd

# Sample PEC reali/anonime
pec_samples = [
    "Fattura Elettronica n.123/2026 - Pagamento entro 30gg, totale €1.200 IVA 22% incl.",
    "Bando Gara Pubblica Appalti - Scadenza offerte 15/03/2026 alle ore 12:00.",
    "Comunicazione protocollata n.456 - Aggiornamento pratica amministrativa."
]

# Augmenter OCR (simula errori scansione/OCR)
ocr_aug = nac.OcrAug(aug_char_p=0.1, aug_max=5)  # 10% char, max 5 per testo
augmented_ocr = ocr_aug.augment(pec_samples, n=2)  # 2 varianti per campione

print("Originali:", pec_samples)
print("OCR Augmented:", augmented_ocr)

# Output esempio:​
'''
Originali: ['Fattura Elettronica n.123...', ...]
OCR Augmented: [
  ['Fattura Eletteronica n.123...', 'Fattuta Elettronica n.l23...'],  # Variante 1 e 2 per prima PEC
  ['Bando Gara Pubbliea Appalti...', ...],
  ...
]
'''
