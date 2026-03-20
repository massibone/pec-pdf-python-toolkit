# Word-level: SynonymAug (sinonimi contestuali, BERT-based)

import nlpaug.augmenter.word as naw

# Sinonimi (usa BERT italiano, scarica modello auto)
syn_aug = naw.SynonymAug(aug_src='wordnet', lang='ita')  # O 'bert' per più accuratezza
augmented_syn = syn_aug.augment(pec_samples, n=2)

print("Synonym Augmented:", augmented_syn)

# Output esempio:​

# ['Fattura Elettronica prot.123 - Saldo entro 30 giorni, importo €1.200 IVA 22% compresa.']  # "Pagamento" → "Saldo"
