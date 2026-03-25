# 🗺️ Mappa PEC_AI_Classifier

--

### 🔍 Analisi Tecnica: Suddivisione dei File per Funzione

| Categoria | File Principali | Cosa fanno? |
| :--- | :--- | :--- |
| **🧠 Il Motore (Core AI)** | `classifier_pytorch.py`, `train.py`, `inference.py` | Definiscono la rete neurale LSTM, gestiscono l'addestramento e le predizioni. |
| **🛠️ La Fabbrica (Augmentation)** | `pec_augmentation.py`, `KeyboardAug.py`, `Word_level.py`, `back_translation.py` | Creano migliaia di varianti delle tue PEC (errori, sinonimi, parafrasi) per rendere il modello robusto. |
| **⚙️ Infrastruttura** | `config.py`, `dataset.py`, `utils.py`, `__init__.py` | Supporto tecnico: configurazioni, caricamento dati e utility comuni. |
| **📄 Documentazione** | `README.md`, `example_usage.py` | Spiegano come installare e usare il progetto. |

---

### 💡 Consigli  (Pulizia e Organizzazione)

*   **File Duplicati**: `nlpaug.py` e `pec_augmentation.py` sembrano sovrapporsi. Ti consiglio di tenere `pec_augmentation.py` come script principale per generare il dataset e usare gli altri (`KeyboardAug.py`, ecc.) come moduli di supporto.

*   **Dataset**: Creato `pec_dataset_augmented.csv` da dare in pasto a `train.py`.

---

