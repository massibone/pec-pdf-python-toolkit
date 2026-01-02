# Python PEC & PDF Automation Toolkit

Toolkit Python per automatizzare operazioni comuni su email/PEC e documenti PDF.

## Cosa fa
- Estrazione allegati da email/PEC
- Rinomina automatica PDF
- Controlli su file e cartelle
- Script modulari e personalizzabili

## Per chi è
- studi professionali
- uffici amministrativi
- chi gestisce molti documenti manualmente

## Perché esiste
Nasce da problemi reali affrontati ogni giorno nella Pubblica Amministrazione.

## Come usarlo
1. Clona il repository
bash
git clone https://github.com/tuousername/pec-automation-toolkit.git
cd pec-automation-toolkit
2. Installa le dipendenze
bash
pip install -r requirements.txt
3. Configura le credenziali
Crea un file config.json nella root del progetto:
json
{
  "imap_server": "imap.tuoserver.it",
  "email": "tua@pec.it",
  "password": "tuapassword"
}
⚠️ IMPORTANTE: Aggiungi config.json al tuo .gitignore per non condividere le credenziali!
📖 Utilizzo
Utilizzo base
python
from pec_automation import PECAutomation

# Inizializza
pec = PECAutomation("imap.server.it", "tua@pec.it", "password")

# Connetti e processa
if pec.connect():
    pec.select_folder("INBOX")
    pec.fetch_emails(limit=50)
    pec.download_attachments(output_folder="allegati")
    pec.export_to_excel("report.xlsx")
    pec.close()
Esecuzione script completo
bash
python pec_automation.py
Solo email non lette
python
pec.fetch_emails(limit=100, unread_only=True)
🎨 Personalizzazione
Modificare le categorie
Modifica la funzione _categorize_email() nel file pec_automation.py:
python
def _categorize_email(self, subject, from_):
    subject_lower = subject.lower() if subject else ""
    
    if "fattura" in subject_lower:
        return "Fatture"
    elif "delibera" in subject_lower:
        return "Delibere"
    # Aggiungi le tue regole...
Cartelle diverse da INBOX
python
pec.select_folder("Archive")  # o "Sent", "Drafts", ecc.

Output
Report Excel generato
Il file Excel contiene:
Data e ora ricezione
Categoria automatica
Mittente
Oggetto
Numero allegati
Nomi degli allegati
Statistiche aggregate
## Contatti
Disponibile per adattamenti e script su misura.
LinkedIn: []
GitHub: [@]
