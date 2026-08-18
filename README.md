# PEC & PDF Automation Toolkit

Toolkit Python per automatizzare workflow documentali legati a PEC, allegati e PDF in contesti amministrativi, studi professionali e uffici che gestiscono grandi volumi di documenti. Il progetto nasce per: ridurre attività manuali ripetitive, migliorare l'organizzazione dei file e velocizzare la reportistica. 

## Panoramica


Questo repository raccoglie script modulari per semplificare attività comuni come:

- estrazione allegati da email/PEC;
- organizzazione dei file scaricati;
- rinomina automatica dei PDF;
- classificazione preliminare dei contenuti;
- esportazione di report Excel per controllo e monitoraggio.


## Funzionalità principali

- Download allegati da email/PEC.
- Selezione della cartella IMAP da elaborare.
- Filtri base sul numero di email o sui messaggi non letti.
- Estrazione e salvataggio degli allegati in cartelle dedicate.
- Rinomina e organizzazione dei PDF.
- Esportazione dati in Excel per reportistica e controllo operativo.
- Struttura modulare, pensata per futuri adattamenti o integrazioni.

## Struttura del progetto

pec-pdf-python-toolkit/
├─ PEC_AI_Classifier/
├─ allegati_pec/
├─ pec_automation.py
├─ config.example.json
├─ README.md
└─ report_pec_*.xlsx


## Output generato

A seconda dello script eseguito, il toolkit può produrre:

- cartelle con allegati estratti;
- PDF rinominati e organizzati;
- file Excel con dati riepilogativi;
- informazioni utili per audit interno o controllo documentale.

Un report Excel tipico può includere:

- data e ora di ricezione;
- mittente;
- oggetto;
- categoria assegnata;
- numero allegati;
- nomi dei file scaricati;
- statistiche aggregate.


## Installazione

Clonare il repository:


git clone https://github.com/massibone/pec-pdf-python-toolkit.git
cd pec-pdf-python-toolkit


Installare le dipendenze:


pip install -r requirements.txt


Creare e attivare un ambiente virtuale è consigliato, soprattutto per mantenere separate le dipendenze del progetto.

## Configurazione

Per motivi di sicurezza, le credenziali non dovrebbero mai essere salvate direttamente in un file versionato nel repository pubblico. La pratica consigliata è usare un file di esempio versionato e un file reale ignorato da Git. 
Creare un file `config.json` locale partendo da `config.example.json`:

```json
{
  "imap_server": "imap.example.it",
  "email": "nome@pec.it",
  "password": "inserisci-qui-la-password"
}
```

Aggiungere `config.json` al file `.gitignore` per evitare la pubblicazione accidentale di credenziali sensibili. 

## Utilizzo rapido

Esempio base:


from pec_automation import PECAutomation

pec = PECAutomation("imap.server.it", "tua@pec.it", "password")

if pec.connect():
    pec.select_folder("INBOX")
    pec.fetch_emails(limit=50)
    pec.download_attachments(output_folder="allegati")
    pec.export_to_excel("report.xlsx")
    pec.close()


Esecuzione dello script principale:


python pec_automation.py


Elaborazione dei soli messaggi non letti:


pec.fetch_emails(limit=100, unread_only=True)


## Personalizzazione

Il toolkit è pensato per essere adattato facilmente a flussi diversi. Alcuni esempi di personalizzazione:

- regole di categorizzazione per oggetto o mittente;
- cartelle IMAP diverse da `INBOX`;
- struttura di salvataggio allegati personalizzata;
- naming convention dedicate per i PDF;
- export Excel con colonne o statistiche aggiuntive.

Esempio di logica personalizzata per la categorizzazione:


def categorize_email(subject, sender):
    subject_lower = subject.lower() if subject else ""

    if "fattura" in subject_lower:
        return "Fatture"
    elif "delibera" in subject_lower:
        return "Delibere"
    else:
        return "Altro"


## Roadmap

Possibili evoluzioni future del progetto:

- classificazione documentale più avanzata;
- estrazione metadati da PDF;
- dashboard riepilogative;
- integrazione con API o servizi documentali;
- miglioramento logging e tracciabilità delle operazioni;
- packaging CLI per utilizzo più semplice.

## Contatti

Per collaborazioni, personalizzazioni o adattamenti su workflow documentali e automazioni Python:

- GitHub: [massibone](https://github.com/massibone)
- LinkedIn: massimo-bonechi

## Supporto

Se il progetto ti è utile:

- lascia una stella al repository;
- apri una Issue per bug o miglioramenti;
- proponi estensioni o casi d'uso reali.
