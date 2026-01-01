
"""
PEC Automation Toolkit
Automatizza la gestione delle email PEC: lettura, estrazione allegati, categorizzazione
Ideale per uffici PA e studi professionali
"""

import imaplib
import email
from email.header import decode_header
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import json

class PECAutomation:
    def __init__(self, imap_server, email_address, password):
        """
        Inizializza la connessione IMAP
        
        Args:
            imap_server: Server IMAP (es. 'imap.pec.it')
            email_address: Indirizzo email PEC
            password: Password della casella
        """
        self.imap_server = imap_server
        self.email_address = email_address
        self.password = password
        self.mail = None
        self.email_data = []
        
    def connect(self):
        """Connette alla casella PEC"""
        try:
            self.mail = imaplib.IMAP4_SSL(self.imap_server)
            self.mail.login(self.email_address, self.password)
            print(f"✓ Connesso a {self.email_address}")
            return True
        except Exception as e:
            print(f"✗ Errore connessione: {e}")
            return False
    
    def select_folder(self, folder="INBOX"):
        """Seleziona la cartella da leggere"""
        try:
            self.mail.select(folder)
            print(f"✓ Cartella selezionata: {folder}")
            return True
        except Exception as e:
            print(f"✗ Errore selezione cartella: {e}")
            return False
    
    def fetch_emails(self, limit=50, unread_only=False):
        """
        Recupera le email dalla casella
        
        Args:
            limit: Numero massimo di email da recuperare
            unread_only: Se True, recupera solo email non lette
        """
        try:
            # Cerca email
            search_criteria = "UNSEEN" if unread_only else "ALL"
            status, messages = self.mail.search(None, search_criteria)
            email_ids = messages[0].split()
            
            # Limita il numero di email
            email_ids = email_ids[-limit:]
            
            print(f"✓ Trovate {len(email_ids)} email")
            
            for email_id in email_ids:
                self._process_email(email_id)
            
            return len(email_ids)
            
        except Exception as e:
            print(f"✗ Errore recupero email: {e}")
            return 0
    
    def _process_email(self, email_id):
        """Processa una singola email"""
        try:
            status, msg_data = self.mail.fetch(email_id, "(RFC822)")
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Estrae informazioni email
                    subject = self._decode_header(msg["Subject"])
                    from_ = self._decode_header(msg.get("From"))
                    date = msg.get("Date")
                    
                    # Conta allegati
                    attachments = []
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_disposition() == "attachment":
                                filename = part.get_filename()
                                if filename:
                                    attachments.append(self._decode_header(filename))
                    
                    # Salva dati
                    email_info = {
                        "id": email_id.decode(),
                        "data": date,
                        "mittente": from_,
                        "oggetto": subject,
                        "num_allegati": len(attachments),
                        "allegati": ", ".join(attachments) if attachments else "Nessuno",
                        "categoria": self._categorize_email(subject, from_)
                    }
                    
                    self.email_data.append(email_info)
                    
        except Exception as e:
            print(f"✗ Errore processing email {email_id}: {e}")
    
    def _decode_header(self, header):
        """Decodifica header email"""
        if header is None:
            return ""
        
        decoded = decode_header(header)
        header_text = ""
        
        for text, encoding in decoded:
            if isinstance(text, bytes):
                try:
                    header_text += text.decode(encoding or "utf-8")
                except:
                    header_text += text.decode("utf-8", errors="ignore")
            else:
                header_text += str(text)
        
        return header_text
    
    def _categorize_email(self, subject, from_):
        """
        Categorizza l'email in base a oggetto e mittente
        PERSONALIZZA QUESTA FUNZIONE per le tue esigenze!
        """
        subject_lower = subject.lower() if subject else ""
        from_lower = from_.lower() if from_ else ""
        
        # Esempi di categorizzazione - MODIFICA A PIACERE
        if "fattura" in subject_lower or "invoice" in subject_lower:
            return "Fatture"
        elif "protocollo" in subject_lower or "prot." in subject_lower:
            return "Protocollo"
        elif "urgente" in subject_lower or "importante" in subject_lower:
            return "Urgente"
        elif any(x in from_lower for x in ["@pec.", "@legalmail."]):
            return "PEC Ufficiale"
        elif "notifica" in subject_lower:
            return "Notifiche"
        else:
            return "Generale"
    
    def download_attachments(self, output_folder="allegati"):
        """Scarica tutti gli allegati delle email processate"""
        Path(output_folder).mkdir(exist_ok=True)
        downloaded = 0
        
        try:
            for email_info in self.email_data:
                email_id = email_info["id"].encode()
                status, msg_data = self.mail.fetch(email_id, "(RFC822)")
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_disposition() == "attachment":
                                    filename = part.get_filename()
                                    if filename:
                                        filename = self._decode_header(filename)
                                        filepath = os.path.join(output_folder, filename)
                                        
                                        # Gestisce duplicati
                                        counter = 1
                                        base, ext = os.path.splitext(filepath)
                                        while os.path.exists(filepath):
                                            filepath = f"{base}_{counter}{ext}"
                                            counter += 1
                                        
                                        with open(filepath, "wb") as f:
                                            f.write(part.get_payload(decode=True))
                                        
                                        downloaded += 1
            
            print(f"✓ Scaricati {downloaded} allegati in '{output_folder}'")
            return downloaded
            
        except Exception as e:
            print(f"✗ Errore download allegati: {e}")
            return downloaded
    
    def export_to_excel(self, filename="report_pec.xlsx"):
        """Esporta i dati in Excel"""
        try:
            df = pd.DataFrame(self.email_data)
            
            # Riordina colonne
            columns_order = ["data", "categoria", "mittente", "oggetto", "num_allegati", "allegati"]
            df = df[columns_order]
            
            # Esporta
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"✓ Report Excel salvato: {filename}")
            
            # Stampa statistiche
            print("\n📊 STATISTICHE:")
            print(f"   Totale email: {len(df)}")
            print(f"   Email con allegati: {len(df[df['num_allegati'] > 0])}")
            print("\n   Per categoria:")
            print(df['categoria'].value_counts().to_string())
            
            return True
            
        except Exception as e:
            print(f"✗ Errore esportazione Excel: {e}")
            return False
    
    def close(self):
        """Chiude la connessione"""
        if self.mail:
            self.mail.close()
            self.mail.logout()
            print("✓ Connessione chiusa")


def main():
    """
    Esempio di utilizzo
    IMPORTANTE: NON committare mai le credenziali reali su GitHub!
    Usa variabili d'ambiente o file config.json (in .gitignore)
    """
    
    # OPZIONE 1: Usa file config.json (RACCOMANDATO)
    # Crea un file config.json con:
    # {
    #   "imap_server": "imap.tuoserver.it",
    #   "email": "tua@pec.it",
    #   "password": "tuapassword"
    # }
    
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        IMAP_SERVER = config["imap_server"]
        EMAIL = config["email"]
        PASSWORD = config["password"]
        
    except FileNotFoundError:
        print("⚠️  File config.json non trovato!")
        print("Crea un file config.json con le tue credenziali (vedi esempio in README)")
        print("\nUSO DEMO MODE con credenziali di esempio...")
        
        # OPZIONE 2: Demo mode (solo per test struttura)
        IMAP_SERVER = "imap.example.com"
        EMAIL = "esempio@pec.it"
        PASSWORD = "password_esempio"
    
    # Inizializza
    pec = PECAutomation(IMAP_SERVER, EMAIL, PASSWORD)
    
    # Connetti
    if pec.connect():
        # Seleziona cartella
        pec.select_folder("INBOX")
        
        # Recupera ultime 20 email
        pec.fetch_emails(limit=20, unread_only=False)
        
        # Scarica allegati
        pec.download_attachments(output_folder="allegati_pec")
        
        # Esporta report Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pec.export_to_excel(f"report_pec_{timestamp}.xlsx")
        
        # Chiudi connessione
        pec.close()


if __name__ == "__main__":
    main()
  
