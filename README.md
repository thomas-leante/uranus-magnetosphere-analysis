# uranus-magnetosphere-analysis
This repository provides data analysis and visualization tools for investigating the plasma environment around Uranus using measurements from the Voyager 2 PLS instrument. Designed to support scientific interpretation and exploration of Uranus’s magnetospheric plasma population.

### 📁 Struttura della repository

La repository include:

- **`requirements.txt`** — contiene le dipendenze necessarie per la creazione dell’ambiente virtuale Python.  
- **`voy2_uranus_plasma_analysis.py`** — script principale che implementa l’intero flusso di lavoro descritto nella relazione: dall’acquisizione automatizzata dei dati tramite API fino alla generazione delle immagini scientifiche e dei dataset finali.  
- **`mlat_function_comparison.py`** — script dedicato alla creazione e al confronto della funzione di conversione dalla latitudine uranocentrica a quella magnetica, basata sul modello OTD di [Ness et al., 1986].  
- **`input/`** — cartella che raccoglie i file indispensabili all’esecuzione degli script:
  - `input/kernel-spice/` — contiene i kernel SPICE.  
  - `vo2MetaK.txt` — meta-kernel di caricamento.  
- **`old-versions/`** — conserva le versioni preliminari del codice, scartate a seguito di modifiche nel flusso di lavoro.  
- **`output/`** — contiene i principali risultati ottenuti: rappresentazioni scientifiche (in formato `.png`) e dataset finali.
