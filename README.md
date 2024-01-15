# Pulizia e classificazione del dataset Foodx-251

## Installazione e uso dell'interfaccia


Installazione dei requirements:
```
pip install -r requirements.txt
```
Avvio dell'interfaccia grafica:
```
python main.py
```

## Strutturazione delle cartelle e dei notebook

- classification: contiene l'addestramento e la valutazione dei vari metodi testati.
    - *cnn_finetuning.ipynb*: notebook relativo al fine tuning della efficientnet
    - *vitb16_finetuning.ipynb*: finetuning del transformer
    - *ensemble.ipynb*: esegue le predizioni sulla efficientnet e sul transformer finetuned.
    - *handcrafted.ipynb*: predizione usando features handcrafted, estratte tramite il notebook *store_handcrafted.ipynb* presente nella root del progetto
-  gui: Piccola interfaccia grafica costuita per permettere una semplice visualizzazione delle 2 funzionalità principali, la classificazione e la category search. Avviabile eseguendo il file *main.py*.
    - *menu.py*: file relativo alla gestione della home dell'applicazione, contiene i tasti per aprire le successive schermate
    - *search.py*: permette di eseguire la category search su un immagine di input e tramite uno slider è possibile regolare il peso attributo alle features estratte da una efficient net senza fine-tuning e a quelle ottenute dalla efficient net dopo il fine-tuning.
    - *classification.py*: caricando un immagine fornisce la lista delle predizioni per le classi più probabili
- outliers_detection: Raccoglie gli script usati per scartare le immagini dal train set. Essi generano un file csv pulito contenente le immagini che vengono effettivamente usate per addestrare i modelli. Le immagini vengono scartate in base alla distanza rispetto le altre immagini della stessa classe.
    - *nn_filtering.ipynb*: scarta le immagini sfruttando features estratte da una rete neurale.
    - *image_discard.ipynb*: file deprecato, esclusione delle immagini basato su features handcrafted. 
- restoration: Contiene notebook per visualizzare l'output del preprocessing
    - *restoration.ipynb*: processa le immagini tramite la pipeline di processing e le salva nel percorso specificato all'inizio del file.
- similarity_search: Contiene gli script usati anche dalla gui per cercare le immagini similari, oltre a ciò è presente anche un notebook per poter visualizzare le immagini simili data una query.
    - *separate_features.ipynb*: estrae le features e le salva in un file
    - *visualize_similarity_search.ipynb*: carica in memoria i modelli e le features salvate per ritornare un collage delle immagini più similari.

