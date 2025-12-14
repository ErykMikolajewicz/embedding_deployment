# Embedding Model Deployment Experiments

## Opis projektu

Projekt powstał z potrzeby znalezienia bardziej efektywnego sposobu budowania obrazów kontenerów z modelami do generowania embeddingów. Dotychczasowe podejście oparte na **Hugging Face** oraz **sentence-transformers**, choć wygodne w użyciu, prowadziło do tworzenia obrazów o **nieakceptowalnie dużych rozmiarach** (rzędu 12 GB).

Główną przyczyną tego problemu była konieczność dołączania bibliotek do uczenia maszynowego, takich jak:
- PyTorch  
- Triton

## Cel projektu

Celem projektu jest:
- przetestowanie metod budowy obrazów kontenerów,
- redukcja rozmiaru obrazów,
- porównanie wydajności różnych środowisk i sposobów wdrożenia.

## Wykorzystywany model

W ramach projektu wykorzystywany jest model embeddingowy **gemma300m**.

Model ten został wybrany ze względu na:
- **uniwersalność** nie tylko semantic search, ale też np. clustering,
- dobre wyniki w rankingach,
- relatywnie **niewielki rozmiar**, który umożliwia sensowne wdrożenie na:
  - CPU,
  - GPU.

## Aktualny stan projektu

Obecnie przygotowane zostały:
- skrypty do budowy obrazów kontenerów,
- skrypty wdrażające model z użyciem CPU na:
  - **Ollama**,
  - **ONNX Runtime**,

### Kwantyzacja modeli

Proces budowy obrazów umożliwia wybór różnych wariantów kwantyzacji, m.in.:
- `int8`,
- `fp16`.

## Plany rozwoju

W kolejnych etapach projektu planowane jest:
- porównanie **wydajności różnych opcji wdrożeniowych** modeli embeddingowych,
- testy dodatkowych środowisk, w tym:
  - `sentence-transformers`,
  - `llama.cpp`,
- przygotowanie skryptów umożliwiających:
  - wdrożenie modeli na **GPU**

## Możliwe rozszerzenia

Rozważane jest również rozszerzenie projektu o:
- deployment w środowisku **Google Cloud Platform (GCP)**
