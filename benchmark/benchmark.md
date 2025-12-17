
# Benchmark modelu embeddingów **gemma-300m**

W ramach benchmarku wykonywano embeddingi, dla artykułów kodeksu postępowania administracyjnego.
Artykuły były posortowane, po długości, w kolejności rosnącej.
Przy każdej wielkości batchy przerabiano cały kodeks postępowania administracyjnego.

## Środowisko testowe

- **Procesor:** 13th Gen Intel Core i7-13650HX
- **Hyperthreading:** wyłączony, akurat tak miałem ustawiony w komputerze
- **System operacyjny:** Linux
- **Wsparcie sprzętowe:**
  - dostępne flagi CPU umożliwiające efektywne obliczenia na liczbach całkowitych **INT8**  
  - brak specjalnego wsparcia sprzętowego dla modelu **FP16**, CPU castuje do **F32**
  - brak flag wskazujących wsparcie dla innych typów, np. **INT4**

## Metodologia

- Liczba pomiarów dla każdego przypadku: **3**
- Wybrano najniższy wynik z każdej próbki
- Mierzony parametr: **czas wykonania**

## Wyniki benchmarku

### Czas wykonania
| Runtime | Batch size | F32 [s] | FP16 [s] | INT8 [s] |
|---------|------------|---------|----------|----------|
| ONNX    | 5          | 19.71   | 19.35    | 10.05    |
| ONNX    | 10         | 22.23   | 23.93    | 11.47    |
| ONNX    | 20         | 25.21   | 27.48    | 12.99    |
| ONNX    | 50         | 35.88   | 39.37    | 19.23    |
| ollama  | 5          | -       | 39.71    | 39.86    |
| ollama  | 10         | -       | 41.70    | 41.16    |
| ollama  | 20         | -       | 41.74    | 40.99    |
| ollama  | 50         | -       | 42.27    | 40.25    |

### Wielkość obrazu
| Runtime   | F32    | FP16   | INT8   |
|-----------|--------|--------|--------|
| ONNX      | 1.59GB | 973MB  | 668MB  |
| ollama    | -      | 4.57GB | 4.29GB |


## Wnioski

### Onnx
- Wyraźnie widoczny efekt batchowania. Dla mniejszych batchy wykonanie było wyraźnie szybsze.
- **INT8** jest wyraźnie najszybszy, zgodnie z oczekiwaniami
- **FP16** minimalnie wolniejszy od **F32**, brak wsparcia, może i tak wymuszać castowanie w procesorze do **F32**

### Ollama
- Wyraźnie wolniejszy niż onnx w każdej konfiguracji, wyniki zdecydowanie rozczarowują
- Niewielka korzyść z mniejszych batchy, słabo wpływający parametr
- Nie udało się wykorzystać wydajnych instrukcji do obliczeń na int 8 lub korzyści zostały przykryte przez inne straty.

**Podsumowanie:**
W przypadku onn udało się istotnie zmniejszyć wielkość obrazu. Jednocześnie uzyskano dobrą wydajność, zwłaszcza dla int8.
Gorzej sytuacja wygląda w przypadku ollamy. Obraz nadal był stosunkowo duży - ponad 4GB. Ponadto wydajność była stosunkowo niska.
Jedyny plus z wykorzystania ollamy to względna prostota, która szła jednak w parze ze znaczną utratą elastyczności.
