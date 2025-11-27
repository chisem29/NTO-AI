# NTO-AI

## Клонирование репозитория 

Сперва убедитесь в том, что у вас установлены ```git, git lfs```, после уже можете запускать 

```bash
git clone https://github.com/chisem29/NTO-AI.git
cd NTO-AI
git lfs pull # для успешной выгрузки больших файлов

```

## Структура проекта
```
.
NTO-AI/
├── main.py                   # Запускаемый файл
├── NTOVOR.ipynb              # Экспериментальная версия
├── data/                     # Исходные данные
│   ├── books.csv
│   ├── book_descriptions.csv
│   ├── book_genres.csv
│   ├── genres.csv
│   ├── test.csv
│   ├── train.csv
│   ├── users.csv
│   └── processed/                      # Обработанные данные
│       ├── processed_features.parquet ✅
│       └── processed_featuresV2.parquet 
├── output/                             # Результаты экспериментов
│   └── models/
│       ├── bert_embeddings.pkl
│       ├── bert_embeddingsV2.pkl ✅
│       ├── LBMGEN.txt
│       ├── LBMGEN2.txt ✅
│       ├── ...
│       ├── MYLBM.txt
│       ├── tfidf_vectorizer.pkl ✅
│       └── tfidf_vectorizerV2.pkl
├── submissions                  # Файлы для отправки
├── Dockerfile                   # Сборщик образа проекта
├── requirements.txt             # Зависимости Python
├── ...
```

## Запуск программы

```powershell
python main.py
```