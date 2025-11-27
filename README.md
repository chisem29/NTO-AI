# NTO-AI

## Клонирование репозитория 

Сперва убедитесь в том, что у вас установлены ```git, git lfs```, после уже можете запускать 

```bash
git clone https://github.com/chisem29/NTO-AI.git
cd NTO-AI
git lfs pull # для выгрузки больших файлов

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
├── ...                          # Прочие файлы
```

## Запуск программы

### Установка необходимых либ

Убедитесь в том, что у вас установлен **```PIP```**, тогда
```
pip install -r requirements.txt
```
### Запуск и получение результатов

#### 1. **Run main.py**

Убедитесь в том, что у вас установлен **```Python```**,

```powershell
python main.py
```
В результате выполнения в директорию ```output/submissions``` будет добавлен ```submission.csv```, а также выведена статистика по нему. 

#### 2. **Скачивание submission.csv**

После успешного выполнения предыдущего пункта, вы сможете скачать ```submission.csv```!