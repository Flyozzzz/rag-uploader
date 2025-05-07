# Generic RAG Uploader API

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-%3E%3D0.110-green) ![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.x-orange)

Полнофункциональный сервис для **пакетной загрузки** документов в Elasticsearch (dense‑vector k‑NN) с автоматической генерацией эмбеддингов. Предназначен для ускоренного построения Retrieval‑Augmented Generation (RAG) приложений.

---

## Содержание

1. [Основные возможности](#основные-возможности)
2. [Архитектура](#архитектура)
3. [Быстрый старт](#быстрый-старт)
4. [API](#api)
5. [Конфигурация `UploadCfg`](#конфигурация-uploadcfg)
6. [Сплиттеры](#сплиттеры)
7. [Эмбеддинги](#эмбеддинги)
8. [Elasticsearch Vector Store](#elasticsearch-vector-store)
9. [Примеры запросов](#примеры-запросов)
10. [Тестирование](#тестирование)
11. [Changelog](#changelog)

---

## Основные возможности

* 🔨 **Пакетная загрузка**: приём нескольких файлов (PDF, DOCX, HTML, MD, TXT, PPTX и др.) одним запросом.
* 📄 **Unstructured Loader**: три режима (`single`, `elements`, `paged`) + передача дополнительных аргументов.
* ✂️ **Гибкая нарезка (сплиттинг)**: более 20 встроенных сплиттеров (см. таблицу ниже), настраиваемые `chunk_size` и `chunk_overlap`.
* 🧠 **Эмбеддинги**: OpenAI *text‑embedding‑3‑small* / HuggingFace *all‑mpnet‑base‑v2* и любые другие Sentence‑Transformers.
* ⚡ **k‑NN поиск** на Elasticsearch 8 (HNSW + BM25 / RRF hybrid).
* 🐳 **Docker Compose**: «поднял и забыл» — API + однострочный Elasticsearch‑кластер.
* 🔄 **Upsert / bulk‑insert**: обновление существующих документов или массовая вставка.

---

## Архитектура

```
┌──────────┐      ┌──────────────┐      ┌────────────────────────┐      ┌─────────────────┐
│  Client  ├────▶│  FastAPI /upload │──▶│  LangChain Splitter      │──▶│  Elasticsearch 8 │
└──────────┘      └──────────────┘      └────────────────────────┘      └─────────────────┘
                                 ▲                  │
                                 │                  ▼
                         ┌───────────────────┐  ┌──────────────┐
                         │ UnstructuredLoader │  │  Embeddings  │
                         └───────────────────┘  └──────────────┘
```

---

## Быстрый старт

```bash
# 1. Клонировать репозиторий
$ git clone https://github.com/Flyozzzz/rag-uploader && cd rag-uploader

# 2. Поднять сервис + однострочный ES‑кластер
$ docker compose up -d --build

# 3. Загрузить файл
$ curl -X POST localhost:8000/upload \
       -F file=@docs/handbook.pdf \
       -F json='@example/cfg.json'
```

По умолчанию API — **`http://localhost:8000`**, Elasticsearch — **`http://localhost:9200`**.

### Запуск без Docker

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API

| Метод  | URL       | Описание                                                                                                            |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/upload` | `multipart/form-data`: один или несколько `file` + поле `json` с конфигом. Разбивает, эмбеддит и индексирует чанки. |

### Жизненный цикл `/upload`

1. Сохранение файла во временную директорию.
2. Парсинг контента `UnstructuredFileLoader`‑ом.
3. Разбиение через выбранный сплиттер.
4. Генерация эмбеддингов.
5. Индексация `ElasticsearchStore.from_documents()`.

Пример ответа:

```json
{
  "chunks": 128,
  "index": "rag_docs",
  "took_ms": 1748
}
```

---

## Конфигурация `UploadCfg`

```jsonc
{
  "index_name": "rag_docs",
  "vector_field": "embedding",       // → vector_query_field
  "distance": "COSINE",              // DOT_PRODUCT | EUCLIDEAN_DISTANCE

  "loader": {
    "mode": "elements",             // single | elements | paged
    "kwargs": {}
  },
  "splitter": {
    "type": "RecursiveCharacterTextSplitter",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "kwargs": {}
  },
  "embedding": {
    "provider": "openai",           // openai | huggingface
    "model": "text-embedding-3-small",
    "kwargs": {}
  },
  "elasticsearch": {
    "hosts": ["http://localhost:9200"],
    "api_key": "",
    "username": "",
    "password": "",
    "cloud_id": "",
    "kwargs": {}
  }
}
```

> **Важно.** В `langchain‑elasticsearch ≥ 0.2` параметры `es_client`, `vector_field`, `dims` удалены/переименованы. Используйте `es_connection`, `vector_query_field`; размерность выводится из первой вставки.

---

## Сплиттеры

### Зачем они нужны?

LLM‑модели ограничены контекстом. Чанки позволяют:

* передавать в промпт только релевантные куски;
* уменьшать стоимость эмбеддингов и inference;
* снижать риск «галлюцинаций».

### Полный список

| Класс                                    | Назначение               | Особенности                                                  |
| ---------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| `RecursiveCharacterTextSplitter`         | Универсальный plain‑text | Рекурсивное деление по «параграф → строка → слово → символ». |
| `CharacterTextSplitter`                  | Плоский текст            | Быстрый split по фиксированному разделителю.                 |
| `TokenTextSplitter`                      | Токено‑базовый           | Деление по кол‑ву токенов выбранного токенизатора.           |
| `SentenceTransformersTokenTextSplitter`  | ST‑модели                | Учитывает true‑tokenizer Sentence‑Transformers.              |
| `SpacyTextSplitter`                      | Разные языки             | Sentence boundary detection spaCy.                           |
| `NLTKTextSplitter`                       | Лингвистические тексты   | Sentence tokenize с NLTK.                                    |
| `KonlpyTextSplitter`                     | Корейский                | Морфемный tokenizer Komoran/Mecab.                           |
| `LatexTextSplitter`                      | LaTeX                    | Не ломает окружения `\begin{}` / `\end{}`.                   |
| `PythonCodeTextSplitter`                 | Python‑код               | Учитывает отступы, def/class и docstrings.                   |
| `JSFrameworkTextSplitter`                | React / Vue / Svelte     | Деление по компонентам JSX/TSX.                              |
| `HTMLHeaderTextSplitter`                 | HTML с `<h1‑h6>`         | Сохраняет иерархию заголовков.                               |
| `HTMLSectionSplitter`                    | HTML статьи              | Группирует контент по секциям.                               |
| `HTMLSemanticPreservingSplitter`         | Сложный HTML             | Не ломает таблицы, списки, медиа.                            |
| `MarkdownHeaderTextSplitter`             | Markdown + заголовки     | Выделяет секции по `#`.                                      |
| `MarkdownTextSplitter`                   | Markdown без секц.       | Деление по строкам/символам с учётом списков.                |
| `ExperimentalMarkdownSyntaxTextSplitter` | MD (beta)                | Тонкая сегментация синтаксиса.                               |
| `RecursiveJsonSplitter`                  | Большие JSON             | DFS‑обход, сохранение вложенности.                           |

> **NB:** `MarkdownHeaderTextSplitter` и `HTMLHeaderTextSplitter` не наследуются от `TextSplitter`, но предоставляют тот же API.

---

## Эмбеддинги

| Провайдер   | Модель                   | Преимущества                 |
| ----------- | ------------------------ | ---------------------------- |
| OpenAI      | `text-embedding-3-small` | Высокое качество, zero‑shot. |
| HuggingFace | `all-mpnet-base-v2`      | Оффлайн, GDPR‑friendly.      |

---

## Elasticsearch Vector Store

Класс `ElasticsearchStore` (LangChain):

* HNSW k‑NN индексация;
* гибрид BM25 + ANN (`hybrid=True`);
* метрики: Cosine / Dot‑Product / Euclidean.

```python
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

es = Elasticsearch("http://localhost:9200")
vs = ElasticsearchStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    index_name="rag_docs",
    es_connection=es,
    vector_query_field="embedding",
    distance_strategy="COSINE",
)
```

---

## Примеры запросов

### cURL

```bash
curl -X POST localhost:8000/upload \
     -F file=@/path/report.docx \
     -F json='{"index_name":"reports","vector_field":"embedding"}'
```

### Python‑клиент

```python
import json, requests

with open("analytics.pdf", "rb") as f:
    r = requests.post(
        "http://localhost:8000/upload",
        files={"file": f},
        data={"json": json.dumps(cfg_dict)},
        timeout=180,
    )
print(r.json())
```

---

## Тестирование

```bash
pytest -q tests/
```

---

## Changelog

### 0.2 → 0.3 (May 2025)

* Перешли на **`langchain-elasticsearch 0.2`**.
* **BREAKING:** `vector_field` → `vector_query_field`, `es_client` → `es_connection`, удалён `dims`.
* Добавлен `distance_strategy`, гибридный поиск, расширенная таблица сплиттеров.
* README полностью переработан и актуализирован.

---

## Планы развития

* 🔍 Поддержка других векторных БД (Milvus/Qdrant).
* 🚀 Асинхронная вставка с batch embeddings.
* 🔧 Swagger‑документация модели конфигурации.
* 🌐 Helm‑chart для Kubernetes‑развёртывания.

---

© 2025 Generic RAG Uploader — MIT License.

