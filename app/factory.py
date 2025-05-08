from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import OpenAIEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from importlib import import_module
from typing import Any
from langchain_text_splitters.base import TextSplitter
from app.config import UploadCfg, QueryCfg
import tempfile, pathlib

def build_loader(path: str, cfg: UploadCfg):
    return UnstructuredFileLoader(path, mode=cfg.loader.mode, **cfg.loader.kwargs)

_SPLITTER_MODULES = {
    # Character-based
    "CharacterTextSplitter": "langchain_text_splitters.character",
    "RecursiveCharacterTextSplitter": "langchain_text_splitters.character",
    "TokenTextSplitter": "langchain_text_splitters.base",
    "SentenceTransformersTokenTextSplitter": "langchain_text_splitters.sentence_transformers",

    # Linguistic
    "SpacyTextSplitter": "langchain_text_splitters.spacy",
    "NLTKTextSplitter": "langchain_text_splitters.nltk",
    "KonlpyTextSplitter": "langchain_text_splitters.konlpy",

    # Documents & code
    "LatexTextSplitter": "langchain_text_splitters.latex",
    "PythonCodeTextSplitter": "langchain_text_splitters.python",
    "JSFrameworkTextSplitter": "langchain_text_splitters.jsx",
    "RecursiveJsonSplitter": "langchain_text_splitters.json",

    # Markdown / HTML
    "MarkdownTextSplitter": "langchain_text_splitters.markdown",
    "ExperimentalMarkdownSyntaxTextSplitter": "langchain_text_splitters.markdown",
    "MarkdownHeaderTextSplitter": "langchain_text_splitters.markdown",
    "HTMLHeaderTextSplitter": "langchain_text_splitters.html",
    "HTMLSectionSplitter": "langchain_text_splitters.html",
    "HTMLSemanticPreservingSplitter": "langchain_text_splitters.html",
}


def _resolve_splitter(name: str):
    """Dynamically import class by name."""
    module_path = _SPLITTER_MODULES.get(name)
    if not module_path:
        raise ValueError(f"Unknown splitter '{name}'. "
                         f"Valid options: {', '.join(_SPLITTER_MODULES)}")
    module = import_module(module_path)
    return getattr(module, name)


def build_splitter(cfg: UploadCfg) -> Any:
    """
    Build any text-splitter from cfg.splitter.type
    Special-case header splitters that don't accept chunk_* kwargs.
    """
    splitter_cls = _resolve_splitter(cfg.splitter.type)

    # Разделители заголовков имеют собственную сигнатуру (без chunk_size/overlap).
    header_splitters = {
        "MarkdownHeaderTextSplitter",
        "HTMLHeaderTextSplitter"
    }

    if splitter_cls.__name__ in header_splitters:
        return splitter_cls(**cfg.splitter.kwargs)

    if issubclass(splitter_cls, TextSplitter):
        return splitter_cls(
            chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap,
            **cfg.splitter.kwargs,
        )

    return splitter_cls(**cfg.splitter.kwargs)


def build_embeddings(cfg: UploadCfg):
    if cfg.embedding.provider == "openai":
        return OpenAIEmbeddings(model=cfg.embedding.model, **cfg.embedding.kwargs)
    elif cfg.embedding.provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=cfg.embedding.model, **cfg.embedding.kwargs)
    else:
        raise ValueError("Unknown embedding provider")

def build_es_client(cfg: UploadCfg):
    return Elasticsearch(hosts=cfg.elasticsearch.hosts,
                         api_key=cfg.elasticsearch.api_key,
                         basic_auth=(cfg.elasticsearch.username, cfg.elasticsearch.password) \
                                    if cfg.elasticsearch.username else None,
                         cloud_id=cfg.elasticsearch.cloud_id,
                         **cfg.elasticsearch.kwargs)

def build_vectorstore(docs, embeddings, es_conn: Elasticsearch, cfg: UploadCfg):
    return ElasticsearchStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=cfg.index_name,
        es_connection=es_conn,
        vector_query_field=cfg.vector_field,
        query_field="text",
        distance_strategy=cfg.distance or "COSINE",
        # strategy=ElasticsearchStore.ApproxRetrievalStrategy(),  # default
        bulk_kwargs={"chunk_size": 1000}
    )

def build_retriever(vs, cfg: QueryCfg):
    """Return a LangChain retriever with the requested search strategy."""
    mapping = {
        "similarity": "similarity",
        "mmr": "mmr",
        "script_score": "script_score",
        "keyword": "similarity_score",
    }
    search_type = mapping.get(cfg.search_type, cfg.search_type)
    kwargs = {"k": cfg.k, **cfg.retriever_kwargs}
    if cfg.score_threshold is not None:
        kwargs["score_threshold"] = cfg.score_threshold
    return vs.as_retriever(search_type=search_type, search_kwargs=kwargs)