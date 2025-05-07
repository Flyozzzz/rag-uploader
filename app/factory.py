from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, \
                                     CharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from app.config import UploadCfg
import tempfile, pathlib

def build_loader(path: str, cfg: UploadCfg):
    return UnstructuredFileLoader(path, mode=cfg.loader.mode, **cfg.loader.kwargs)

def build_splitter(cfg: UploadCfg):
    cls_map = {
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "CharacterTextSplitter": CharacterTextSplitter,
        "HTMLHeaderTextSplitter": HTMLHeaderTextSplitter
    }
    cls = cls_map.get(cfg.splitter.type, RecursiveCharacterTextSplitter)
    return cls(chunk_size=cfg.splitter.chunk_size,
               chunk_overlap=cfg.splitter.chunk_overlap,
               **cfg.splitter.kwargs)

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
