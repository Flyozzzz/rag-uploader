from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmbeddingCfg(BaseModel):
    provider: str = Field("openai", examples=["openai", "huggingface"])
    model:   str = Field(...,  examples=["text-embedding-3-small"])
    kwargs:  Dict[str, Any] = Field(default_factory=dict)

class ESAuth(BaseModel):
    hosts:    List[str] = ["http://localhost:9200"]
    username: Optional[str] = None
    password: Optional[str] = None
    api_key:  Optional[str] = None
    cloud_id: Optional[str] = None
    kwargs:   Dict[str, Any] = Field(default_factory=dict)

class SplitterCfg(BaseModel):
    type:          str = Field("RecursiveCharacterTextSplitter")
    chunk_size:    int = 1024
    chunk_overlap: int = 128
    kwargs:        Dict[str, Any] = Field(default_factory=dict)

class LoaderCfg(BaseModel):
    mode: str = Field("single", examples=["single", "elements", "paged"])
    kwargs: Dict[str, Any] = Field(default_factory=dict)

class UploadCfg(BaseModel):
    index_name:      str
    loader:          LoaderCfg = LoaderCfg()
    splitter:        SplitterCfg = SplitterCfg()
    embedding:       EmbeddingCfg
    elasticsearch:   ESAuth
    upsert:          bool = True
    metadata:        Dict[str, Any] = Field(default_factory=dict)
    vector_field:    str = "embedding"
    distance: str | None = "COSINE"
