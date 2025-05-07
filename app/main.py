from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import json, shutil, tempfile, pathlib, uuid, os
from app.config import UploadCfg
from app.factory import build_loader, build_splitter, build_embeddings, build_es_client, build_vectorstore
import uvicorn
app = FastAPI(title="Generic RAG Uploader API")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...),
                 config: str = Form(...)):
    try:
        cfg = UploadCfg(**json.loads(config))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad config: {e}"})

    es_client   = build_es_client(cfg)
    embeddings  = build_embeddings(cfg)
    splitter    = build_splitter(cfg)

    total_chunks, total_docs = 0, 0
    for upf in files:
        # сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(upf.filename).suffix) as tmp:
            shutil.copyfileobj(upf.file, tmp)
            tmp_path = tmp.name

        loader = build_loader(tmp_path, cfg)
        docs   = loader.load()
        total_docs += len(docs)

        # добавляем пользовательские метаданные
        for d in docs:
            d.metadata.update(cfg.metadata | {"source_file": upf.filename})

        chunks = splitter.split_documents(docs)
        total_chunks += len(chunks)

        # пишем в ES
        vs = build_vectorstore(chunks, embeddings, es_client, cfg)
        if cfg.upsert:
            vs.add_documents(chunks)
        else:
            vs._bulk_insert(chunks)   # низкоуровневый insert без перезаписи

        # чистим tmp
        os.unlink(tmp_path)

    return {
        "index": cfg.index_name,
        "uploaded_files": len(files),
        "docs": total_docs,
        "chunks": total_chunks
    }
