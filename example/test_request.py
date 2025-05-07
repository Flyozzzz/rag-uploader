import requests, json, pathlib, sys
F = pathlib.Path(__file__).with_suffix(".pdf").name   # sample.pdf в той же папке

cfg = {
    "index_name": "demo-index",
    "loader": {"mode": "single"},
    "splitter": {"type": "RecursiveCharacterTextSplitter",
                 "chunk_size": 1024, "chunk_overlap": 128},
    "embedding": {"provider": "huggingface",
                  "model": "intfloat/multilingual-e5-large-instruct"},
    "elasticsearch": {
        "hosts": ["http://localhost:9200"]
    },
    "metadata": {"project": "demo"}
}

resp = requests.post(
    "http://localhost:8001/upload",
    files=[("files", (F, open(F, "rb"), "application/pdf"))],
    data={"config": json.dumps(cfg)}
)
print(resp.status_code, resp.json())
