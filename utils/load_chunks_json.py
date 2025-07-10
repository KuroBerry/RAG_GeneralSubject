import json

def load_chunks_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks
