import json
import os

def save_chunks_to_json(chunks, output_path):
    """
    Lưu danh sách các chunk (list of dict) ra file JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tạo folder nếu chưa có

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(chunks)} chunks vào: {output_path}")