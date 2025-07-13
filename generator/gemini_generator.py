import warnings
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google import genai
from dotenv import load_dotenv
from retriever.hybrid_search import hybrid_retriever

#==========================================================
# Lấy API key từ file .env
load_dotenv("../.env")
GEMINI_API_KEY = os.getenv("GEMINI_API")

# Gọi LLM kết hợp câu hỏi và context để trả lời
def generate_answer(input_querry):
    context = hybrid_retriever(input_querry)

    prompt = f"""
    Bạn là một Generator chuyên nghiệp, được giao nhiệm vụ trả lời câu hỏi dựa trên các đoạn văn bản sau (context). 

    QUAN TRỌNG: 
    - **Chỉ được sử dụng thông tin có trong context.** 
    - **Không được đưa ra bất kỳ suy đoán, bổ sung, hay kiến thức bên ngoài nào.** 
    - Nếu không tìm thấy câu trả lời trong context, hãy trả lời: **"Thông tin không có trong đoạn văn được cung cấp."**

    Yêu cầu khi trả lời:
    - Tóm tắt và tổng hợp thông tin từ các đoạn context có liên quan.
    - Hạn chế sao chép nguyên văn toàn bộ một đoạn nào từ context.
    - Văn phong rõ ràng, súc tích, mang tính học thuật.
    - Hãy ghi nguồn gốc của thông tin trong câu trả lời bằng id của đoạn văn bản trong context, đặt id ở cuối thông tin đó.

    ---

    Câu hỏi: {input_querry}

    Context:
    {context}
    """

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

#==========================================================

def main():
    print("=== Hệ thống hỏi đáp chạy liên tục ===")
    print("Nhập '0' để thoát.")
    while True:
        input_querry = input("\nNhập câu hỏi: ")
        if input_querry.strip() == '0':
            print("Đã thoát.")
            break
        print("\nĐang tạo câu trả lời...")
        answer = generate_answer(input_querry)
        print("\nCâu trả lời:")
        print("=" * 50)
        print(answer)
        print("=" * 80)

if __name__ == "__main__":
    main()
