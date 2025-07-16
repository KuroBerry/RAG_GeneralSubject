import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import retriever.cache_data as cache_data
from retriever import cache_data

def get_router(input_query, model_choice):
    prompt = f"""
    Bạn là trợ lý phân loại câu hỏi học thuật. Hãy xác định câu hỏi sau thuộc môn nào.
    
    Trước khi vô thực hiện phân loại về môn học, hãy xử lý logic của câu hỏi trước. Xử lý theo yêu cầu sau:
    ** Xử lý logic câu hỏi: **
    * **Nếu câu hỏi không có logic thì trả về "unlogic", không cần thực hiện các yêu cầu phân loại bên dưới**
    * **Nếu câu hỏi có logic thì tiếp tục xử lý* (để phân loại môn học)
    
    **Hướng dẫn phân loại:**
    * **Lịch Sử Đảng**: Nếu câu hỏi liên quan đến lịch sử hình thành, các sự kiện quan trọng, các giai đoạn phát triển, các lãnh đạo, chủ trương, đường lối của Đảng Cộng sản Việt Nam.
        * **Ví dụ**: "Chiến dịch Điện Biên Phủ diễn ra vào thời gian nào?", "Nghị quyết Trung ương 4 khóa XII nói về vấn đề gì?"
    * **Triết Học Mác-Lênin**: Nếu câu hỏi liên quan đến các khái niệm, nguyên lý, quy luật của chủ nghĩa duy vật biện chứng, chủ nghĩa duy vật lịch sử, học thuyết giá trị thặng dư, và tư tưởng của Karl Marx, Friedrich Engels, Vladimir Lenin.
        * **Ví dụ**: "Mâu thuẫn là gì trong triết học biện chứng?", "Học thuyết hình thái kinh tế xã hội được hiểu như thế nào?"
    * **Normal**: Nếu câu hỏi thuộc những câu hỏi chào hỏi như "Hello", "Chào bạn",... đoạn chat thể hiện cảm xúc hay còn gọi là normal chatting, thì phân loại là "normal".

    **Câu hỏi cần phân loại:** "{input_query}"

    **Chỉ trả lời bằng một trong năm từ sau:**
    * `triet-hoc`
    * `lich-su-dang`
    * `normal` (Nếu câu hỏi thuộc những câu hỏi chào hỏi, đoạn chat thể hiện cảm xúc hay còn gọi là normal chatting)
    * `unlogic` (Nếu câu hỏi không có logic)
    * `unknown` (Nếu không xác định được câu hỏi thuộc môn nào trong 3 phạm trù trên)
    """
    GEMINI_API_KEY = cache_data.get_gemini_key()
    print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")
    print(f"Type: {type(GEMINI_API_KEY)}")
    # client = genai.Client(api_key=GEMINI_API_KEY)
    client = cache_data.get_gemini_model()
    response = client.models.generate_content(
        model=f"{model_choice}",
        contents=prompt,
    )

    return response.text

def generate_answer(input_query, context, router, chat_history=None, model_choice=None):
    if router == 'triet-hoc' or router == 'lich-su-dang':
        # prompt = f"""
        # Mày là một chuyên gia trong việc trả lời các môn học đại cương về chính trị bậc đại học không chính quy.
        # Đây là lịch sử chat dùng để hỗ trợ các câu hỏi tiếp theo của người dùng:
        # {chat_history}
        
        # Dưới đây chính là query của người dùng về các câu hỏi liên quan tới Triết học. 
        # {input_query}
        
        # Còn dưới đây là những context được cung cấp đên mày để trả lời câu hỏi của người dùng.
        # {context}
        
        # Yêu cầu khi trả lời:
        # - Tóm tắt và tổng hợp thông tin từ các đoạn context có liên quan.
        # - Hạn chế sao chép nguyên văn toàn bộ một đoạn nào từ context.
        # - Văn phong rõ ràng, súc tích, mang tính học thuật.
        # - Hãy ghi nguồn gốc của thông tin trong câu trả lời bằng id của đoạn văn bản trong context, đặt id ở cuối thông tin đó.
        # - Dựa vào nhưng thông tin trên, mày hãy thực hiện trả lời câu hỏi của người dùng. Chỉ trả lời theo nội dung context cung cấp. Nếu những nội dung đó không liên quan đến câu hỏi thì trả lời "Context không liên quan".

        # ## Câu trả lời cần được trình bày như sau:
            # **Nội dung:**  
        #     Chính sách nhân nhượng của quân ta với quân tưởng:\n
        #     \ta) Giải thích rõ.\n
        #     \tb) Áp dụng biện pháp ngăn chặn.\n 
        #     \tc) Sử dụng vũ lực khi cần thiết.\n
        #     Tóm lại: ...
        # Câu trên chỉ là ví dụ. Nhưng hãy theo format đó để trình bày trả lời.       
        # """
        
        prompt = f"""
            Mày là một chuyên gia trong việc trả lời các môn học đại cương về chính trị bậc đại học không chính quy.
            Đây là lịch sử chat dùng để hỗ trợ các câu hỏi tiếp theo của người dùng:
            {chat_history}

            Dưới đây chính là query của người dùng về các câu hỏi liên quan tới Triết học:
            {input_query}

            Còn dưới đây là những context được cung cấp đến mày để trả lời câu hỏi của người dùng:
            {context}

            Yêu cầu khi trả lời:
            - Tóm tắt và tổng hợp thông tin từ các đoạn context có liên quan.
            - Hạn chế sao chép nguyên văn toàn bộ một đoạn nào từ context.
            - Văn phong rõ ràng, súc tích, mang tính học thuật.
            - Ghi rõ nguồn gốc của thông tin bằng ID của đoạn văn bản trong context, đặt ID trong ngoặc vuông ở cuối câu (ví dụ: [doc_3]).
            - Chỉ trả lời theo nội dung context cung cấp. Nếu những nội dung đó không liên quan đến câu hỏi thì trả lời "Context không liên quan".
            - Trả lời phải tuân thủ **định dạng Markdown** như bên dưới (có xuống dòng, đánh số mục rõ ràng):

            ## Câu trả lời cần được trình bày như sau (tuân thủ định dạng):

            **Nội dung:**  
            Chính sách nhân nhượng của quân ta với quân Tưởng:

            \ta) Giải thích rõ:  
            \t\t- Nêu lý do chính trị, hoàn cảnh lịch sử, mục tiêu ban đầu.  
            \t\t- Trích nguồn có liên quan [doc_2].

            \tb) Áp dụng biện pháp ngăn chặn:  
            \t\t- Những hành động thực tế đã thực hiện, các phương án đề ra.  
            \t\t- Trích nguồn nếu có [doc_4].

            \tc) Sử dụng vũ lực khi cần thiết:  
            \t\t- Nêu điều kiện cụ thể để chuyển sang biện pháp vũ lực.  
            \t\t- Trích dẫn theo ngữ cảnh [doc_1].

            **Tóm lại:**  
            - Tổng hợp ý chính toàn bộ nội dung trên, nhấn mạnh bài học/rút ra.

            Đây là một ví dụ về output:
            
            Nội dung: Triết học là một lĩnh vực nghiên cứu sâu rộng, được hiểu khái quát như sau:
            - Khái niệm và bản chất:
                + Triết học là hệ thống tri thức lý luận chung nhất của con người về thế giới, cũng như về vị trí và vai trò của con người trong thế giới đó [TrietHoc_chunk_00000].
                + Từ nguyên của thuật ngữ "triết học" ở phương Đông (Trung Quốc, Ấn Độ) mang ý nghĩa truy tìm bản chất, trí tuệ sâu sắc, hoặc là con đường suy ngẫm để dẫn đến lẽ phải. Ở phương Tây (Hy Lạp), thuật ngữ "philosophia" có nghĩa là yêu mến sự thông thái, nhấn mạnh khát vọng tìm kiếm chân lý [TrietHoc_chunk_00000].
                + Ngay từ đầu, triết học đã là hoạt động tinh thần biểu hiện khả năng nhận thức, đánh giá của con người, tồn tại như một hình thái ý thức xã hội [TrietHoc_chunk_00000].

            - Đối tượng và phạm vi nghiên cứu:
                + Triết học nghiên cứu thế giới với tư cách là một chỉnh thể, tìm ra những quy luật chung nhất chi phối sự vận động của chỉnh thể đó (bao gồm xã hội loài người và con người trong cuộc sống cộng đồng), và thể hiện chúng một cách có hệ thống dưới dạng duy lý [TrietHoc_chunk_00000].
                + Nó xem xét thế giới như một chỉnh thể và cố gắng đưa ra một hệ thống các quan niệm về chỉnh thể đó thông qua việc tổng kết toàn bộ lịch sử khoa học và lịch sử tư tưởng triết học [TrietHoc_chunk_00004].
                + Triết học là sự diễn tả thế giới quan bằng lý luận. Dù có nhiều quan điểm khác nhau, nhưng điểm chung là triết học nghiên cứu những vấn đề chung nhất của giới tự nhiên, xã hội và con người, cũng như mối quan hệ giữa con người (đặc biệt là tư duy con người) với thế giới xung quanh [TrietHoc_chunk_00004].

            Tóm lại: Triết học là một hệ thống tri thức lý luận toàn diện, nghiên cứu những quy luật chung nhất của thế giới tự nhiên, xã hội và tư duy. Nó giúp con người nhận thức sâu sắc về bản thân và vị trí của mình trong vũ trụ, đồng thời là sự diễn tả thế giới quan dưới dạng lý luận học thuật.
            Lưu ý: Giữ đúng định dạng xuống dòng, thụt đầu dòng và trình bày như ví dụ trên. Không viết liền mạch trong một đoạn duy nhất.
            """
    
    if router == "normal":
        prompt = """
        Mày là một con chatbot chuyên trả lời các câu hỏi normal chatting, các câu hỏi hay câu nói thể hiện cảm xúc.
        
        Đây là lịch sử chat dùng để hỗ trợ các câu hỏi tiếp theo của người dùng:
        {chat_history}
        
        Đây là câu hỏi của người dùng liên quan đến các câu hỏi normal chatting.
        {input_query}
        
        Yêu cầu khi trả lời:
        - Đối với câu chào hỏi thì hãy đáp lại người dùng một cách thân thiện.
        - Nếu người dùng yêu cầu chatbot giới thiệu về bản thân thì hãy giới thiệu bản thân là một chatbot hỗ trợ môn học đại cương về chính trị bậc đại học không chính quy.
        - Đối với các câu cảm thán thì hãy đáp lại một cách thân thiện, gần gũi.

        Kết quả trả lời:
        Ví dụ:
        - Người dùng: "Chào bạn"
        - Chatbot: "Chào bạn, tôi là một chatbot hỗ trợ môn học đại cương về chính trị bậc đại học không chính quy. Tôi có thể giúp gì cho bạn hôm nay?"
        
        - Người dùng: "Hôm nay trời đẹp quá"
        - Chatbot: "Thật tuyệt vời! Thời tiết đẹp luôn mang lại cảm giác thoải mái. Bạn có kế hoạch gì cho ngày hôm nay không?"
        """
    # prompt = f"""
    # Mày là một chuyên gia trong việc trả lời các môn học đại cương về chính trị bậc đại học không chính quy.
    # Đây là lịch sử chat dùng để hỗ trợ các câu hỏi tiếp theo của người dùng:
    # {chat_history}
    
    # Dưới đây chính là query của người dùng về các câu hỏi liên quan tới các môn chính trị. 
    # {input_query}
    
    # Còn dưới đây là những context được cung cấp đên mày để trả lời câu hỏi của người dùng.
    # {context}
    
    # Yêu cầu khi trả lời:
    # - Tóm tắt và tổng hợp thông tin từ các đoạn context có liên quan.
    # - Hạn chế sao chép nguyên văn toàn bộ một đoạn nào từ context.
    # - Văn phong rõ ràng, súc tích, mang tính học thuật.
    # - Hãy ghi nguồn gốc của thông tin trong câu trả lời bằng id của đoạn văn bản trong context, đặt id ở cuối thông tin đó.
    # - Dựa vào nhưng thông tin trên, mày hãy thực hiện trả lời câu hỏi của người dùng. Chỉ trả lời theo nội dung context cung cấp. Nếu những nội dung đó không liên quan đến câu hỏi thì trả lời "Context không liên quan".
    # """
    
    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    client = cache_data.get_gemini_model()
    response = client.models.generate_content(
        model=f"{model_choice}",
        contents=prompt,
    )

    return response.text

def main():
    input_query = "ĐCSVN thành lập năm nào?"
    namespace = get_router(input_query, "gemini-2.5-flash")
    print(f"Câu hỏi: {input_query}")
    print(f"Namespace phân loại: {namespace}")
    
if __name__ == "__main__":
    main()