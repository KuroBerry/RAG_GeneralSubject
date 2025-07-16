import time
import random
import threading
import streamlit as st
from agent import get_router
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from retriever.parent_retrieval import parent_document_search as parent_retrieval
from agent import generate_answer

# Cache responses cho normal chatting
NORMAL_RESPONSES = [
    "Xin chào! Tôi là trợ lý AI của bạn. Tôi có thể giúp bạn trả lời các câu hỏi về Triết học Mác-Lênin và Lịch sử Đảng. Bạn có câu hỏi gì không?",
    "Chào bạn! Rất vui được trò chuyện với bạn. Tôi có thể hỗ trợ bạn về các vấn đề học thuật liên quan đến Triết học và Lịch sử Đảng.",
    "Hi! Tôi ở đây để giúp bạn. Hãy đặt câu hỏi về Triết học Mác-Lênin hoặc Lịch sử Đảng nhé!",
    "Chào bạn! Tôi luôn sẵn sàng hỗ trợ bạn. Có điều gì tôi có thể giúp về Triết học và Lịch sử Đảng không?",
    "Xin chào! Tôi là trợ lý chuyên về Triết học và Lịch sử Đảng. Bạn cần tôi giúp gì hôm nay?",
]

UNKNOWN_RESPONSES = [
    "Đây có vẻ không phải là câu hỏi thuộc phạm vi kiến thức của tôi. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
    "Xin lỗi, tôi không có kiến thức về nội dung bạn muốn tìm. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
    "Kiến thức này không nằm trong phạm vi của tôi. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
]

UNLOGIC_RESPONSES = [
    "Câu hỏi của bạn có vấn đề về ngữ nghĩa. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
    "Xin lỗi, có vẻ như câu hỏi của bạn đang có vấn đề. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
    "Câu này không có ngữ nghĩa. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
]

def reset_conversation():
    """
    Đặt lại toàn bộ trạng thái của cuộc trò chuyện.
    """
    st.session_state['messages'] = [
        {"role": "assistant", "content": random.choice(
            [
                "Xin chào! Tôi là trợ lý môn của bạn. Tôi sẽ giúp bạn trả lời các câu hỏi về Triết học và Lịch sử Đảng. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
                "Chào bạn! Bạn cần tôi giúp gì không? Tôi có thể trả lời mọi câu hỏi của bạn về Triết học và Lịch sử Đảng.",
                "Bạn muốn biết gì về Triết học và Lịch sử Đảng? Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
        ])}
    ] 
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    msgs.clear()  
    st.rerun()

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def clean_response(response):
    """
    Làm sạch response để hiển thị tốt hơn
    """
    if not isinstance(response, str):
        response = str(response)
    
    # Loại bỏ khoảng trắng thừa
    response = response.strip()
    
    # Thay thế escape characters
    response = response.replace('\\n', '\n')
    response = response.replace('\\t', '\t')
    
    return response
    
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="Legal Assistant", 
        page_icon="💬",
        layout="wide" 
    )

def setup_sidebar():
    with st.sidebar:
        st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>⚙️ Cấu Hình</h1>
        """, unsafe_allow_html=True)
        
        with st.expander("🚀 Chọn Model AI", expanded=True):
            model_choice = st.selectbox(
                "Chọn Model để trả lời:",
                ["gemini-2.5-pro", "gemini-2.5-flash"],
                index=0
            )
            st.caption("🔹 Model AI sẽ ảnh hưởng đến tốc độ và độ chính xác của câu trả lời.")
          
        with st.expander("🔎 Tính Năng Trích Xuất Văn Bản"):
            retrieval_choice = st.selectbox(
                "Chọn phương thức truy vấn:",
                ["Hybrid retrieval", "Parent documents retrieval"],
                index=0
            )
            st.caption("Tính năng này sẽ quyết định cách dữ liệu được truy xuất.")
            
            num_retrieval_docs = st.slider(
                "🔢 Số lượng văn bản truy xuất:",
                min_value=1,  
                max_value=10,  
                value=5 
            )
  
            
        with st.expander("🛠 Tuỳ Chọn Khác"):
            if st.button("🗑 Xóa cuộc trò chuyện", use_container_width=True):
                reset_conversation()
            st.markdown("""
            <small style='color: grey;'>Xóa lịch sử chat để bắt đầu cuộc trò chuyện mới.</small>
            """, unsafe_allow_html=True)
        
    return model_choice, retrieval_choice, num_retrieval_docs
        
def setup_chat_interface(model_choice):
    st.title("Trợ Lý Học Thuật 💬")
    if model_choice == "gemini-2.5-pro" or model_choice == "gemini-2.5-flash":
        st.caption("Trợ lý AI được hỗ trợ bởi Google")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if "messages" not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": random.choice(
                [
                "Xin chào! Tôi là trợ lý môn của bạn. Tôi sẽ giúp bạn trả lời các câu hỏi về Triết học và Lịch sử Đảng. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
                "Chào bạn! Bạn cần tôi giúp gì không? Tôi có thể trả lời mọi câu hỏi của bạn về Triết học và Lịch sử Đảng.",
                "Bạn muốn biết gì về Triết học và Lịch sử Đảng? Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
            ])}
        ]
        msgs.add_ai_message(st.session_state.messages[0]["content"])
    
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    return msgs

def user_input(msgs, model_choice, retrieval_choice):
    if input_query:= st.chat_input("Hãy hỏi tôi bất cứ điều gì về Triết học và Lịch Sử Đảng!"):
        st.session_state['messages'].append({"role": "user", "content": input_query})
        with st.chat_message("user"):
            st.markdown(input_query)   
        msgs.add_user_message(input_query)
        
        # Hiển thị ngay lập tức rằng AI đang xử lý
        with st.chat_message("assistant"):
            # Hiển thị spinner ngay lập tức
            with st.spinner("Đang suy nghĩ..."):
                response_placeholder = st.empty()
                timer_placeholder = st.empty()
                start_time = time.time()
                
                # Hiển thị thông báo ngay lập tức
                response_placeholder.markdown("🤔 Đang phân tích câu hỏi...")
                
                # Khởi tạo timer ngay sau khi bắt đầu
                def update_timer():
                    while True:
                        elapsed_time = time.time() - start_time
                        timer_placeholder.caption(f"⏱️ Đang xử lý: {elapsed_time:.2f} giây")
                        time.sleep(0.1)

                timer_thread = threading.Thread(target=update_timer, daemon=True)
                timer_thread.start()
                
                # Cache router classification để tránh gọi AI nhiều lần
                if "router_cache" not in st.session_state:
                    st.session_state["router_cache"] = {}
                
                # Check cache trước
                if input_query in st.session_state["router_cache"]:
                    router = st.session_state["router_cache"][input_query]
                    response_placeholder.markdown("🔄 Đang sử dụng cache...")
                else:
                    # Phân loại router bên trong spinner
                    router = get_router(input_query, model_choice)
                    router = router.strip(" ")
                    # Lưu vào cache
                    st.session_state["router_cache"][input_query] = router
                
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                if router == "triet-hoc":
                    response_placeholder.markdown("🔍 Đang tìm kiếm thông tin về Triết học...")
                    st.caption("This is triet-hoc")
                    if retrieval_choice == "Parent documents retrieval":
                        st.caption("This is parent retrieval")
                        response_placeholder.markdown("📚 Đang truy xuất tài liệu...")
                        context = parent_retrieval(input_query, namespace="triet-hoc-children", p_namespace="triet-hoc")

                    elif retrieval_choice == "Hybrid retrieval":
                        st.caption("This is hybrid retrieval")
                        response_placeholder.markdown("🔎 Đang tìm kiếm hybrid...")
                        context = "No context"

                    response_placeholder.markdown("✍️ Đang tạo câu trả lời...")
                    response = generate_answer(input_query=input_query, context=context, router=router, chat_history=chat_history, model_choice=model_choice)
                        
                elif router == "lich-su-dang":
                    response_placeholder.markdown("🔍 Đang tìm kiếm thông tin về Lịch sử Đảng...")
                    st.caption("This is lich-su-dang")
                    if retrieval_choice == "Parent documents retrieval":
                        st.caption("This is parent retrieval")
                        response_placeholder.markdown("📚 Đang truy xuất tài liệu...")
                        context = parent_retrieval(input_query, namespace="lich-su-dang-children", p_namespace="lich-su-dang")

                    elif retrieval_choice == "Hybrid retrieval":
                        st.caption("This is hybrid retrieval")
                        response_placeholder.markdown("🔎 Đang tìm kiếm hybrid...")
                        context = "No context"

                    response_placeholder.markdown("✍️ Đang tạo câu trả lời...")
                    response = generate_answer(input_query=input_query, context=context, router=router, chat_history=chat_history, model_choice=model_choice)
                    # response = normal_response(transform_prompt, model_choice, chat_history)
                
                elif router == "normal":
                    response_placeholder.markdown("💬 Đang chuẩn bị trả lời...")
                    st.caption("This is normal")
                    # Trả lời nhanh cho normal chatting không cần gọi AI
                    response = random.choice(NORMAL_RESPONSES)
                    # Không cần gọi generate_answer cho normal chatting
                    # response = generate_answer(input_query=input_query, context=None, router=router, chat_history=chat_history, model_choice=model_choice)
                
                elif router == "unknown":
                    response_placeholder.markdown("❓ Đang xử lý câu hỏi...")
                    st.caption("This is Unknown")
                    response = random.choice(UNKNOWN_RESPONSES)
                    
                else:
                    response_placeholder.markdown("⚠️ Đang xử lý câu hỏi...")
                    response = random.choice(UNLOGIC_RESPONSES)  
                
                # Hiển thị response ngay lập tức với format đúng
                if isinstance(response, str):
                    # Làm sạch response trước khi hiển thị
                    clean_resp = clean_response(response)
                    # Hiển thị toàn bộ response ngay lập tức để giữ format
                    response_placeholder.markdown(clean_resp)
                    final_response = clean_resp
                else:
                    # Fallback nếu response không phải string
                    final_response = clean_response(str(response))
                    response_placeholder.markdown(final_response)
                
                # Lưu response vào session state và chat history
                st.session_state['messages'].append({"role": "assistant", "content": final_response})
                msgs.add_ai_message(final_response)
                
                # Hiển thị thời gian phản hồi cuối cùng
                end_time = time.time()
                elapsed_time = end_time - start_time
                timer_placeholder.caption(f"⏱️ Thời gian phản hồi: {elapsed_time:.2f} giây")
def main():
    setup_page()
    model_choice, retrieval_choice, num_retrieval_docs = setup_sidebar()
        
    msgs = setup_chat_interface(model_choice)
    user_input(msgs, model_choice, retrieval_choice)
    
if __name__ == "__main__":
    main()