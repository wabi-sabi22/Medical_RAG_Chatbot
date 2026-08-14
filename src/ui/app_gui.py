

import gradio as gr
import requests
import os

# Lấy từ biến môi trường, mặc định trỏ về service "backend" trong Docker compose
backend_url = os.getenv("BACKEND_URL", "http://backend:8000/api/v1/chat")

def ask_medical_bot(message, history):
    try:
        response = requests.post(backend_url, json={"question": message})
        if response.status_code == 200:
            data = response.json()
            return data["answer"]
        else:
            return f" Lỗi: Backend phản hồi không thành công ({response.status_code})."
    except Exception as e:
        return f" Không thể kết nối đến Backend API tại {backend_url}. (Chi tiết: {e})"

custom_theme = gr.themes.Soft().set(
    background_fill_primary="#e6f0fa",
    background_fill_secondary="#d0e4f7",
    body_background_fill="#e6f0fa"
)

# ĐOẠN MỚI: Thêm CSS ép trình duyệt cho phép bôi đen (highlight) văn bản
custom_css = """
.message, .message p, .chatbot {
    user-select: text !important;
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
}
"""

demo = gr.ChatInterface(
    fn=ask_medical_bot,
    title="🏥 Trợ Lý Số Y Tế",
    description="Trích xuất thông tin từ file PDF. " \
    "Lưu ý: Chatbot chỉ hỗ trợ cung cấp thông tin tham khảo từ tài liệu gốc , không có chuyên môn chuyên sâu để thay thế các bác sĩ và chuyên viên y tế. ",
    chatbot=gr.Chatbot(show_copy_button=True),
    theme=custom_theme,
    css=custom_css  # Chèn CSS vào giao diện
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)