# Medical RAG Chatbot

## Giới thiệu
Medical RAG Chatbot là một hệ thống hỏi đáp y tế tự động áp dụng phương pháp Retrieval-Augmented Generation (RAG). Dự án kết hợp cơ sở dữ liệu vector (Qdrant) và cơ sở dữ liệu đồ thị (Memgraph) để tra cứu thông tin y tế một cách chính xác. Ngoài ra, dự án sử dụng Apache Airflow để điều phối và quản lý các luồng xử lý dữ liệu.

### Kiến trúc tổng quan
![Kiến trúc tổng quan](medical_rag_architecture_overview.png)

*Lưu ý: Qdrant DB và Memgraph được cấu hình chạy hoàn toàn ở môi trường local bên trong Docker, nên bạn không cần phải cung cấp API Key cho hai dịch vụ này.*

## Cấu trúc thư mục
```
medical_rag_project/
├── benchmark/        # Chứa kịch bản đánh giá hiệu suất mô hình và kết quả báo cáo
├── dags/             # Chứa luồng công việc (DAGs) của Apache Airflow để nạp và xử lý dữ liệu
├── data/             # Lưu trữ tài liệu y khoa đầu vào (PDF/Word) và các tệp kết quả
├── src/              # Thư mục chứa mã nguồn chính của toàn bộ hệ thống
│   ├── api/          # Các endpoint FastAPI xử lý yêu cầu phía backend
│   ├── core/         # Logic cốt lõi (LangGraph workflow, Embedding, Qdrant/Memgraph, LlamaParse)
│   ├── ui/           # Mã nguồn giao diện người dùng (Gradio UI)
│   ├── config.py     # Nạp và quản lý các biến môi trường cấu hình hệ thống
│   └── database.py   # Module khởi tạo và quản lý kết nối cơ sở dữ liệu
├── .env.sample       # File mẫu chứa các biến cấu hình dự án để người dùng tham khảo
├── docker-compose.yml# File cấu hình triển khai toàn bộ hệ thống bằng Docker
├── Dockerfile        # File build image Docker cho các dịch vụ
├── requirements.txt  # Danh sách tất cả thư viện Python cần cài đặt
├── run_backend.py    # Script khởi chạy API Backend (chạy trên port 8000)
└── run_frontend.py   # Script khởi chạy giao diện Chatbot UI (chạy trên port 7860)
```

## Hướng dẫn cài đặt và sử dụng (bằng Docker)

Dự án này được đóng gói hoàn toàn bằng Docker, giúp việc triển khai trở nên đồng bộ và dễ dàng.

### 1. Yêu cầu hệ thống
- Đã cài đặt **Docker** và **Docker Compose**.

### 2. Thiết lập biến môi trường (.env)
Dự án yêu cầu một số API Key để hoạt động. Bạn hãy copy file `.env.sample` đổi tên thành `.env` và điền các khóa bảo mật sau:
- **HF_TOKEN (Hugging Face Token):** Đăng nhập vào [Hugging Face](https://huggingface.co/), truy cập mục *Settings > Access Tokens* và tạo một token mới (quyền Read).
- **GROQ_API_KEY:** Tạo khóa API tại [Groq Console](https://console.groq.com/keys).
- **LLAMA_CLOUD_API_KEY:** Tạo khóa API tại [LlamaCloud](https://cloud.llamaindex.ai/).

*(Qdrant DB và Memgraph đã được cấu hình chạy nội bộ trong Docker nên không cần thiết lập API Key)*

### 3. Khởi chạy hệ thống
Mở terminal tại thư mục gốc của dự án (`medical_rag_project`) và chạy lệnh sau để tải, build và khởi động toàn bộ các dịch vụ:

```bash
docker-compose up -d --build
```

Sau khi quá trình khởi động hoàn tất, bạn có thể truy cập các dịch vụ qua các đường dẫn sau:
- **Frontend (Giao diện Chatbot):** [http://localhost:7860](http://localhost:7860)
- **Backend (API):** [http://localhost:8000](http://localhost:8000)
- **Airflow Webserver:** [http://localhost:8080](http://localhost:8080) *(Tài khoản mặc định: admin / admin)*
- **Qdrant (Vector DB):** `localhost:6333`
- **Memgraph-lab:** [http://localhost:3000](http://localhost:3000)

### 4. Theo dõi log và kiểm tra
Để xem log của các dịch vụ đang chạy ngầm, sử dụng lệnh:
```bash
docker-compose logs -f
```

### 5. Dừng hệ thống
Để dừng và tắt toàn bộ các container của hệ thống, chạy lệnh sau:
```bash
docker-compose down
```

---
**Ghi chú quan trọng:** Ngừng hỗ trợ Llama 3.1 8B Instant và sẽ loại bỏ nó vào ngày 16 tháng 8 năm 2026. Sau ngày loại bỏ, các yêu cầu đến mô hình này sẽ không còn được xử lý nữa. 

Khuyến khích bạn chuyển đổi khối lượng công việc của mình sang mô hình thay thế được đề xuất của chúng tôi, GPT OSS 20B.
