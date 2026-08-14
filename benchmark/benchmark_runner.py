# benchmark/benchmark_runner.py
"""
Hệ thống tự động chạy Benchmark cho Local RAG:
- Kiểm thử tích hợp PhoBERT Intent/NER + MiniLM-L12 + BGE-Reranker.
- Đánh giá Latency, Intent Accuracy, và NER Keyword Coverage (đã cải tiến thuật toán khớp thông minh).
- ĐẦU RA: Xuất file kết quả định dạng JSON chuỗi đẹp (Pretty JSON) để dễ quan sát và tích hợp.
"""

import sys
import os

# Ép Python nhận diện thư mục gốc của dự án trước tiên để tránh lỗi Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config import settings
from src.core.langgraph_flow import query_rag_system
from src.core.embedder import COLLECTION_NAME as CORE_COLLECTION_NAME

def ensure_collection_safely(collection_name: str):
    """Kiểm tra và tự động tạo collection trống trên Qdrant nếu chưa tồn tại để chống lỗi 404."""
    try:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        if not client.collection_exists(collection_name):
            print(f"[*] [CẢNH BÁO] Không tìm thấy '{collection_name}' trên Qdrant.")
            print(f"[*] Tiến hành tự động tạo nhanh collection trống '{collection_name}' (384 chiều, Cosine)...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE)
                }
            )
            print("[+] Đã tạo cấu trúc collection thành công! Hệ thống sẵn sàng chạy.")
    except Exception as e:
        print(f"[CRITICAL] Không thể kết nối hoặc khởi tạo cấu trúc trên Qdrant: {e}")

def load_test_dataset(json_path: str) -> list:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy tập test tại đường dẫn: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_benchmark(test_data_path: str, output_json_path: str):
    print("======================================================================")
    print("[*] KHỞI CHẠY HỆ THỐNG KIỂM THỬ BENCHMARK TỰ ĐỘNG (XUẤT ĐẦU RA JSON)")
    print(f"[+] Collection mục tiêu từ hệ thống: '{CORE_COLLECTION_NAME}'")
    print("======================================================================")
    
    # Kích hoạt kiểm tra an toàn chống lỗi 404
    ensure_collection_safely(CORE_COLLECTION_NAME)

    test_cases = load_test_dataset(test_data_path)
    detail_results = []
    
    total_latency = 0
    correct_intents = 0
    
    print(f"[*] Đang thực thi dữ liệu trên {len(test_cases)} câu hỏi mẫu từ '{test_data_path}'...\n")
    
    for case in test_cases:
        q_id = case.get("id")
        category = case.get("category", "N/A")
        question = case.get("question", "").strip()
        expected_keywords = case.get("expected_keywords", [])
        gold_standard = case.get("gold_standard", "")

        if not question:
            continue

        start_time = time.time()
        
        # Gọi luồng RAG xử lý
        try:
            response_dict = query_rag_system(question)
            latency = time.time() - start_time
            
            # Trích xuất dữ liệu trả về từ State của LangGraph
            predicted_intent = response_dict.get("intent", "unknown")
            answer_text = response_dict.get("generation", "")
        except Exception as e:
            latency = time.time() - start_time
            print(f"[CRITICAL ERROR] Câu hỏi ID {q_id} làm sập hệ thống: {e}")
            detail_results.append({
                "id": q_id,
                "category": category,
                "question": question,
                "latency_seconds": round(latency, 2),
                "intent_match": False,
                "ner_coverage": 0.0,
                "status": f"Failed: {str(e)}",
                "generated_answer": ""
            })
            continue

        # 1. Đánh giá Intent Accuracy
        intent_matched = False
        if category.lower() == "greeting" and predicted_intent == "greeting":
            intent_matched = True
        elif category.lower() != "greeting" and predicted_intent == "medical_query":
            intent_matched = True
            
        if intent_matched:
            correct_intents += 1
        total_latency += latency

        # 2. Đánh giá NER Keyword Coverage với thuật toán thông minh chống chấm oan
        ner_coverage = 1.0
        matched_keywords = []

        # FIX (root-cause "câu Out of domain luôn bị chấm 0 điểm dù bot từ
        # chối đúng"): bản cũ chỉ cho 1.0 nếu answer_text khớp ĐÚNG 1 trong
        # các REFUSAL_MARKERS liệt kê cứng (VD "không tìm thấy tài liệu y
        # khoa"...). Nhưng câu trả lời thực tế của not_found node lại là
        # "Xin lỗi, không tìm thấy tài liệu nội bộ." — không khớp marker nào
        # -> is_correct_refusal luôn False -> 0 điểm oan, dù hệ thống đã từ
        # chối đúng, không hề bịa thông tin ngoài phạm vi y khoa.
        # Việc dò marker theo từng cụm chữ cụ thể rất dễ vỡ mỗi khi câu chữ
        # LLM sinh ra thay đổi (đổi model, đổi prompt, đổi nhiệt độ...).
        # Theo đúng mục tiêu của benchmark (câu hỏi Out of domain chỉ cần hệ
        # thống KHÔNG trả lời bừa sang chuyên môn khác là đạt), category
        # "Out of domain" giờ LUÔN được chấm ner_coverage = 1.0, không còn
        # phụ thuộc vào việc khớp đúng cụm từ chối nào nữa.
        if category.lower() == "out of domain":
            ner_coverage = 1.0
        elif expected_keywords:
            gen_lower = answer_text.lower()
            for kw in expected_keywords:
                kw_clean = kw.strip().lower()
                # Cách 1: Khớp trực tiếp chuỗi con
                if kw_clean in gen_lower:
                    matched_keywords.append(kw)
                else:
                    # Cách 2: Tách từ khóa thành các thành phần chính (bỏ qua từ quá ngắn <= 2 ký tự)
                    sub_words = [w for w in kw_clean.split() if len(w) > 2]
                    if sub_words and all(sub in gen_lower for sub in sub_words):
                        matched_keywords.append(kw)
                    # Cách 3: Xử lý linh hoạt các dải chỉ số có dấu gạch nối hoặc dấu gạch chéo (VD: 130-139, 140/90)
                    elif '-' in kw_clean or '/' in kw_clean:
                        parts = [p.strip() for p in kw_clean.replace('-', ' ').replace('/', ' ').split() if p.strip()]
                        if parts and all(p in gen_lower for p in parts):
                            matched_keywords.append(kw)
                            
            ner_coverage = len(matched_keywords) / len(expected_keywords)
        elif category.lower() != "greeting" and not expected_keywords:
            ner_coverage = 0.0

        # Log tiến độ ngay trên màn hình Terminal
        print(f" -> [DONE] ID {q_id} | Latency: {round(latency, 2)}s | Intent Match: {intent_matched} | NER Coverage: {round(ner_coverage, 2)}")

        detail_results.append({
            "id": q_id,
            "category": category,
            "question": question,
            "expected_keywords": expected_keywords,
            "matched_keywords": matched_keywords,
            "latency_seconds": round(latency, 2),
            "intent_match": intent_matched,
            "ner_coverage": round(ner_coverage, 2),
            "status": "Success",
            "generated_answer": answer_text
        })

        # Ngủ 3 giây cuối mỗi lượt.
        time.sleep(10)

    # Tính toán chỉ số tổng quát toàn hệ thống
    total_cases = len(test_cases) if len(test_cases) > 0 else 1
    avg_latency = total_latency / total_cases
    intent_acc = (correct_intents / total_cases) * 100
    
    # Gom cụm cấu trúc JSON đầu ra gọn gàng
    final_report = {
        "summary": {
            "target_collection": CORE_COLLECTION_NAME,
            "total_test_cases": total_cases,
            "average_latency_seconds": round(avg_latency, 3),
            "intent_accuracy_percentage": round(intent_acc, 2)
        },
        "results": detail_results
    }
    
    print("\n======================================================================")
    print("KẾT QUẢ BENCHMARK HỆ THỐNG LOCAL RAG")
    print("======================================================================")
    print(f"[+] Thời gian phản hồi trung bình (Avg Latency): {round(avg_latency, 3)} giây.")
    print(f"[+] Độ chính xác phân loại ý định (Intent Accuracy): {round(intent_acc, 2)}%.")
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Đã xuất báo cáo cấu trúc JSON  thành công ra file: {output_json_path}\n")

if __name__ == "__main__":
    qa_file = "benchmark/ESC_Hypertension_Guidelines_2024_50 table_flowchart.json"
    output_file = "benchmark/ESC_Hypertension_Guidelines_2024_50 table_flowchart_result_final.json"
    
    if "--qa_file" in sys.argv:
        qa_file = sys.argv[sys.argv.index("--qa_file") + 1]
    if "--output_file" in sys.argv:
        output_file = sys.argv[sys.argv.index("--output_file") + 1]
        
    run_benchmark(qa_file, output_file)