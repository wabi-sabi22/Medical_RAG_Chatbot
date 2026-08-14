# src/core/embedder.py
"""
Embedder Local (Hybrid Dense + Sparse BM25):
- Model Dense: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 chiều)
- Model Sparse: Qdrant/bm25 (fastembed) — bù cho điểm yếu của dense với tên
  thuốc, viết tắt y khoa, liều lượng (xem FIX BM25 bên dưới).
- Tối ưu hóa RAM và quản lý batch để nạp dữ liệu ổn định trên CPU.
"""

import os
import gc
import uuid
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector,
)
from src.config import settings
from src.core.graph_store import upsert_structured_ref, ensure_indexes

# =========================================================================
# Trước đây collection chỉ có 1 vector "dense" (MiniLM 384d). Dense giỏi bắt
# ngữ nghĩa chung nhưng yếu với term match chính xác — tên thuốc riêng
# (Perindopril, Amlodipine...), viết tắt y khoa (ƯCMC, CTTA, HATT...), số
# liều cụ thể (5mg, 140/90 mmHg) dễ bị chìm nếu chunk chứa đúng từ đó không
# nằm gần top similarity ngữ nghĩa. BM25 bù lại bằng cách match CHÍNH XÁC
# theo term, không cần "hiểu" ngữ nghĩa.
#
# Chọn fastembed "Qdrant/bm25" thay vì tự cài BM25 thủ công (rank_bm25 + IDF
# tính tay) vì model này STATELESS: sparse vector của mỗi câu/chunk tính độc
# lập, không cần thống kê IDF của toàn corpus. Điều này quan trọng vì
# ingest (Airflow container) và query (backend container) là 2 tiến trình
# khác nhau — nếu BM25 cần IDF toàn corpus thì phải đồng bộ file thống kê
# giữa 2 container, dễ lệch. Với Qdrant/bm25, mỗi bên tự encode độc lập vẫn
# ra kết quả nhất quán.
# =========================================================================
SPARSE_MODEL_NAME = "Qdrant/bm25"
SPARSE_VECTOR_NAME = "bm25"

_local_sparse_model = None


def get_sparse_model() -> SparseTextEmbedding:
    global _local_sparse_model
    if _local_sparse_model is None:
        print(f"[*] Đang nạp sparse BM25 model {SPARSE_MODEL_NAME}...")
        _local_sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _local_sparse_model

# Namespace cố định để sinh point ID xác định (deterministic) từ nội dung chunk.
# KHÔNG được đổi giá trị này sau khi đã ingest dữ liệu thật, nếu không toàn bộ
# ID cũ sẽ đổi và gây trùng lặp dữ liệu thay vì ghi đè đúng chỗ.
_QDRANT_ID_NAMESPACE = uuid.UUID("7d6f2f2e-2a3f-4a4a-9c3b-8f2a6b1c9d10")


def make_point_id(doc) -> str:
    """
    Sinh UUID5 xác định từ (đường dẫn nguồn + nội dung chunk).
    Lý do đổi từ ID tăng dần (start_id + index) sang cách này:
    - Trước đây upload_to_qdrant() nhận start_id mặc định = 0. DAG ingest
      (medical_ingest_dag.py) gọi hàm này KHÔNG truyền start_id ở mỗi lần
      chạy mới -> mỗi lần chạy sau ghi đè (upsert) lên đúng các point ID
      0..N-1 mà lần chạy trước đã tạo, xoá mất dữ liệu PDF cũ mà không báo lỗi.
    - UUID5 xác định theo nội dung giải quyết tận gốc: ingest lại đúng file cũ
      sẽ tự ghi đè đúng chunk của chính nó (idempotent), còn file khác/lần
      chạy khác sinh chunk khác nội dung -> ID khác, không bao giờ đụng nhau.
      Không cần biến đếm toàn cục (next_id) truyền qua các lần gọi nữa.
    """
    source = doc.metadata.get("source", "")
    key = f"{source}::{doc.page_content}"
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, key))

# ── Cấu hình Local Model & Qdrant ──────────────────────────────────────────
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  
VECTOR_SIZE        = 384  
COLLECTION_NAME    = "medical_docs_minilm_384"

EMBED_BATCH_SIZE   = 64   # Giảm nhẹ xuống 64 để CPU xử lý mượt mà hơn
UPSERT_BATCH_SIZE  = 150


# =========================================================================
# FIX (root-cause "Broken DAG ... DagBag import timeout ... after 30.0s"):
# Trước đây `local_embed_model = SentenceTransformer(...)` nằm Ở TOP-LEVEL
# của module -> lệnh này CHẠY NGAY MỖI KHI FILE ĐƯỢC IMPORT, kể cả khi
# Airflow chỉ đang quét thư mục dags/ để dựng DagBag (chưa hề chạy task
# nào). medical_ingest_dag.py import embedder.py -> Airflow vô tình phải
# tải/nạp cả model SentenceTransformer (~470MB, có gọi mạng qua SSL tới
# HuggingFace Hub) MỖI LẦN quét file -> chậm hơn dagbag_import_timeout mặc
# định (30s) -> Airflow coi DAG bị lỗi ("Broken DAG").
#
# Fix: KHÔNG nạp model ở top-level nữa. Dùng lazy-load: model chỉ thực sự
# được nạp lần đầu tiên có hàm nào đó CẦN dùng tới nó (tức là lúc TASK thật
# sự chạy embed_texts_in_batches, không phải lúc Airflow import file để
# parse DAG). Các lần gọi sau tái sử dụng lại instance đã nạp (cache bằng
# biến module-level _local_embed_model).
# =========================================================================
_local_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _local_embed_model
    if _local_embed_model is None:
        print(f"[*] Đang nạp embedding model {EMBEDDING_MODEL} vào CPU...")
        _local_embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _local_embed_model


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY
    )


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    # Idempotent, chạy mỗi lần ingest: tạo index Memgraph nếu chưa có (không
    # lỗi nếu đã tồn tại). Đặt cạnh ensure_collection để cả 2 kho luôn sẵn
    # sàng cùng lúc, không cần bước setup riêng.
    try:
        ensure_indexes()
    except Exception as e:
        print(f"[Memgraph Cảnh báo] Không tạo được index (có thể đã tồn tại): {e}")

    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        # CẢNH BÁO (không tự sửa): collection cũ được tạo trước khi có BM25
        # sẽ không có named sparse vector "bm25" -> upsert điểm mới kèm
        # sparse vector vào collection cũ này sẽ lỗi. Qdrant không hỗ trợ
        # thêm sparse vector config vào collection đã tồn tại — phải xoá và
        # tạo lại (mất dữ liệu cũ, cần re-ingest toàn bộ).
        if SPARSE_VECTOR_NAME not in (info.config.params.sparse_vectors or {}):
            print(
                f"[CẢNH BÁO] Collection '{collection_name}' đã tồn tại nhưng CHƯA có "
                f"sparse vector '{SPARSE_VECTOR_NAME}'. Cần xoá collection này rồi "
                f"chạy lại DAG để tạo mới với hybrid dense+sparse, nếu không upsert "
                f"bên dưới sẽ lỗi."
            )
        print(f"[*] Collection '{collection_name}' đã tồn tại ({info.points_count:,} điểm). Sẽ append thêm.")
        return

    print(f"[*] Tạo mới collection Hybrid Dense+Sparse(BM25) '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False))
        }
    )
    print(f"[+] Đã tạo collection Hybrid thành công ({VECTOR_SIZE}d dense COSINE + sparse BM25).")


def embed_sparse_texts_in_batches(texts: list[str]):
    """Sinh sparse vector BM25 cho danh sách text, cùng nhịp batch với dense
    (embed_texts_in_batches) để 2 model không cùng lúc đè RAM CPU."""
    all_sparse = []
    num_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    model = get_sparse_model()

    for batch_idx, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE)):
        batch = texts[start: start + EMBED_BATCH_SIZE]
        try:
            embeddings = list(model.embed(batch))
            all_sparse.extend(embeddings)
            print(f"[+] Đã sinh sparse BM25 xong batch {batch_idx + 1}/{num_batches}")
            gc.collect()
        except Exception as e:
            print(f"[LỖI] Sparse BM25 batch {batch_idx + 1}/{num_batches} thất bại: {e}")
            raise

    return all_sparse


def embed_texts_in_batches(texts: list[str]) -> list[list[float]]:
    all_vectors = []
    num_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE)):
        batch = texts[start: start + EMBED_BATCH_SIZE]
        try:
            embeddings = get_embed_model().encode(batch, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
            all_vectors.extend(embeddings.tolist())
            print(f"[+] Đã embed xong batch {batch_idx + 1}/{num_batches}")
            # Giải phóng RAM lập tức sau mỗi batch
            del embeddings
            gc.collect()
        except Exception as e:
            print(f"[LỖI] Embedding batch {batch_idx + 1}/{num_batches} thất bại: {e}")
            raise

    return all_vectors


def upload_to_qdrant(
    documents: list,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    LƯU Ý (fix root-cause "Qdrant chỉ còn 80 điểm sau nhiều lần chạy DAG"):
    Trước đây hàm này dùng point_id = start_id + local_idx, với start_id
    mặc định = 0. medical_ingest_dag.py gọi upload_to_qdrant(documents=...)
    KHÔNG truyền start_id -> mỗi lần DAG chạy (mỗi batch file mới) đều bắt
    đầu lại từ ID 0, ghi đè (upsert) lên đúng các point mà lần chạy trước
    đã tạo. Kết quả: dữ liệu của các PDF ingest trước bị mất sạch mà không
    có lỗi nào báo ra, chỉ còn lại chunk của lần chạy DAG gần nhất.

    Fix: point_id giờ là UUID5 xác định (deterministic) theo (đường dẫn
    nguồn + nội dung chunk) qua make_point_id(). Ingest lại đúng file cũ sẽ
    tự ghi đè đúng chunk của chính nó; file khác/lần chạy khác không bao giờ
    trùng ID. Không cần start_id/next_id truyền qua các lần gọi nữa.
    """
    client = get_qdrant_client()
    ensure_collection(client, collection_name)

    texts = [doc.page_content for doc in documents]
    total = len(texts)
    print(f"[*] Tổng {total:,} chunks cần xử lý Dense bằng Local MiniLM...")

    print("[*] Bắt đầu sinh Dense Embedding cục bộ...")
    all_dense_vectors = embed_texts_in_batches(texts)

    print("[*] Bắt đầu sinh Sparse BM25 Embedding cục bộ...")
    all_sparse_vectors = embed_sparse_texts_in_batches(texts)

    points = []
    for doc, dense_vec, sparse_vec in zip(documents, all_dense_vectors, all_sparse_vectors):
        points.append(
            PointStruct(
                id=make_point_id(doc),
                vector={
                    "dense": dense_vec,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            )
        )

    print(f"[*] Upsert {len(points):,} điểm lên Qdrant (batch={UPSERT_BATCH_SIZE})...")
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[start: start + UPSERT_BATCH_SIZE]
        end   = min(start + UPSERT_BATCH_SIZE, len(points))
        client.upsert(collection_name=collection_name, points=batch)
        print(f"  → Upsert {start}–{end} / {len(points)} ✓")

    # =====================================================================
    # GHI SONG SONG SANG MEMGRAPH (chỉ bảng/sơ đồ có ref_id xác định).
    # LÝ DO GỘP THEO ref_id TRƯỚC KHI UPSERT: sơ đồ dài bị split_long_text()
    # (parser.py) chia thành nhiều Document "part_idx" nhưng CÙNG chung 1
    # ref_id (VD Hình 4 phần 1/3, 2/3, 3/3). Nếu upsert từng phần riêng lẻ,
    # MERGE theo ref_key trong Memgraph sẽ GHI ĐÈ lẫn nhau -> chỉ còn phần
    # cuối cùng. Ở đây gộp lại thành 1 node/ref_id với nội dung đầy đủ,
    # đúng thứ tự part_idx, trước khi ghi — Memgraph luôn trả về TOÀN BỘ
    # bảng/sơ đồ, không phải 1 mảnh ngẫu nhiên.
    # =====================================================================
    structured_groups: dict[str, dict] = {}
    for doc, pid in zip(documents, (p.id for p in points)):
        md = doc.metadata
        ref_id = md.get("ref_id", "")
        if not ref_id or not (md.get("is_table") or md.get("is_flowchart")):
            continue
        ref_type = "table" if md.get("is_table") else "flowchart"
        key = f"{ref_type}:{ref_id}"
        group = structured_groups.setdefault(key, {
            "ref_type": ref_type, "ref_id": ref_id,
            "ref_label": md.get("ref_label", ""), "title": md.get("title", ""),
            "source": md.get("source", ""), "file_name": md.get("file_name", ""),
            "parts": [], "point_ids": [],
        })
        group["parts"].append((md.get("part_idx", 1), doc.page_content))
        group["point_ids"].append(pid)

    if structured_groups:
        print(f"[*] Ghi {len(structured_groups):,} bảng/sơ đồ sang Memgraph (song song Qdrant)...")
        for key, g in structured_groups.items():
            merged_content = "\n\n".join(
                content for _, content in sorted(g["parts"], key=lambda x: x[0])
            )
            try:
                upsert_structured_ref(
                    ref_type=g["ref_type"], ref_id=g["ref_id"], ref_label=g["ref_label"],
                    title=g["title"], content=merged_content, source=g["source"],
                    file_name=g["file_name"], qdrant_point_ids=g["point_ids"],
                )
            except Exception as e:
                print(f"[Memgraph Lỗi upsert '{key}']: {e}")
        print(f"[+] Đã đồng bộ {len(structured_groups):,} node sang Memgraph ✓")

    return total
