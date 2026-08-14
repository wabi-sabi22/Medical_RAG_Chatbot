# src/core/graph_store.py
"""
Memgraph: kho tra cứu CHÍNH XÁC theo khoá (ref_type, ref_id) cho bảng/sơ đồ
y khoa (VD "Bảng 10", "Hình 1").

TẠI SAO CẦN TÁCH RIÊNG KHỎI QDRANT:
- Qdrant làm tốt việc tìm theo NGỮ NGHĨA (dense vector similarity), nhưng
  không đảm bảo đúng "Bảng 10" luôn nằm trong top-k, và CrossEncoder rerank
  ở grade_documents() có thể loại nó ra nếu câu mô tả bảng không giống cách
  diễn đạt câu hỏi -> đây chính là lỗi đã gặp trước đó.
- Memgraph lưu theo KHOÁ CHÍNH XÁC ref_key = "<ref_type>:<ref_id>". Khi câu
  hỏi nhắc rõ số bảng/hình (bắt bằng regex _RE_TBL/_RE_FIG trong
  langgraph_flow.py), tra cứu ở đây luôn trả đúng node — không phụ thuộc
  điểm rerank hay ngưỡng lọc nào.

qdrant_point_ids lưu kèm để truy ngược lại đúng các point trong Qdrant đã
sinh ra node này (phục vụ debug/đối chiếu), KHÔNG dùng làm khoá chính —
khoá chính vẫn là ref_key để 2 lần ingest cùng 1 bảng luôn ghi đè đúng chỗ
(idempotent), giống cách make_point_id() làm phía Qdrant.
"""
from neo4j import GraphDatabase
from langchain_core.documents import Document
from src.config import settings

# =========================================================================
# LAZY-LOAD DRIVER: tránh lặp lại lỗi "Broken DAG ... DagBag import timeout"
# đã từng gặp với SentenceTransformer/CrossEncoder (xem embedder.py,
# langgraph_flow.py). GraphDatabase.driver() không gọi mạng ngay lúc tạo
# object (giống QdrantClient) nên về lý thuyết an toàn ở top-level, nhưng
# vẫn lazy-load cho nhất quán và để dễ mock trong test.
# =========================================================================
_driver = None


def get_memgraph_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(settings.MEMGRAPH_URL, auth=None)
    return _driver


def ensure_indexes():
    """Idempotent — an toàn gọi lại nhiều lần (Memgraph tự bỏ qua nếu đã có)."""
    with get_memgraph_driver().session() as session:
        session.run("CREATE INDEX ON :StructuredRef(ref_key)")
        session.run("CREATE INDEX ON :StructuredRef(ref_type)")
        session.run("CREATE INDEX ON :StructuredRef(ref_id)")


def upsert_structured_ref(
    ref_type: str,
    ref_id: str,
    ref_label: str,
    title: str,
    content: str,
    source: str,
    file_name: str,
    qdrant_point_ids: list[str],
) -> None:
    """Ghi (hoặc ghi đè idempotent) 1 node bảng/sơ đồ hoàn chỉnh vào Memgraph.
    Gọi 1 LẦN DUY NHẤT cho mỗi ref_id sau khi đã gộp đủ các phần (xem
    embedder.upload_to_qdrant) — không gọi lặp theo từng Document riêng lẻ,
    tránh ghi đè mất nội dung các phần trước của cùng 1 sơ đồ dài."""
    with get_memgraph_driver().session() as session:
        session.run(
            """
            MERGE (n:StructuredRef {ref_key: $ref_key})
            SET n.ref_type = $ref_type,
                n.ref_id = $ref_id,
                n.ref_label = $ref_label,
                n.title = $title,
                n.content = $content,
                n.source = $source,
                n.file_name = $file_name,
                n.qdrant_point_ids = $point_ids
            """,
            ref_key=f"{ref_type}:{ref_id}",
            ref_type=ref_type,
            ref_id=ref_id,
            ref_label=ref_label,
            title=title,
            content=content,
            source=source,
            file_name=file_name,
            point_ids=qdrant_point_ids,
        )


def query_exact_refs(tbls: list[str], figs: list[str]) -> list[Document]:
    """Tra cứu CHÍNH XÁC theo số bảng/hình được nhắc trong câu hỏi (tbls/figs
    là list số dạng string, VD ["10"], lấy từ _RE_TBL/_RE_FIG.findall() bên
    langgraph_flow.py). Trả về [] ngay nếu câu hỏi không nhắc số nào — tránh
    1 round-trip Memgraph vô ích cho các câu hỏi văn bản thường."""
    if not tbls and not figs:
        return []

    docs: list[Document] = []
    with get_memgraph_driver().session() as session:
        for ref_type, ids in (("table", tbls), ("flowchart", figs)):
            if not ids:
                continue
            result = session.run(
                "MATCH (n:StructuredRef {ref_type: $ref_type}) "
                "WHERE n.ref_id IN $ids RETURN n",
                ref_type=ref_type,
                ids=ids,
            )
            for record in result:
                n = record["n"]
                docs.append(Document(
                    page_content=n["content"],
                    metadata={
                        "source": n.get("source", ""),
                        "file_name": n.get("file_name", ""),
                        "is_table": ref_type == "table",
                        "is_flowchart": ref_type == "flowchart",
                        "exact_match": True,          # cờ để grade_documents() không rerank/lọc
                        "ref_label": n.get("ref_label", ""),
                        "title": n.get("title", ""),
                    },
                ))
    return docs