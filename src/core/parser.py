# src/core/parser.py
import os
import json
import re
import hashlib
from pathlib import Path
from llama_parse import LlamaParse
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from src.config import settings

CACHE_DIR = Path("data/.llamaparse_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


# =========================================================================
# FIX #2 (root-cause "câu hỏi 'Theo Bảng 10 ...' bị chấm 0 điểm"):
# Trước đây MarkdownTextSplitter chunk TOÀN BỘ markdown theo kích thước cố
# định (1000 ký tự) rồi mới đoán chunk nào là bảng. Vì cắt vô điều kiện theo
# độ dài, dòng heading "### Bảng 10 ..." rất hay bị tách sang MỘT CHUNK KHÁC
# với thân bảng của chính nó -> parse_markdown_table_to_json() không thấy
# heading trong chunk đang xử lý -> title rơi về mặc định "Bảng dữ liệu y
# khoa", MẤT LUÔN định danh "Bảng 10" mà câu hỏi lẫn expected_keywords đều
# cần. Fix: quét TOÀN BỘ markdown gốc bằng regex để tìm khối bảng nguyên vẹn
# TRƯỚC khi chunk, đồng thời tìm dòng heading gần nhất PHÍA TRƯỚC khối bảng
# đó trong văn bản gốc (không giới hạn trong 1 chunk) làm title_hint. Nhờ
# vậy heading luôn đi kèm đúng bảng của nó bất kể splitter cắt thế nào.
# =========================================================================
_TABLE_BLOCK_RE = re.compile(
    r'(?P<block>(?:^\|.*\|[ \t]*\n)+)',
    re.MULTILINE
)

_HEADING_HINT_RE = re.compile(r'(bảng|hình|sơ\s*đồ)\s*\d+', re.IGNORECASE)

# =========================================================================
# FIX #5 (root-cause "Memgraph không tra được 'Bảng 10'/'Hình 1' theo đúng
# tên"): trước đây title chỉ dùng để HIỂN THỊ, không có khoá định danh riêng
# biệt cho việc tra cứu chính xác. Thêm ref_id ("10") + ref_label ("Bảng 10")
# tách khỏi title tự do, dùng làm KHOÁ trong Memgraph (xem graph_store.py).
# =========================================================================
_REF_ID_RE = re.compile(r'(bảng|hình|sơ\s*đồ)\s*(\d+)', re.IGNORECASE)


def extract_ref(title: str) -> tuple[str, str]:
    """('10', 'Bảng 10') hoặc ('', '') nếu title không chứa số bảng/hình xác
    định."""
    m = _REF_ID_RE.search(title or "")
    if not m:
        return "", ""
    kind = "Bảng" if m.group(1).lower().startswith("b") else "Hình"
    return m.group(2), f"{kind} {m.group(2)}"


# =========================================================================
# FIX #6 (root-cause "bảng/sơ đồ có trích xuất được nhưng không thấy trong
# Memgraph"): trước đây khi extract_ref() không bắt được số thứ tự (title
# không có dạng "Bảng N"/"Hình N" — VD bảng không có tiêu đề rõ ràng, hoặc
# LlamaParse đặt heading khác định dạng) thì ref_id trả về RỖNG. Ở
# embedder.py, upload_to_qdrant() lọc `if not ref_id: continue` trước khi
# ghi sang Memgraph -> MỌI bảng/sơ đồ không bắt được số bị loại hoàn toàn
# khỏi Memgraph, dù vẫn được LlamaParse/Groq trích xuất và upsert vào Qdrant
# bình thường (đây là lý do Qdrant có đủ chunk nhưng Memgraph thiếu).
#
# Fix: không còn bắt buộc phải khớp được TÊN bảng/hình. Bảng/sơ đồ nào đã
# được trích xuất (vision-extracted) thành công đều PHẢI có ref_id để lên
# Memgraph. Khi không bắt được số, sinh ref_id dự phòng bằng hash ổn định
# theo (file_name + nội dung) — vẫn idempotent: ingest lại đúng file cũ ra
# đúng ref_id cũ (ghi đè đúng chỗ), không đụng ref_id của bảng/sơ đồ khác.
# =========================================================================
def make_fallback_ref(kind: str, file_name: str, content: str) -> tuple[str, str]:
    """Sinh (ref_id, ref_label) dự phòng khi extract_ref() không bắt được số
    thứ tự trong tiêu đề. kind: "table" | "flowchart"."""
    digest = hashlib.sha1(f"{file_name}::{content}".encode("utf-8")).hexdigest()[:12]
    label_kind = "Bảng" if kind == "table" else "Hình"
    return f"unnamed-{digest}", f"{label_kind} (không rõ số) — {file_name}"


def _find_title_before(full_text: str, block_start: int, window: int = 400) -> str:
    """Tìm dòng tiêu đề gần nhất PHÍA TRƯỚC vị trí block_start trong văn bản
    GỐC (không phải trong 1 chunk đã cắt) — ưu tiên dòng nhắc "Bảng N" /
    "Hình N", quét lùi tối đa `window` ký tự để không cuốn nhầm cả đoạn văn
    không liên quan làm tiêu đề."""
    preceding = full_text[max(0, block_start - window):block_start]
    lines = [l.strip() for l in preceding.strip().split('\n') if l.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if _HEADING_HINT_RE.search(line):
            return line.lstrip('#').strip()
    last = lines[-1]
    if last.startswith('#') or len(last) < 150:
        return last.lstrip('#').strip()
    return ""


def parse_markdown_table_to_json(md_text: str, title_hint: str = "") -> dict:
    """Phân tích bảng Markdown thành JSON, có cơ chế TỰ ĐỘNG ĐIỀN KHUYẾT
    (Forward-Fill) cho các ô bị gộp. `title_hint` lấy từ dòng heading tìm
    thấy PHÍA TRƯỚC bảng trong markdown GỐC (xem _find_title_before) — nên
    không còn bị mất định danh kiểu "Bảng 10" dù bảng bị chunk tách heading."""
    lines = md_text.strip().split('\n')
    headers = []
    rows = []
    title = title_hint or "Bảng dữ liệu y khoa"
    prev_cells = []  # Lưu lại dòng trước đó để đắp dữ liệu

    for line in lines:
        line = line.strip()
        if not line.startswith('|') and not line.endswith('|'):
            if line and len(line) < 150:
                title = line.replace("#", "").strip()
            continue

        clean_line = line.strip('|')
        cells = [cell.strip() for cell in clean_line.split('|')]

        if all(re.match(r'^-+$', c.replace(':', '')) for c in cells if c):
            continue

        if not headers:
            headers = cells
        else:
            # TÍNH NĂNG ĐIỀN KHUYẾT: Nếu cột 1 trống, lấy dữ liệu cột 1 của dòng trên đắp xuống
            if len(cells) > 0 and cells[0] == "" and prev_cells and len(prev_cells) > 0:
                cells[0] = prev_cells[0]

            prev_cells = cells
            rows.append({"cells": cells, "group": "Chung"})

    # ---> FIX LỖI "VẠN VẬT ĐỀU LÀ BẢNG" <---
    # Nếu bảng chỉ có 1 cột hoặc cột đầu tiên tên là "Nội dung", 
    # hệ thống từ chối nhận diện đây là bảng để fallback về text thuần.
    if len(headers) <= 1 or (len(headers) > 0 and "nội dung" in headers[0].lower()):
        return {}

    return {"title": title, "headers": headers, "rows": rows, "drugs": []}


# =========================================================================
# FIX #1 (root-cause "bảng/sơ đồ hay bị loại khỏi top-k retrieval/rerank"):
# embedder.py và langgraph_flow.py đều nhúng/so khớp trực tiếp doc.page_content
# bằng SentenceTransformer/CrossEncoder — các model này huấn luyện trên VĂN
# BẢN TỰ NHIÊN, không phải cú pháp JSON. Trước đây page_content của bảng là
# json.dumps(table_json) (kiểu {"title":...,"headers":[...],"rows":[{"cells":
# [...]}...]}) nên độ tương đồng ngữ nghĩa với câu hỏi tiếng Việt tự nhiên
# thấp hơn hẳn so với 1 đoạn text thường -> bảng dễ thua trong top-k dense
# retrieval và bị loại ở ngưỡng lọc rerank (grade_documents). Fix: chuyển
# JSON bảng -> mô tả văn bản tự nhiên để dùng làm page_content (embed +
# rerank), JSON gốc vẫn giữ nguyên trong metadata["json_data"] để generate()
# dùng khi cần.
# =========================================================================
def table_json_to_natural_text(table_json: dict) -> str:
    title = table_json.get("title", "Bảng dữ liệu y khoa")
    headers = table_json.get("headers", [])
    rows = table_json.get("rows", [])

    lines = [f"{title}."]
    if headers:
        lines.append("Các cột: " + ", ".join(h for h in headers if h) + ".")

    for row in rows:
        cells = row.get("cells", [])
        if not cells:
            continue
        if headers and len(headers) == len(cells):
            # Ghép "Tên cột: giá trị" theo từng cặp header-cell để câu văn có
            # ngữ nghĩa rõ ràng thay vì chỉ nối các ô bằng dấu "|" như cũ.
            pairs = [f"{h}: {c}" for h, c in zip(headers, cells) if h and c]
            if pairs:
                lines.append("- " + "; ".join(pairs))
        else:
            lines.append("- " + " | ".join(c for c in cells if c))

    return "\n".join(l for l in lines if l)


def parse_markdown_flowchart_to_json(md_text: str, title_hint: str = "") -> dict:
    """Đóng gói Flowchart thành JSON. KHÔNG còn cắt cứng ở 800 ký tự như
    trước (dễ mất đúng nhánh rẽ mà câu hỏi hỏi tới) — giữ nguyên toàn bộ nội
    dung, việc chia nhỏ (nếu quá dài) do get_chunks_from_pdf() đảm nhiệm ở
    bước sau bằng split_long_text(), mỗi mảnh vẫn giữ tiêu đề để nhận diện."""
    lines = md_text.strip().split('\n')
    title = title_hint or (lines[0][:100].replace("#", "").strip() if lines else "Lưu đồ y khoa")

    return {
        "title": title,
        "nodes": [{"id": "N1", "type": "info", "text": md_text.strip()}],
        "edges": []
    }


def extract_json_with_groq(text_content: str, extract_type: str, title_hint: str = "") -> dict:
    """
    HÀM ĐÃ ĐƯỢC THAY THẾ BẰNG PYTHON THUẦN.
    Giữ lại tên hàm gốc để không làm hỏng các file khác.
    """
    try:
        if extract_type == "table":
            return parse_markdown_table_to_json(text_content, title_hint)
        else:
            return parse_markdown_flowchart_to_json(text_content, title_hint)
    except Exception as e:
        print(f"[LỖI Fast Parser]: {e}")
        return {}


# =========================================================================
# FIX #3 (root-cause "Hình 1/4/5/6 nhiều nhánh bị hỏi trúng đúng phần bị cắt"):
# parse_markdown_flowchart_to_json() bản cũ cắt cứng md_text[:800], nên nếu
# nhánh câu hỏi hỏi tới nằm sau ký tự thứ 800 thì dữ liệu đã mất TRƯỚC KHI
# kịp embed. Fix: không cắt nữa; nếu văn bản dài hơn chunk_size, chia thành
# NHIỀU Document liên tiếp, MỖI Document đều được gắn tiêu đề ở đầu (VD
# "Hình 4: ... (phần 2/3)") để dù retrieval chỉ khớp đúng 1 mảnh, LLM vẫn
# biết mảnh đó thuộc hình nào — không còn dữ liệu nào bị mất khỏi hệ thống.
# =========================================================================
def split_long_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = text.strip()
    n = len(text)
    if n <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Cố gắng cắt ở ranh giới dòng/câu gần nhất để không cắt ngang từ/ý
            boundary = text.rfind('\n', start, end)
            if boundary <= start:
                boundary = text.rfind('. ', start, end)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap)

    return chunks


def get_chunks_from_pdf(file_path: str) -> list[Document]:
    file_name = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)
    cache_file = CACHE_DIR / f"{file_hash}.md"

    # 1. LLAMAPARSE NHẬN DIỆN (TÍCH HỢP SẴN VISION)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            full_markdown = f.read()
    else:
        print(f"[*] Đang nạp '{file_name}' bằng LlamaParse...")
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="vi",
            premium_mode=True,
            parsing_instruction=(
                "Bạn là hệ thống phân tích tài liệu y khoa. "
                "1. BẢNG BIỂU: Trích xuất thành Markdown Table. Đảm bảo không gộp ô sai lệch. "
                "2. SƠ ĐỒ/LƯU ĐỒ (Flowchart): Nhận diện MỌI lưu đồ chẩn đoán/điều trị. Tóm tắt chi tiết "
                "các bước rẽ nhánh và MÔ TẢ ĐÓ PHẢI NẰM TRONG CẶP THẺ [START_FLOWCHART] và [END_FLOWCHART]."
            )
        )
        parsed_docs = parser.load_data(file_path)
        full_markdown = "\n\n".join([doc.text for doc in parsed_docs])

        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(full_markdown)

    documents = []

    # 2. XỬ LÝ SƠ ĐỒ (FLOWCHART) — GIỮ TOÀN BỘ NỘI DUNG, CHIA MẢNH CÓ TIÊU ĐỀ
    flowchart_pattern = re.compile(r'\[START_FLOWCHART\](.*?)\[END_FLOWCHART\]', re.DOTALL)
    for match in flowchart_pattern.finditer(full_markdown):
        flowchart_text = match.group(1).strip()
        if not flowchart_text:
            continue

        title_hint = _find_title_before(full_markdown, match.start())
        flowchart_json = extract_json_with_groq(flowchart_text, "flowchart", title_hint)
        if not flowchart_json or "nodes" not in flowchart_json:
            continue

        title = flowchart_json.get("title", "Lưu đồ y khoa")
        ref_id, ref_label = extract_ref(title)
        if not ref_id:
            ref_id, ref_label = make_fallback_ref("flowchart", file_name, flowchart_text)
        full_text = flowchart_json["nodes"][0]["text"]
        text_parts = split_long_text(full_text, chunk_size=900, overlap=120)
        total_parts = len(text_parts)

        for idx, part in enumerate(text_parts, start=1):
            part_label = f" (phần {idx}/{total_parts})" if total_parts > 1 else ""
            # Luôn gắn tiêu đề ở đầu MỖI mảnh -> mảnh nào cũng "tự biết" thuộc
            # hình nào, nên câu hỏi kiểu "Hình 4 ..." vẫn match được dù
            # retrieval chỉ lấy đúng 1 mảnh trong số nhiều mảnh của hình đó.
            page_content = f"{title}{part_label}\n{part}"
            part_json = {
                "title": title,
                "nodes": [{"id": f"N{idx}", "type": "info", "text": part}],
                "edges": []
            }
            documents.append(Document(
                page_content=page_content,
                metadata={
                    "source": file_path, "file_name": file_name,
                    "is_flowchart": True, "is_table": False, "json_data": part_json,
                    # ref_id/ref_label CHUNG cho mọi phần của cùng 1 hình -> khi
                    # upsert vào Memgraph (embedder.py) các phần sẽ MERGE lại
                    # đúng 1 node theo ref_key "flowchart:<ref_id>" (ghi đè bằng
                    # phần cuối cùng lặp qua, nội dung Qdrant vẫn giữ đủ từng
                    # phần riêng để retrieval bám sát đoạn câu hỏi hỏi tới).
                    "title": title, "ref_id": ref_id, "ref_label": ref_label,
                    "part_idx": idx, "total_parts": total_parts,
                }
            ))

    # Loại bỏ flowchart khỏi text chính để không bị xử lý lại ở bước chunk text thường
    main_markdown = flowchart_pattern.sub('', full_markdown)

    # 3. TRÍCH BẢNG TRỰC TIẾP TỪ MARKDOWN GỐC — TRƯỚC KHI CHUNK — để heading
    #    "Bảng N" luôn đi liền với đúng bảng của nó (xem giải thích FIX #2 ở trên).
    table_spans = []
    for match in _TABLE_BLOCK_RE.finditer(main_markdown):
        block = match.group('block')
        # Cần tối thiểu 2 dòng (header + dòng phân cách |---|---|)
        if block.count('\n') < 2 or '-' not in block:
            continue

        title_hint = _find_title_before(main_markdown, match.start())
        table_json = extract_json_with_groq(block, "table", title_hint)
        if table_json and table_json.get("rows"):
            table_text = table_json_to_natural_text(table_json)
            title = table_json.get("title", "Bảng dữ liệu y khoa")
            ref_id, ref_label = extract_ref(title)
            if not ref_id:
                ref_id, ref_label = make_fallback_ref("table", file_name, table_text)
            documents.append(Document(
                page_content=table_text,
                metadata={
                    "source": file_path, "file_name": file_name,
                    "is_table": True, "is_flowchart": False, "json_data": table_json,
                    "title": title, "ref_id": ref_id, "ref_label": ref_label,
                }
            ))
            table_spans.append((match.start(), match.end()))

    # Xoá các khối bảng đã trích khỏi văn bản chính để không bị chunk trùng lần 2
    for start, end in sorted(table_spans, reverse=True):
        main_markdown = main_markdown[:start] + main_markdown[end:]

    # 4. CHUNKING PHẦN TEXT THƯỜNG CÒN LẠI
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=150)
    raw_chunks = text_splitter.create_documents([main_markdown])

    for doc in raw_chunks:
        content = doc.page_content.strip()
        if not content:
            continue

        # Lưới an toàn: phòng khi vẫn còn sót bảng nhỏ chưa bắt được ở bước 3
        # (hiếm gặp, VD bảng nằm sát mép do LlamaParse xuất định dạng lạ).
        if "|" in content and "-|-" in content.replace(" ", ""):
            table_json = extract_json_with_groq(content, "table")
            if table_json and table_json.get("rows"):
                table_text = table_json_to_natural_text(table_json)
                title = table_json.get("title", "Bảng dữ liệu y khoa")
                ref_id, ref_label = extract_ref(title)
                if not ref_id:
                    ref_id, ref_label = make_fallback_ref("table", file_name, table_text)
                documents.append(Document(
                    page_content=table_text,
                    metadata={
                        "source": file_path, "file_name": file_name,
                        "is_table": True, "is_flowchart": False, "json_data": table_json,
                        "title": title, "ref_id": ref_id, "ref_label": ref_label,
                    }
                ))
                continue

        if len(content) > 50:
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": file_path, "file_name": file_name,
                    "is_table": False, "is_flowchart": False
                }
            ))

    return documents