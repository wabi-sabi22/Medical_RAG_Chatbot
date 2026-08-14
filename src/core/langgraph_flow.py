# src/core/langgraph_flow.py
import os, re, unicodedata, json
import numpy as np
import diskcache as dc
from typing import TypedDict, List
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from groq import Groq, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import logging
import time as _time

logger = logging.getLogger("rag_flow")
logging.basicConfig(level=logging.INFO)

_RE_RETRY_AFTER = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")

def _extract_retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "headers"):
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    m = _RE_RETRY_AFTER.search(str(exc))
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0.0
        return minutes * 60 + float(m.group(2))
    return None

def _groq_wait(retry_state):
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitError):
        wait_s = _extract_retry_after_seconds(exc)
        if wait_s is not None:
            logger.warning(f"[Groq TPD/rate limit] Server yêu cầu chờ {wait_s:.1f}s — đang chờ đúng thời gian này (không đoán mù).")
            return wait_s + 1.5  
    return wait_exponential(multiplier=2, min=2, max=20)(retry_state)

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from fastembed import SparseTextEmbedding
from src.config import settings
from src.core.graph_store import query_exact_refs

# ==========================================
# 1. CẤU HÌNH & TỪ ĐIỂN Y KHOA 
# ==========================================
COLLECTION_NAME, RETRIEVE_LIMIT, RERANK_TOP_K, GREETING_SIM_THRESHOLD = "medical_docs_minilm_384", 8, 3, 0.55
groq_client = Groq(api_key=settings.GROQ_API_KEY, timeout=20.0)

qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in text if unicodedata.category(c) != "Mn")).lower()

STOPWORDS = {"là","gì","thế","nào","bao","nhiêu","cho","tôi","hỏi","biết","bạn","có","thể","giúp","với","không","được","của","trong","theo","dựa","vào","về","cách","làm","sao","như","khi","để","ở","tại","xin","chào","vậy","này","kia","đó","các","những","cái","chi","tiết","thông","tin","dưới","đây","trên","ra","ai","mấy","cần","nên","phải","hay","hoặc","và","từ","đến","nếu","thì","nhưng","vì","do","bởi","rằng","lại","còn","cũng","đang","đã","sẽ","rồi","chưa","từng","rất","quá","lắm","hơn","nhất","bị"}

MED_ABBR = {
    "btm":"bệnh thận mạn", "bmv":"bệnh mạch vành", "đtđ":"đái tháo đường", "dtđ":"đái tháo đường", 
    "ucmc":"ức chế men chuyển", "ưcmc":"ức chế men chuyển", "ctta":"chẹn thụ thể angiotensin", 
    "cb":"chẹn beta", "ckca":"chẹn kênh canxi", "lt":"lợi tiểu", "mra":"kháng aldosterone", 
    "sglt2i":"ức chế kênh đồng vận chuyển natri glucose 2", "hfref":"suy tim phân suất tống máu giảm", 
    "hfpef":"suy tim phân suất tống máu bảo tồn", "tha":"tăng huyết áp", "hatt":"huyết áp tâm thu", 
    "hattr":"huyết áp tâm trương", "kks":"khó kiểm soát", "hapk":"huyết áp phòng khám", 
    "hatn":"huyết áp tại nhà", "halt":"huyết áp liên tục", "ytnc":"yếu tố nguy cơ", 
    "od":"tổn thương cơ quan đích", "must":"sàng lọc dinh dưỡng phổ cập", 
    "nrs":"tầm soát nguy cơ suy dinh dưỡng", "sga":"đánh giá dinh dưỡng chủ quan", 
    "mna":"đánh giá dinh dưỡng tối thiểu", "pcr":"tỷ lệ protein creatinine niệu", 
    "acr":"tỷ lệ albumin creatinine niệu", "ace":"ức chế men chuyển", "arb":"chẹn thụ thể angiotensin",
    "esrd":"bệnh thận mạn giai đoạn cuối", "gfr":"mức lọc cầu thận", "egfr":"mức lọc cầu thận ước đoán",
    "pth":"hormone tuyến cận giáp", "fgf-23":"yếu tố tăng trưởng nguyên bào sợi 23", 
    "dexa":"đo mật độ xương bằng tia x", "nrs-2002":"tầm soát nguy cơ suy dinh dưỡng",
    "ttr":"thời gian trong ranh giới đích", "map":"huyết áp trung bình", 
    "hellp":"hội chứng hellp tiền sản giật", "tma":"bệnh lý vi mạch huyết khối", 
    "rdn":"triệt đốt thần kinh giao cảm động mạch thận", "mica":"kháng nguyên mica",
    "iv ig":"globulin miễn dịch tĩnh mạch", "atg":"kháng thể đa dòng kháng bạch cầu lympho",
    "ccpd":"lọc màng bụng liên tục bằng máy", "capd":"lọc màng bụng liên tục ngoại trú",
    "arni":"ức chế thụ thể angiotensin neprilysin", "glp-1 ra":"đồng vận thụ thể glp-1",
    "maces":"biến cố tim mạch chính", "hfmref":"suy tim phân suất tống máu giảm nhẹ",
    "ais":"đột quỵ thiếu máu não cục bộ cấp", "epo":"erythropoietin",
    
    # BỔ SUNG TỪ KHÓA ESC 2024 / Y KHOA QUỐC TẾ:
    "office bp":"huyết áp phòng khám",
    "hbpm":"huyết áp tại nhà",
    "abpm":"huyết áp lưu động",
    "sbp":"huyết áp tâm thu",
    "dbp":"huyết áp tâm trương",
    "cvd":"bệnh tim mạch",
    "ckd":"bệnh thận mạn",
    "hmod":"tổn thương cơ quan đích do tăng huyết áp",
    "alara":"mức thấp nhất có thể đạt được một cách hợp lý",
    "aerobic":"thể dục nhịp điệu",
    "resistance training":"tập lực cơ",
    "white-coat":"áo choàng trắng",
    "masked":"ẩn giấu",
    "elevated bp":"huyết áp tăng cao"
}

DOMAIN_TERMS = [
    "amlodipine","furosemide","bisoprolol","captopril","losartan","valsartan","diltiazem",
    "nifedipine","enalapril","suy tim","mạch vành","đái tháo đường","đột quy","bệnh thận",
    "ho khan","thai kỳ","tâm thần","gút","hen","có thai","kháng trị","tăng huyết áp",
    "phân tầng","cơn tha","suy dinh dưỡng","espen","bmi","albumin","muac","ldl-cholesterol",
    "natri","loãng xương","bmd","phốt pho","protein","chất xơ","chế độ ăn","thực đơn",
    "năng lượng","kcal","canxi","vitamin d","chất béo","béo phì","protein niệu","quai henlé",
    "tiền sản giật","đái máu","thận hư","prednisolone","iga","lupus","mlct","viêm bàng quang",
    "trimethoprim","fluoroquinolones","candida","fluconazol","nang đơn thận","fena","banff",
    "sỏi thận","sỏi tiết niệu","ứ nước","ứ mủ",
    "cyclophosphamide", "chlorambucil", "azathioprine", "cyclosporine a", "mycophenolate mofetil", 
    "rituximab", "bexarotene", "methotrexate", "plicamycin", "temozolomide", "procarbazine",
    "bendroflumethiazide", "chlorthalidone", "hydrochlorothiazide", "indapamide", "bumetanide", 
    "torsemide", "amiloride", "eplerenone", "triamterene",
    "rifampicine", "metronidazole", "norfloxacin", "ofloxacin", "cefuroxime", "cefotaxime", 
    "ceftriaxone", "cefoperazone", "azithromycin", "doxycyclin", "erythromycin", "itraconazol",
    "benazepril", "fosinopril", "lisinopril", "perindopril", "quinapril", "ramipril", 
    "trandolapril", "imidapril", "azilsartan", "candesartan", "eprosatan", "irbesartan", 
    "olmesartan", "telmisartan", "verapamil", "felodipine", "isradipine", "nitrendipine", 
    "lercanidipine", "nicardipine", "acebutalol", "atenolol", "carvedilol", "labetalol", 
    "metoprolol succinate", "metoprolol tartrate", "nadolol", "nebivolol", "propranolol", 
    "esmolol", "aliskiren", "doxazosin", "prazosin", "terazosin", "hydralazine", "minoxidil", 
    "clonidine", "methyldopa", "reserpine", "nitroprusside", "urapidil", "nitroglycerine",
    "erythropoietin", "epoetin alfa", "epoetin beta", "darbepoetin alfa", "mircera", 
    "ferurnoxytol", "sắt carboxymaltose", "sắt isomaltoside", "carotenoid", "lycopen", 
    "resveratrol", "phytosterol",
    "protein bence-jones", "protein tamm-horsfall", "hội chứng thận hư", "bệnh thận iga", 
    "viêm thận lupus", "thải ghép tối cấp", "thải ghép cấp tế bào", "phản ứng màng lọc type a", 
    "phản ứng màng lọc type b", "hội chứng mất quân bình", "microprotein niệu", 
    "tiểu albumine vi lượng", "c4d", "c5b-9", "c3",
    "thể dục nhịp điệu", "tập lực cơ", "esc 2024", "vsh/vnha"
]

MED_ABBR_ND = {strip_diacritics(k): v for k, v in MED_ABBR.items()}
DOMAIN_TERMS_ND = {strip_diacritics(t): t for t in DOMAIN_TERMS}

_RE_TBL = re.compile(r'\b(?:bang|table|tbl)\s*(\d+)', re.I)
_RE_FIG = re.compile(r'\b(?:hinh|so\s*do|luu\s*do|figure|fig)\.?\s*(\d+)', re.I)
_RE_DOS = re.compile(r'\b\d+([.,]\d+)?\s*(mg|g|ml|mmol|mcg|kg)\b', re.I)
_RE_LAB = re.compile(r'\begfr\b|\bmlct\b|\bcreatinine\b|\bacr\b|\bpcr\b', re.I)

_TBL_LIKE_WORDS = ["lieu", "bao nhieu", "bang", "toi da", "mg", "hinh", "so do", "luu do", "table", "figure", "fig"]

def _is_table_like(q_norm: str) -> bool:
    """Nhận diện câu hỏi có khả năng liên quan bảng/hình, kể cả khi không nêu số cụ thể
    (VD: câu hỏi nối tiếp về liều thuốc mà không lặp lại 'Bảng 10')."""
    return any(w in q_norm for w in _TBL_LIKE_WORDS)

class LocalVnNER:
    def __call__(self, text: str) -> list:
        entities, seen = [], set()
        def _add(w: str, tag: str):
            if (k := strip_diacritics(w)) not in seen and k:
                seen.add(k); entities.append({"word": w, "entity": tag})
        
        nt = strip_diacritics(text)
        for m in _RE_TBL.finditer(nt): _add(f"bảng {m.group(1)}", "TABLE_REF")
        for m in _RE_FIG.finditer(nt): _add(f"hình/sơ đồ {m.group(1)}", "FIGURE_REF")
        for m in _RE_DOS.finditer(nt): _add(m.group(0), "DOSAGE")
        for m in _RE_LAB.finditer(nt): _add(m.group(0), "LAB_VALUE")
        for n, o in DOMAIN_TERMS_ND.items():
            if n in nt: _add(o, "MEDICAL_TERM")
        for w in nt.replace(",", " ").replace(".", " ").replace("?", " ").split():
            if w in MED_ABBR_ND: _add(MED_ABBR_ND[w], "MEDICAL_ABBR")
            if len(w) > 1 and w not in STOPWORDS: _add(w, "CLEAN_KEYWORD")
        return entities

ner_pipeline = LocalVnNER()

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH
# ==========================================
_local_embed_model = None
_rerank_model = None
_greet_vecs = None
_sparse_model = None

_GREET_SEED_SENTENCES = ["Xin chào", "Chào bạn", "Chào bác sĩ", "Alo", "Hế lô", "Bạn khỏe không", "Bạn là ai", "Cảm ơn", "Tạm biệt"]

def get_embed_model() -> SentenceTransformer:
    global _local_embed_model
    if _local_embed_model is None:
        print("[*] Đang nạp SentenceTransformer (MiniLM) lên CPU...")
        _local_embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return _local_embed_model

def get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        print("[*] Đang nạp CrossEncoder rerank model lên CPU...")
        _rerank_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")
    return _rerank_model

def get_greet_vecs():
    global _greet_vecs
    if _greet_vecs is None:
        _greet_vecs = get_embed_model().encode(_GREET_SEED_SENTENCES, convert_to_tensor=True)
    return _greet_vecs

def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        print("[*] Đang nạp sparse BM25 model (Qdrant/bm25)...")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model

class GraphState(TypedDict):
    question: str; intent: str; entities: List[dict]; question_vector: List[float]
    raw_documents: List[Document]; filtered_documents: List[Document]; generation: str; grade: str

# ==========================================
# 3. CÁC NODE XỬ LÝ
# ==========================================
def intent_pre_process(s: GraphState) -> dict:
    sims = util.cos_sim(get_embed_model().encode(s["question"], convert_to_tensor=True), get_greet_vecs())[0]
    return {"intent": "greeting" if float(sims.max()) >= GREETING_SIM_THRESHOLD else "medical_query", "entities": ner_pipeline(s["question"])}

def embed_question(s: GraphState) -> dict:
    q, ents = s["question"], s.get("entities", [])
    eq = f"Tra cứu y khoa: {q} ({' '.join([e['word'] for e in ents])})" if ents else f"Tra cứu y khoa, tim mạch, huyết áp, thận: {q}"
    return {"question_vector": get_embed_model().encode(eq).tolist()}

def retrieve(s: GraphState) -> dict:
    q_norm = strip_diacritics(s["question"])
    tbls, figs = list(set(_RE_TBL.findall(q_norm))), list(set(_RE_FIG.findall(q_norm)))

    try:
        exact_docs = query_exact_refs(tbls, figs)
    except Exception as e:
        print(f"[Memgraph Err]: {e}")
        exact_docs = []

    limit = RETRIEVE_LIMIT * 2 if (tbls or figs or _is_table_like(q_norm)) else RETRIEVE_LIMIT

    try:
        sparse_q = next(iter(get_sparse_model().embed([s["question"]])))
        pts = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=s["question_vector"], using="dense", limit=limit * 3),
                Prefetch(
                    query=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()),
                    using="bm25",
                    limit=limit * 3,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        ).points
        qdrant_docs = [Document(page_content=h.payload.get("page_content", ""), metadata=h.payload.get("metadata", {})) for h in pts]
    except Exception as e:
        print(f"[Qdrant Hybrid Err]: {e} — fallback về dense-only")
        try:
            pts = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=s["question_vector"], using="dense", limit=limit).points
            qdrant_docs = [Document(page_content=h.payload.get("page_content", ""), metadata=h.payload.get("metadata", {})) for h in pts]
        except Exception as e2:
            print(f"[Qdrant Err]: {e2}")
            qdrant_docs = []

    exact_labels = {d.metadata.get("ref_label") for d in exact_docs if d.metadata.get("ref_label")}
    qdrant_docs = [d for d in qdrant_docs if d.metadata.get("ref_label") not in exact_labels]

    return {"raw_documents": exact_docs + qdrant_docs}

def grade_documents(s: GraphState) -> dict:
    if not s["raw_documents"]: return {"filtered_documents": [], "grade": "not_found"}
    q_norm = strip_diacritics(s["question"])
    # [THAY ĐỔI 2]: Bổ sung các từ khóa nhận diện bảng, hình, sơ đồ, lưu đồ (dùng chung với retrieve())
    is_tbl = _is_table_like(q_norm)

    exact_docs = [d for d in s["raw_documents"] if d.metadata.get("exact_match")]
    rerank_pool = [d for d in s["raw_documents"] if not d.metadata.get("exact_match")]

    top_k = 8 if is_tbl else RERANK_TOP_K
    f_docs = list(exact_docs)
    remaining = top_k - len(f_docs)

    if rerank_pool and remaining > 0:
        scores = 1 / (1 + np.exp(-np.asarray(get_rerank_model().predict([[s["question"], d.page_content] for d in rerank_pool]))))
        scored = []
        for d, score in zip(rerank_pool, scores):
            fs = float(score) + (0.05 if (is_tbl and any(k in d.page_content.lower() for k in ["bảng", "hình", "sơ đồ", "lưu đồ", "["])) else 0.0)
            scored.append((d, fs))
        scored.sort(key=lambda x: x[1], reverse=True)
        # Nới ngưỡng cắt cho câu hỏi dạng bảng: bảng nhiều trang có thể có dòng khớp thấp hơn top-1 khá nhiều
        margin = 0.4 if is_tbl else 0.3
        threshold = max(0.05, scored[0][1] - margin) if scored else 0.05
        f_docs += [d for d, c in scored if c >= threshold][:remaining]

    return {"filtered_documents": f_docs, "grade": "found" if f_docs else "not_found"}

_TRANSIENT_GROQ_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_TPD_CACHE = dc.Cache("cache/tpd_usage")
_DAILY_TPD_LIMIT = int(os.environ.get("GROQ_TPD_LIMIT", 500_000))
_TPD_WARN_THRESHOLD = int(os.environ.get("GROQ_TPD_WARN_AT", 30_000))

def _utc_day_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _track_tpd_usage(total_tokens: int) -> tuple[int, int]:
    key = _utc_day_key()
    used = (_TPD_CACHE.get(key) or 0) + total_tokens
    _TPD_CACHE.set(key, used, expire=90_000)
    remaining = _DAILY_TPD_LIMIT - used
    if remaining < _TPD_WARN_THRESHOLD:
        logger.warning(f"[TPD CẢNH BÁO] Còn lại ước tính ~{remaining:,}/{_DAILY_TPD_LIMIT:,} token hôm nay (UTC).")
    return used, remaining

def get_tpd_status() -> dict:
    used = _TPD_CACHE.get(_utc_day_key()) or 0
    return {"used": used, "limit": _DAILY_TPD_LIMIT, "remaining": _DAILY_TPD_LIMIT - used}

@retry(
    wait=_groq_wait,
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(_TRANSIENT_GROQ_ERRORS),
    reraise=True,
)
def call_llm(msgs: list) -> str:
    try:
        resp = groq_client.chat.completions.create(
            messages=msgs, model=settings.LLM_MODEL_NAME, temperature=0.0, max_tokens=450, timeout=15.0
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            used, remaining = _track_tpd_usage(usage.total_tokens)
            logger.info(f"[Groq usage] +{usage.total_tokens} token | Hôm nay: {used:,}/{_DAILY_TPD_LIMIT:,} (còn ~{remaining:,})")
        return resp.choices[0].message.content
    except RateLimitError as e:
        if "tokens per day" in str(e).lower() or "TPD" in str(e):
            logger.error(f"[Groq TPD gần/đã cạn] {e}.")
        raise
    except _TRANSIENT_GROQ_ERRORS:
        raise
    except Exception as e:
        logger.error(f"[Groq lỗi vĩnh viễn - không retry] {type(e).__name__}: {e}")
        raise

def format_content_safely(content: str, max_len: int = 700) -> str:
    content = content.strip()
    if len(content) <= max_len:
        return content
    cut = content.rfind('\n', 0, max_len)
    if cut < max_len * 0.5:  
        cut = max_len
    return content[:cut].rstrip() + "..."

def generate(s: GraphState) -> dict:
    v_docs = []
    for i, d in enumerate(s["filtered_documents"]):
        is_structured = d.metadata.get("is_table") or d.metadata.get("is_flowchart")
        # [THAY ĐỔI 1]: Tăng mạnh max_len để tránh cắt cụt bảng và lưu đồ
        max_len = 8000 if is_structured else 3000
        content = format_content_safely(d.page_content, max_len=max_len)
        v_docs.append(f"<doc id='{i+1}'>\n{content}\n</doc>")

    # [THAY ĐỔI 3]: Chỉnh lại prompt một chút để không cấm ngặt việc LLM nội suy các con số đơn giản từ văn bản (VD: tính phần trăm từ tổng số người)
    pmpt = (
        "Bạn là trợ lý y khoa. CHỈ trả lời các câu hỏi dựa trên NGỮ CẢNH (được đặt trong thẻ <doc>) được cung cấp dưới đây.\n"
        "Bắt buộc tuân thủ 3 quy tắc sau:\n"
        "1. Ưu tiên trích xuất chính xác thông tin từ văn bản. Nếu câu hỏi yêu cầu tính toán dựa trên số liệu có sẵn trong ngữ cảnh (như tính tỷ lệ phần trăm hoặc ước tính số lượng), hãy thực hiện phép tính cẩn thận và trả lời.\n"
        "2. Trích xuất chính xác nguyên văn các con số, ngày tháng, và thuật ngữ y khoa từ văn bản gốc.\n"
        "3. Nếu ngữ cảnh chứa thuật ngữ tiếng Anh hoặc viết tắt y khoa, có thể dịch sang tiếng Việt để trả lời ,đảm bảo tính chính xác.\n"
        
    )
    pmpt += "\n--- NGỮ CẢNH ---\n" + "\n".join(v_docs)
    
    try: 
        return {"generation": call_llm([{"role": "system", "content": pmpt}, {"role": "user", "content": s["question"]}])}
    except Exception as e: 
        return {"generation": f"[ERROR] Hết token hoặc lỗi LLM: {e}"}

# ==========================================
# 4. XÂY DỰNG GRAPH
# ==========================================
def fast_greet(s: GraphState): return {"generation": "Xin chào! Tôi là trợ lý y khoa..."}
def not_found(s: GraphState): return {"generation": "Xin lỗi, không tìm thấy tài liệu nội bộ."}

graph = StateGraph(GraphState)
for n, f in [("intent_pre_process", intent_pre_process), ("fast_greet", fast_greet), ("embed_question", embed_question), ("retrieve", retrieve), ("grade_documents", grade_documents), ("generate", generate), ("not_found", not_found)]: graph.add_node(n, f)
graph.set_entry_point("intent_pre_process")
graph.add_conditional_edges("intent_pre_process", lambda s: s["intent"], {"greeting": "fast_greet", "medical_query": "embed_question"})
graph.add_edge("embed_question", "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", lambda s: s["grade"], {"found": "generate", "not_found": "not_found"})
for e in ["generate", "not_found", "fast_greet"]: graph.add_edge(e, END)
rag_graph = graph.compile()

def query_rag_system(question: str) -> dict:
    res = rag_graph.invoke({
        "question": question, 
        "intent": "", 
        "entities": [], 
        "question_vector": [], 
        "raw_documents": [], 
        "filtered_documents": [], 
        "generation": "", 
        "grade": ""
    })
    return res