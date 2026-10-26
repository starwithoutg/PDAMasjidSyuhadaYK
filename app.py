import os
import re
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

APP_DIR = Path(__file__).parent
DEFAULT_XLSX = APP_DIR / "data" / "Metadata_DCMA_RAG_Masjid_Syuhada.xlsx"
SHEET_CONFIG = {
    "Metadata_Dublin_Core": {
        "kind": "dokumen sumber",
        "title_cols": ["dc:title", "record_id"],
        "text_cols": ["dc:description", "dc:subject", "evidence_note", "dc:coverage", "dc:source", "dc:creator"],
        "id_col": "record_id",
        "page_col": "pdf_pages",
    },
    "Elemen_Arsitektur": {
        "kind": "elemen arsitektur",
        "title_cols": ["dc:title", "element_code", "element_id"],
        "text_cols": ["form_description", "dc:description", "material", "finish", "location_text",
                      "room_location_code", "quantity", "significance_source_code", "significance_label",
                      "condition_summary", "dc:subject", "evidence_text", "rag_keywords"],
        "id_col": "element_id",
        "page_col": "source_page_pdf",
    },
    "Inventarisasi_Kerusakan": {
        "kind": "rekod kerusakan",
        "title_cols": ["dc:title", "damage_code_source", "damage_id"],
        "text_cols": ["element_category", "damage_type", "damage_description", "suspected_cause_source",
                      "severity", "recommended_action", "building_mass", "room_or_location_code",
                      "evidence_text", "rag_keywords"],
        "id_col": "damage_id",
        "page_col": "source_page_pdf",
    },
    "Panduan_Pelestarian": {
        "kind": "kebijakan pelestarian",
        "title_cols": ["dc:title", "policy_id"],
        "text_cols": ["priority_source", "element_or_scope", "recommendation_source_text",
                      "interpretation_note", "evidence_text", "rag_keywords"],
        "id_col": "policy_id",
        "page_col": "source_page_pdf",
    },
    "Chunk_RAG": {
        "kind": "chunk RAG",
        "title_cols": ["chunk_title", "chunk_id"],
        "text_cols": ["chunk_text", "keywords", "retrieval_filter_hint"],
        "id_col": "chunk_id",
        "page_col": "source_page_pdf",
    },
    "Pemetaan_DC": {
        "kind": "pemetaan Dublin Core",
        "title_cols": ["dcmi_element"],
        "text_cols": ["definisi_ringkas", "penggunaan_dalam_proyek", "contoh_nilai", "catatan_validasi"],
        "id_col": "dcmi_element",
        "page_col": None,
    },
    "Controlled_Vocab": {
        "kind": "kosakata terkendali",
        "title_cols": ["field", "value"],
        "text_cols": ["definition_or_usage", "source_or_rule"],
        "id_col": "field",
        "page_col": None,
    },
}

st.set_page_config(
    page_title="Chatbot Dokumentasi Masjid Syuhada",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.7rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#e7f4f3,#f5f7fb); border: 1px solid #d4e5e5;}
.hero h1 {margin: 0; color: #164e63; font-size: 2rem;}
.hero p {margin: .5rem 0 0; color: #334155;}
.small-muted {font-size: .85rem; color: #64748b;}
.source-card {border: 1px solid #dbe3ea; border-radius: 10px; padding: .8rem 1rem; margin: .4rem 0;}
</style>
<div class="hero">
  <h1>Chatbot Dokumentasi Masjid Syuhada</h1>
  <p>Asisten pencarian berbasis metadata Dublin Core untuk elemen arsitektur, inventarisasi kerusakan, dan panduan pelestarian.</p>
</div>
""", unsafe_allow_html=True)

def clean_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()

def row_to_record(sheet_name: str, row: dict) -> dict | None:
    cfg = SHEET_CONFIG[sheet_name]
    row = {str(k).strip(): v for k, v in row.items()}
    title = next((clean_value(row.get(c)) for c in cfg["title_cols"] if clean_value(row.get(c))), "")
    if not title:
        return None
    rid = clean_value(row.get(cfg["id_col"])) or f"{sheet_name}-{abs(hash(title)) % 10**8}"
    parts = []
    for col, value in row.items():
        value = clean_value(value)
        if value and col not in {"doc_url"}:
            parts.append(f"{col}: {value}")
    body = " | ".join(parts)
    # Build a natural-language searchable text without losing source field names.
    searchable = " ".join([title, cfg["kind"], body])
    source_id = clean_value(row.get("source_id")) or clean_value(row.get("record_id")) or ""
    source_title = clean_value(row.get("dc:source")) or clean_value(row.get("source_title")) or sheet_name
    page = clean_value(row.get(cfg["page_col"])) if cfg["page_col"] else ""
    location = clean_value(row.get("room_location_code")) or clean_value(row.get("location_text"))
    code = clean_value(row.get("element_code")) or clean_value(row.get("damage_code_source"))
    return {
        "id": rid, "title": title, "kind": cfg["kind"], "sheet": sheet_name,
        "text": searchable, "source_id": source_id, "source_title": source_title,
        "page": page, "location": location, "code": code, "raw": row,
    }

@st.cache_data(show_spinner=False)
def load_records(file_path: str, file_mtime: float):
    xls = pd.ExcelFile(file_path)
    records = []
    sheet_counts = {}
    for sheet_name, cfg in SHEET_CONFIG.items():
        if sheet_name not in xls.sheet_names:
            continue
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=object)
        df.columns = [str(c).strip() for c in df.columns]
        count_before = len(records)
        for row in df.to_dict(orient="records"):
            rec = row_to_record(sheet_name, row)
            if rec and rec["text"].strip():
                records.append(rec)
        sheet_counts[sheet_name] = len(records) - count_before
    # Remove duplicates by source sheet and stable record ID.
    unique = {}
    for rec in records:
        unique[(rec["sheet"], rec["id"])] = rec
    return list(unique.values()), sheet_counts, xls.sheet_names

def tokenize(text: str):
    return re.findall(r"[a-zA-ZÀ-ÿ0-9_.-]+", text.lower())

def keyword_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    overlap = len(q & t) / max(len(q), 1)
    phrase_bonus = 0.2 if query.lower() in text.lower() else 0.0
    return overlap + phrase_bonus

@st.cache_resource(show_spinner=False)
def build_tfidf_index(texts: tuple[str, ...]):
    if TfidfVectorizer is None:
        return None, None
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", lowercase=True)
    matrix = vectorizer.fit_transform(list(texts))
    return vectorizer, matrix

def get_openai_client():
    api_key = ""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def retrieve(query: str, records: list[dict], top_k: int, mode: str):
    if not records:
        return []
    if mode == "Keyword":
        ranked = [(keyword_score(query, r["text"]), r) for r in records]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **r} for s, r in ranked[:top_k] if s > 0]

    # TF-IDF is local, no API key required.
    texts = tuple(r["text"] for r in records)
    vectorizer, matrix = build_tfidf_index(texts)
    if vectorizer is None:
        ranked = [(keyword_score(query, r["text"]), r) for r in records]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **r} for s, r in ranked[:top_k] if s > 0]
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [{"score": float(scores[i]), **records[i]} for i in indices if scores[i] > 0]

def answer_with_openai(query: str, retrieved: list[dict], client, model: str):
    context_blocks = []
    for i, r in enumerate(retrieved, 1):
        citation = f"[S{i}] {r['source_title']} | ID: {r['id']} | Sheet: {r['sheet']} | Halaman: {r['page'] or 'tidak tercantum'}"
        context_blocks.append(citation + "\n" + r["text"])
    context = "\n\n---\n\n".join(context_blocks)
    system_prompt = """
Anda adalah chatbot dokumentasi arsitektur Masjid Syuhada. Jawab dalam bahasa Indonesia yang jelas.
Gunakan hanya informasi pada KONTEKS yang diberikan. Jangan mengarang fakta, nomor halaman, bahan,
kondisi, penyebab kerusakan, atau rekomendasi. Jika konteks tidak cukup, katakan bahwa data belum
ditemukan atau perlu verifikasi pada dokumen sumber. Bedakan fakta sumber dari dugaan yang ditulis
dalam sumber. Jangan menganggap dugaan penyebab sebagai kepastian. Untuk jawaban faktual, sertakan
sitasi [S1], [S2] sesuai konteks. Jangan memberi diagnosis struktur atau rekomendasi konservasi final
tanpa dukungan eksplisit sumber. Jika sumber mencatat halaman "sekitar", beri tahu pengguna bahwa
nomor halaman perlu diverifikasi.
"""
    user_prompt = f"PERTANYAAN:\n{query}\n\nKONTEKS SUMBER:\n{context}\n\nJawab ringkas tetapi cukup lengkap dan sertakan sitasi."
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or "Tidak ada jawaban yang dihasilkan."

def fallback_answer(query: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return ("Saya belum menemukan rekod yang cukup relevan dalam workbook. "
                "Coba gunakan istilah elemen, kode, material, lokasi ruang, kerusakan, atau prioritas pelestarian yang lebih spesifik.")
    lines = [
        "Saya menemukan rekod berikut. Mode ini menggunakan pencarian lokal dan belum membuat jawaban generatif dari model bahasa.",
        "",
    ]
    for i, r in enumerate(retrieved[:4], 1):
        lines.append(f"{i}. **{r['title']}** ({r['kind']})")
        lines.append(f"   - Rekod: `{r['id']}` | Sumber: {r['source_title']} | Halaman: {r['page'] or 'tidak tercantum'}")
        # Keep answer grounded and short.
        raw = r["raw"]
        preferred = ["form_description", "dc:description", "damage_description",
                     "recommendation_source_text", "chunk_text", "evidence_note",
                     "dc:subject", "material", "finish", "priority_source"]
        summary = next((clean_value(raw.get(k)) for k in preferred if clean_value(raw.get(k))), "")
        if not summary:
            summary = r["text"][:500]
        lines.append(f"   - Informasi: {summary}")
        lines.append(f"   - Sitasi: [S{i}]")
    lines.append("")
    lines.append("Catatan: rekod dengan halaman bertanda 'sekitar' perlu diperiksa kembali pada PDF asli.")
    return "\n".join(lines)

with st.sidebar:
    st.header("Pengaturan")
    uploaded_file = st.file_uploader("Unggah workbook metadata (.xlsx)", type=["xlsx"])
    default_path = str(DEFAULT_XLSX)
    if uploaded_file is not None:
        temp_path = APP_DIR / "uploaded_metadata.xlsx"
        temp_path.write_bytes(uploaded_file.getvalue())
        data_path = str(temp_path)
        data_mtime = temp_path.stat().st_mtime
    elif DEFAULT_XLSX.exists():
        data_path = default_path
        data_mtime = DEFAULT_XLSX.stat().st_mtime
    else:
        data_path = ""
        data_mtime = 0

    search_mode = st.radio("Metode retrieval", ["TF-IDF (disarankan)", "Keyword"], index=0)
    top_k = st.slider("Jumlah sumber yang dicari", min_value=2, max_value=8, value=5)
    st.caption("TF-IDF berjalan lokal dan tidak membutuhkan API key.")
    st.divider()
    st.subheader("Mode jawaban")
    client = get_openai_client()
    use_llm = st.checkbox("Gunakan OpenAI untuk merangkai jawaban", value=bool(client))
    model_name = st.text_input("Model OpenAI", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if use_llm and not client:
        st.warning("OPENAI_API_KEY belum tersedia. Aplikasi akan memakai jawaban berbasis rekod tanpa generasi LLM.")
    st.divider()
    st.caption("Publikasikan hanya data yang memang boleh diakses publik. Jangan unggah API key ke repositori.")

if not data_path:
    st.error("File Excel default tidak ditemukan. Unggah file .xlsx melalui sidebar.")
    st.stop()

try:
    records, sheet_counts, workbook_sheets = load_records(data_path, data_mtime)
except Exception as exc:
    st.error(f"Gagal membaca workbook Excel: {exc}")
    st.stop()

with st.sidebar:
    st.success(f"{len(records)} rekod siap dicari")
    with st.expander("Rincian rekod per sheet"):
        for name, count in sheet_counts.items():
            st.write(f"**{name}**: {count}")

col1, col2, col3 = st.columns(3)
col1.metric("Total rekod terindeks", len(records))
col2.metric("Sheet yang diproses", len(sheet_counts))
col3.metric("Mode pencarian", "TF-IDF" if search_mode.startswith("TF-IDF") else "Keyword")

st.markdown("### Ajukan pertanyaan")
examples = [
    "Apa material atap dak beton tingkat 1?",
    "Apa saja prioritas 1 dalam Panduan Pelestarian?",
    "Apa kerusakan yang tercatat pada jendela?",
    "Apa arti kode signifikansi IS?",
]
selected_example = st.selectbox("Contoh pertanyaan", ["Pilih contoh..."] + examples)
default_question = "" if selected_example == "Pilih contoh..." else selected_example

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Lihat sumber yang digunakan"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['title']}**  \n"
                        f"Jenis: {src['kind']} · ID: `{src['id']}`  \n"
                        f"Sumber: {src['source_title']} · Halaman: {src['page'] or 'tidak tercantum'}"
                    )
                    st.caption(f"Sheet: {src['sheet']} | Kode: {src['code'] or '-'} | Lokasi: {src['location'] or '-'}")

question = st.chat_input("Tulis pertanyaan tentang elemen, kerusakan, atau pelestarian...")
if question is None and default_question:
    if st.button("Gunakan contoh pertanyaan"):
        question = default_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    method = "Keyword" if search_mode == "Keyword" else "TF-IDF"
    with st.spinner("Mencari rekod yang relevan..."):
        results = retrieve(question, records, top_k, method)
        if use_llm and client and results:
            try:
                answer = answer_with_openai(question, results, client, model_name)
            except Exception as exc:
                answer = fallback_answer(question, results)
                st.warning(f"OpenAI tidak berhasil digunakan. Menampilkan hasil pencarian lokal. Detail: {exc}")
        else:
            answer = fallback_answer(question, results)

    sources = [{
        "id": r["id"], "title": r["title"], "kind": r["kind"],
        "source_title": r["source_title"], "page": r["page"],
        "sheet": r["sheet"], "code": r["code"], "location": r["location"],
        "score": round(float(r["score"]), 4),
    } for r in results]

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("Sumber yang ditemukan", expanded=True):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[S{i}] {src['title']}**  \n"
                        f"Jenis: {src['kind']} · ID: `{src['id']}` · Sheet: `{src['sheet']}`  \n"
                        f"Sumber: {src['source_title']} · Halaman: {src['page'] or 'tidak tercantum'}"
                    )
                    st.caption(f"Skor retrieval: {src['score']} | Kode: {src['code'] or '-'} | Lokasi: {src['location'] or '-'}")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

st.divider()
with st.expander("Tentang data dan keterbatasan"):
    st.write(
        "Aplikasi ini membaca workbook metadata Dublin Core dan mencari rekod dari sheet dokumen, "
        "elemen arsitektur, inventarisasi kerusakan, panduan pelestarian, chunk RAG, pemetaan Dublin Core, "
        "dan kosakata terkendali. Retrieval default memakai TF-IDF. Jika OPENAI_API_KEY tersedia dan "
        "opsi OpenAI diaktifkan, hasil pencarian dipakai sebagai konteks untuk menyusun jawaban. "
        "Kualitas jawaban dibatasi oleh kelengkapan workbook. Rekod contoh dan nomor halaman yang belum pasti "
        "harus diverifikasi sebelum publikasi sebagai sumber resmi."
    )
