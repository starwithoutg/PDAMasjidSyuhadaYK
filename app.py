import os
import re
import html
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
    "Metadata_Dublin_Core": {"kind": "dokumen sumber", "title_cols": ["dc:title", "record_id"], "id_col": "record_id", "page_col": "pdf_pages"},
    "Elemen_Arsitektur": {"kind": "elemen arsitektur", "title_cols": ["dc:title", "element_code", "element_id"], "id_col": "element_id", "page_col": "source_page_pdf"},
    "Inventarisasi_Kerusakan": {"kind": "rekod kerusakan", "title_cols": ["dc:title", "damage_code_source", "damage_id"], "id_col": "damage_id", "page_col": "source_page_pdf"},
    "Panduan_Pelestarian": {"kind": "panduan pelestarian", "title_cols": ["dc:title", "policy_id"], "id_col": "policy_id", "page_col": "source_page_pdf"},
    "Chunk_RAG": {"kind": "potongan konteks", "title_cols": ["chunk_title", "chunk_id"], "id_col": "chunk_id", "page_col": "source_page_pdf"},
    "Pemetaan_DC": {"kind": "pemetaan Dublin Core", "title_cols": ["dcmi_element"], "id_col": "dcmi_element", "page_col": None},
    "Controlled_Vocab": {"kind": "kosakata terkendali", "title_cols": ["field", "value"], "id_col": "field", "page_col": None},
}

st.set_page_config(page_title="Dokumentasi Masjid Syuhada", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.35rem;padding-bottom:3rem}
.hero{padding:1.5rem 1.7rem;border-radius:18px;background:linear-gradient(125deg,#102d3b,#1c5961);color:#fff;margin-bottom:1rem}
.hero h1{font-size:1.85rem;line-height:1.2;margin:0;color:#fff}
.hero p{margin:.55rem 0 0;color:#e3f2f3;font-size:1rem;max-width:850px}
.kicker{font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:#a9d5d6;font-weight:700;margin-bottom:.5rem}
.info-strip{padding:.8rem 1rem;border:1px solid rgba(128,145,160,.35);border-radius:12px;background:rgba(128,145,160,.07);margin:.8rem 0 1.2rem}
.answer-box{padding:1rem 1.1rem;border:1px solid rgba(80,150,130,.4);border-radius:13px;background:rgba(80,150,130,.08);margin:.3rem 0 .7rem}
.source-box{border:1px solid rgba(128,145,160,.3);border-radius:12px;padding:.8rem 1rem;margin:.5rem 0}
.source-title{font-weight:700;font-size:1rem;margin-bottom:.25rem}
.muted{color:#8b98a8;font-size:.86rem}
.stChatMessage{border-radius:14px}
[data-testid="stMetric"]{border:1px solid rgba(128,145,160,.25);padding:.7rem .85rem;border-radius:12px}
</style>
<div class="hero"><div class="kicker">Arsip · Arsitektur · Pelestarian</div><h1>Asisten Dokumentasi Masjid Syuhada</h1><p>Temukan informasi elemen arsitektur, lokasi yang tercatat, kerusakan, dan arahan pelestarian. Setiap jawaban disertai sumber agar mudah diperiksa kembali.</p></div>
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
    parts = [f"{col}: {clean_value(value)}" for col, value in row.items() if clean_value(value) and col != "doc_url"]
    body = " | ".join(parts)
    source_id = clean_value(row.get("source_id")) or clean_value(row.get("record_id"))
    source_title = clean_value(row.get("dc:source")) or clean_value(row.get("source_title")) or sheet_name
    # Workbook columns sometimes contain trailing spaces (for example, "doc_url ").
    doc_url = next((clean_value(v) for k, v in row.items()
                    if str(k).strip().lower() in {"doc_url", "document_url", "google_drive_url", "url_gdrive", "source_url"}
                    and clean_value(v).startswith(("https://", "http://"))), "")
    if not doc_url:
        identifier = clean_value(row.get("dc:identifier"))
        if "drive.google.com/" in identifier or "docs.google.com/" in identifier:
            doc_url = identifier
    page = clean_value(row.get(cfg["page_col"])) if cfg["page_col"] else ""
    location = clean_value(row.get("room_or_location_code")) or clean_value(row.get("room_location_code")) or clean_value(row.get("location_text"))
    code = clean_value(row.get("element_code")) or clean_value(row.get("damage_code_source"))
    return {"id": rid, "title": title, "kind": cfg["kind"], "sheet": sheet_name, "text": f"{title} {cfg['kind']} {body}",
            "source_id": source_id, "source_title": source_title, "page": page, "location": location, "code": code, "doc_url": doc_url, "raw": row}


@st.cache_data(show_spinner=False)
def load_records(file_path: str, file_mtime: float):
    xls = pd.ExcelFile(file_path)
    records, sheet_counts = [], {}
    for sheet_name in SHEET_CONFIG:
        if sheet_name not in xls.sheet_names:
            continue
        df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=object)
        df.columns = [str(c).strip() for c in df.columns]
        before = len(records)
        for row in df.to_dict(orient="records"):
            rec = row_to_record(sheet_name, row)
            if rec:
                records.append(rec)
        sheet_counts[sheet_name] = len(records) - before
    unique = {}
    for rec in records:
        unique[(rec["sheet"], rec["id"])] = rec
    records = list(unique.values())
    # Resolve document links for RAG chunks and other rows that refer to a source ID.
    urls_by_source_id, urls_by_source_title = {}, {}
    for rec in records:
        if rec.get("doc_url"):
            if rec.get("source_id"):
                urls_by_source_id.setdefault(rec["source_id"], rec["doc_url"])
            if rec.get("source_title"):
                urls_by_source_title.setdefault(normalize(rec["source_title"]), rec["doc_url"])
    for rec in records:
        if not rec.get("doc_url"):
            rec["doc_url"] = urls_by_source_id.get(rec.get("source_id"), "") or urls_by_source_title.get(normalize(rec.get("source_title", "")), "")
    return records, sheet_counts, xls.sheet_names


def tokenize(text: str):
    return re.findall(r"[a-zA-ZÀ-ÿ0-9_.-]+", text.lower())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-ZÀ-ÿ0-9 ]+", " ", str(text).lower())).strip()


@st.cache_resource(show_spinner=False)
def build_tfidf_index(texts: tuple[str, ...]):
    if TfidfVectorizer is None:
        return None, None
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", lowercase=True, sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(list(texts))


def classify_query(query: str) -> dict:
    q = normalize(query)
    intent = "general"
    if any(x in q for x in ["berapa", "jumlah", "ada berapa", "hitung"]): intent = "count"
    elif any(x in q for x in ["di mana", "dimana", "lokasi", "letak", "berada di", "terletak"]): intent = "location"
    elif any(x in q for x in ["kerusakan", "rusak", "retak", "mengelupas", "bocor", "lapuk", "korosi"]): intent = "damage"
    elif any(x in q for x in ["prioritas", "pelestarian", "konservasi", "perbaikan", "penanganan", "rekomendasi"]): intent = "preservation"
    elif any(x in q for x in ["material", "bahan", "finishing", "terbuat dari", "bentuk", "ukuran", "signifikansi", "kode"]): intent = "element"
    target = None
    for term, synonyms in {
        "pintu": ["pintu", "daun pintu", "kusen pintu"], "jendela": ["jendela", "kaca jendela"],
        "atap": ["atap", "dak beton", "kubah", "talang"], "kaca": ["kaca"], "kusen": ["kusen"],
        "kubah": ["kubah"], "talang": ["talang"],
    }.items():
        if any(s in q for s in synonyms):
            target = term
            break
    damage_type = "retak" if "retak" in q else ("mengelupas" if "mengelupas" in q else None)
    return {"intent": intent, "target": target, "damage_type": damage_type, "query": q}


def record_matches_target(rec: dict, target: str | None) -> bool:
    if not target: return True
    raw = rec["raw"]
    fields = " ".join(clean_value(raw.get(k)) for k in ["dc:title", "element_category", "element_type", "damage_type", "damage_description", "dc:subject", "rag_keywords", "element_or_scope", "chunk_text"])
    f = normalize(fields)
    aliases = {"pintu": ["pintu"], "jendela": ["jendela"], "atap": ["atap", "dak beton", "kubah", "talang"], "kaca": ["kaca"], "kusen": ["kusen"], "kubah": ["kubah"], "talang": ["talang"]}
    return any(a in f for a in aliases.get(target, [target]))


def record_matches_damage(rec: dict, qinfo: dict) -> bool:
    if rec["sheet"] != "Inventarisasi_Kerusakan":
        return False
    if not record_matches_target(rec, qinfo["target"]): return False
    if qinfo["damage_type"]:
        fields = normalize(" ".join(clean_value(rec["raw"].get(k)) for k in ["dc:title", "damage_type", "damage_description", "rag_keywords"]))
        return qinfo["damage_type"] in fields
    return True


def retrieve(query: str, records: list[dict], top_k: int, mode: str):
    qinfo = classify_query(query)
    pool = records
    # Route by question intent so similar words in unrelated record types do not dominate.
    if qinfo["intent"] in ("count", "damage"):
        damage_pool = [r for r in records if r["sheet"] == "Inventarisasi_Kerusakan" and record_matches_target(r, qinfo["target"])]
        if qinfo["damage_type"]:
            typed = [r for r in damage_pool if record_matches_damage(r, qinfo)]
            pool = typed if typed else damage_pool
        else: pool = damage_pool
    elif qinfo["intent"] == "location":
        pool = [r for r in records if r["sheet"] == "Elemen_Arsitektur" and record_matches_target(r, qinfo["target"])]
        if not pool: pool = [r for r in records if record_matches_target(r, qinfo["target"]) and r["sheet"] in ("Chunk_RAG", "Metadata_Dublin_Core")]
    elif qinfo["intent"] == "preservation":
        pool = [r for r in records if r["sheet"] == "Panduan_Pelestarian"]
    elif qinfo["intent"] == "element":
        pool = [r for r in records if r["sheet"] == "Elemen_Arsitektur" and record_matches_target(r, qinfo["target"])] or records
    if not pool: return [], qinfo
    if mode == "Keyword" or TfidfVectorizer is None:
        qtokens = set(tokenize(query))
        ranked = []
        for r in pool:
            rtokens = set(tokenize(r["text"]))
            score = len(qtokens & rtokens) / max(len(qtokens), 1)
            if qinfo["target"] and record_matches_target(r, qinfo["target"]): score += .45
            if qinfo["damage_type"] and qinfo["damage_type"] in normalize(r["text"]): score += .4
            ranked.append((score, r))
    else:
        texts = tuple(r["text"] for r in records)
        vectorizer, matrix = build_tfidf_index(texts)
        if vectorizer is None: return retrieve(query, records, top_k, "Keyword")
        qvec = vectorizer.transform([query])
        all_scores = cosine_similarity(qvec, matrix).ravel()
        by_key = {(r["sheet"], r["id"]): float(all_scores[i]) for i, r in enumerate(records)}
        ranked = []
        for r in pool:
            score = by_key.get((r["sheet"], r["id"]), 0.0)
            if qinfo["target"] and record_matches_target(r, qinfo["target"]): score += .35
            if qinfo["damage_type"] and qinfo["damage_type"] in normalize(r["text"]): score += .3
            ranked.append((score, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, rec in ranked:
        if score <= 0: continue
        # Avoid showing the same underlying record twice when both source row and RAG chunk exist.
        parent = clean_value(rec["raw"].get("parent_record_id"))
        if rec["sheet"] == "Chunk_RAG" and any(x["id"] == parent for x in selected): continue
        selected.append({"score": float(score), **rec})
        if len(selected) >= top_k: break
    return selected, qinfo


def get_openai_client():
    api_key = ""
    try: api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception: pass
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key or OpenAI is None: return None
    return OpenAI(api_key=api_key)


def source_label(rec: dict) -> str:
    page = rec["page"] or "tidak tercantum"
    if "sekitar" in page.lower(): page += " (perlu verifikasi)"
    return f"{rec['source_title']}, halaman {page}"


def answer_local(query: str, results: list[dict], qinfo: dict) -> str:
    if not results:
        return "Saya belum menemukan informasi yang cukup untuk menjawab pertanyaan itu di metadata yang tersedia. Coba sebutkan nama elemen, kode, atau jenis kerusakannya."
    intent, target = qinfo["intent"], qinfo["target"]
    if intent == "count":
        exact = [r for r in results if r["sheet"] == "Inventarisasi_Kerusakan" and record_matches_damage(r, qinfo)]
        if qinfo["damage_type"] and target:
            label = f"rekod kerusakan {target} dengan kondisi {qinfo['damage_type']}"
        elif target and qinfo["damage_type"]:
            label = f"rekod {target} yang mengalami {qinfo['damage_type']}"
        elif target: label = f"rekod kerusakan yang berkaitan dengan {target}"
        else: label = "rekod kerusakan yang sesuai dengan pertanyaan"
        if not exact:
            return f"Saya belum menemukan rekod yang secara jelas memenuhi kriteria {label} di workbook yang tersedia. Ini tidak membuktikan bahwa kerusakan tersebut tidak ada; data inventarisasi yang dimuat mungkin belum lengkap."
        names = "; ".join(r["title"] for r in exact)
        return f"Di workbook yang saat ini dimuat, saya menemukan **{len(exact)} rekod** {label}. Rekod yang cocok: {names}. Angka ini merujuk pada data yang tersedia di aplikasi, bukan jaminan jumlah keseluruhan di lapangan."
    if intent == "location":
        useful = [r for r in results if r["sheet"] == "Elemen_Arsitektur"]
        if not useful:
            return "Saya belum menemukan lokasi yang dapat dipastikan dari metadata yang tersedia. Rekod yang ditemukan belum menyebut lokasi spesifik untuk elemen yang ditanyakan, jadi saya tidak ingin menebaknya."
        bits = []
        for r in useful[:3]:
            loc = r["location"] or clean_value(r["raw"].get("location_text"))
            if loc: bits.append(f"**{r['title']}** tercatat di {loc}")
        if not bits:
            return "Saya menemukan rekod elemen yang berkaitan, tetapi kolom lokasinya belum terisi. Lokasi perlu diverifikasi pada gambar teknis atau PDF sumber."
        return "Berdasarkan metadata yang tersedia, " + "; ".join(bits) + ". Saya belum dapat memastikan lokasi yang lebih spesifik daripada keterangan tersebut."
    if intent in ("damage",):
        damage = [r for r in results if r["sheet"] == "Inventarisasi_Kerusakan"]
        if not damage:
            return "Saya belum menemukan rekod kerusakan yang sesuai di metadata yang tersedia."
        statements = []
        for r in damage[:4]:
            raw = r["raw"]
            detail = clean_value(raw.get("damage_description")) or clean_value(raw.get("evidence_text"))
            cause = clean_value(raw.get("suspected_cause_source"))
            s = f"**{r['title']}**: {detail}"
            if cause and any(x in cause.lower() for x in ["diduga", "dugaan"]): s += f" Sumber menyebut penyebabnya sebagai dugaan: {cause}"
            statements.append(s)
        return "Dalam inventarisasi yang dimuat, saya menemukan: " + " ".join(statements) + " Lokasi rinci atau tingkat keparahan yang tidak tercantum tidak saya simpulkan sendiri."
    if intent == "preservation":
        items = []
        for r in results[:5]:
            raw = r["raw"]
            priority = clean_value(raw.get("priority_source"))
            recommendation = clean_value(raw.get("recommendation_source_text")) or r["title"]
            items.append(f"{priority + ': ' if priority else ''}{recommendation}")
        return "Panduan pelestarian mencatat hal-hal berikut: " + "; ".join(items) + ". Rincian metode pelaksanaan tetap perlu diperiksa pada dokumen sumber."
    # Natural concise answer for element/document questions.
    r = results[0]
    raw = r["raw"]
    facts = []
    for label, key in [("Material", "material"), ("Finishing", "finish"), ("Lokasi", "room_location_code"), ("Bentuk", "form_description"), ("Jumlah", "quantity"), ("Signifikansi pada sumber", "significance_source_code")]:
        val = clean_value(raw.get(key))
        if val: facts.append(f"{label.lower()} {val}")
    if facts: return f"Untuk **{r['title']}**, metadata mencatat " + "; ".join(facts) + "."
    summary = next((clean_value(raw.get(k)) for k in ["dc:description", "evidence_text", "chunk_text", "definition_or_usage", "recommendation_source_text"] if clean_value(raw.get(k))), "")
    if summary: return f"**{r['title']}**: {summary}"
    return "Saya menemukan rekod yang mungkin relevan, tetapi isinya belum cukup untuk memberi jawaban yang pasti."


def answer_with_openai(query: str, results: list[dict], client, model: str, qinfo: dict):
    context_blocks = []
    for i, r in enumerate(results, 1):
        context_blocks.append(f"[S{i}] {r['title']} | Jenis: {r['kind']} | ID: {r['id']} | Sumber: {source_label(r)}\n{r['text']}")
    system = """Anda adalah asisten dokumentasi arsitektur Masjid Syuhada. Jawab dengan bahasa Indonesia yang natural, langsung, dan tidak kaku. Jawab pertanyaan pengguna, bukan sekadar mendaftar hasil pencarian. Gunakan hanya konteks sumber. Jangan mengarang lokasi, jumlah, material, halaman, kondisi, atau penyebab. Untuk pertanyaan jumlah, hitung hanya rekod unik yang memenuhi seluruh kriteria dan jelaskan bahwa hitungan terbatas pada workbook yang tersedia. Untuk lokasi, sebutkan lokasi hanya jika tertulis jelas; jika tidak, katakan belum tersedia. Jangan mencampur elemen yang berbeda hanya karena memiliki kata yang sama. Bedakan dugaan sumber dari fakta. Cantumkan sitasi [S1] setelah klaim yang didukung. Jika halaman tertulis 'sekitar', tandai perlu diverifikasi. Jika data tidak cukup, jelaskan secara singkat."""
    user = f"Pertanyaan: {query}\nMaksud pertanyaan terdeteksi: {qinfo['intent']}; target: {qinfo['target'] or 'tidak spesifik'}.\nKonteks:\n" + "\n\n---\n\n".join(context_blocks) + "\n\nSusun jawaban singkat dan natural."
    response = client.chat.completions.create(model=model, temperature=0.1, messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return response.choices[0].message.content or answer_local(query, results, qinfo)


with st.sidebar:
    st.header("Sumber data")
    uploaded_file = st.file_uploader("Unggah workbook metadata (.xlsx)", type=["xlsx"])
    if uploaded_file is not None:
        temp_path = APP_DIR / "uploaded_metadata.xlsx"
        temp_path.write_bytes(uploaded_file.getvalue())
        data_path = str(temp_path)
        data_mtime = temp_path.stat().st_mtime
    elif DEFAULT_XLSX.exists():
        data_path, data_mtime = str(DEFAULT_XLSX), DEFAULT_XLSX.stat().st_mtime
    else: data_path, data_mtime = "", 0
    st.divider()
    st.header("Pencarian")
    search_mode = st.radio("Metode pencarian", ["TF-IDF", "Kata kunci"], index=0)
    top_k = st.slider("Jumlah sumber pendukung", 2, 8, 4)
    st.caption("Pencarian memprioritaskan kategori sesuai maksud pertanyaan, bukan hanya kemiripan kata.")
    st.divider()
    st.header("Penyusunan jawaban")
    client = get_openai_client()
    use_llm = st.checkbox("Gunakan AI untuk menyusun jawaban", value=bool(client))
    model_name = st.text_input("Model AI", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if use_llm and not client: st.warning("API key belum tersedia. Aplikasi akan memakai penyusunan jawaban lokal yang lebih ringkas.")
    st.divider()
    st.caption("Untuk publikasi, pastikan metadata dan nomor halaman telah diverifikasi terhadap dokumen asli.")

if not data_path:
    st.error("Workbook Excel belum tersedia. Unggah file .xlsx pada panel kiri.")
    st.stop()
try:
    records, sheet_counts, workbook_sheets = load_records(data_path, data_mtime)
except Exception as exc:
    st.error(f"Workbook tidak dapat dibaca: {exc}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Rekod tersedia", len(records))
m2.metric("Kategori data", len(sheet_counts))
m3.metric("Mode jawaban", "AI + sumber" if use_llm and client else "Lokal + sumber")
st.markdown("### Apa yang ingin Anda ketahui?")
st.caption("Tanyakan dengan bahasa sehari-hari. Contoh berikut dapat dipilih untuk memulai.")
examples = [
    "Ada berapa rekod pintu yang mengalami retak?",
    "Di mana lokasi elemen atap dak beton tingkat 1?",
    "Apa material atap dak beton tingkat 1?",
    "Kerusakan apa yang tercatat pada jendela?",
    "Apa saja yang masuk Prioritas 1 pelestarian?",
]
selected_example = st.selectbox("Contoh pertanyaan", ["Pilih contoh pertanyaan..."] + examples, label_visibility="collapsed")
if "messages" not in st.session_state: st.session_state.messages = []
if st.button("Mulai percakapan baru", use_container_width=False): st.session_state.messages = []; st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sumber pendukung ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[S{i}] {src['title']}**  \n{src['source_title']} · ID `{src['id']}` · Halaman {src['page'] or 'tidak tercantum'}")
                    st.caption(f"Kategori: {src['kind']} · Lokasi: {src['location'] or 'tidak tercantum'} · Status halaman: {src['page_status']}")
                    if src.get("doc_url"):
                        st.link_button("Buka dokumen sumber di Google Drive", src["doc_url"], key=f"history_doc_{msg.get('id','')}_{i}_{src['id']}")
                    else:
                        st.caption("Tautan dokumen tidak tersedia pada metadata untuk sumber ini.")

question = st.chat_input("Contoh: Apa yang tercatat tentang kerusakan pintu?")
if question is None and selected_example != "Pilih contoh pertanyaan...":
    if st.button("Tanyakan contoh ini"): question = selected_example

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.spinner("Memeriksa metadata dan memilih sumber yang relevan..."):
        results, qinfo = retrieve(question, records, top_k, "Keyword" if search_mode == "Kata kunci" else "TF-IDF")
        if use_llm and client and results:
            try: answer = answer_with_openai(question, results, client, model_name, qinfo)
            except Exception: answer = answer_local(question, results, qinfo)
        else: answer = answer_local(question, results, qinfo)
    sources = []
    for r in results:
        sources.append({"id": r["id"], "title": r["title"], "kind": r["kind"], "source_title": r["source_title"], "page": r["page"], "sheet": r["sheet"], "code": r["code"], "location": r["location"], "doc_url": r.get("doc_url", ""), "page_status": "perlu verifikasi" if "sekitar" in (r["page"] or "").lower() else "tercantum", "score": round(r["score"], 3)})
    with st.chat_message("assistant"):
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        if sources:
            with st.expander(f"Periksa sumber jawaban ({len(sources)})", expanded=True):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"<div class='source-box'><div class='source-title'>[S{i}] {html.escape(src['title'])}</div><div>{html.escape(src['source_title'])}</div><div class='muted'>ID: {html.escape(src['id'])} · Halaman: {html.escape(src['page'] or 'tidak tercantum')} · {html.escape(src['kind'])}</div></div>", unsafe_allow_html=True)
                    if src.get("doc_url"):
                        st.link_button("Buka dokumen sumber di Google Drive", src["doc_url"], key=f"source_doc_{len(st.session_state.messages)}_{i}_{src['id']}")
                    else:
                        st.caption("Tautan Google Drive belum tersedia pada rekod ini.")
                    if src["page_status"] == "perlu verifikasi": st.caption("Catatan: nomor halaman pada metadata bersifat perkiraan dan perlu dicek pada PDF asli.")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with st.expander("Tentang sumber dan batasan jawaban"):
    st.write("Jawaban disusun dari workbook metadata yang diunggah atau disertakan bersama aplikasi. Hasil pencarian bukan pengganti pemeriksaan dokumen asli. Jumlah yang disebutkan hanya menghitung rekod yang tersedia di workbook. Jika lokasi, halaman, atau detail kerusakan tidak tercatat, aplikasi akan menyatakannya secara terbuka.")
