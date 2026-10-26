# Chatbot RAG Dokumentasi Masjid Syuhada

Aplikasi Streamlit untuk menelusuri workbook metadata Dublin Core Masjid Syuhada. Retrieval lokal menggunakan TF-IDF sehingga aplikasi tetap bisa digunakan tanpa API key. Jika `OPENAI_API_KEY` disetel, aplikasi dapat menggunakan OpenAI untuk merangkai jawaban dari rekod yang ditemukan.

## Fitur

- Chat UI berbahasa Indonesia.
- Membaca sheet metadata dokumen, elemen arsitektur, kerusakan, panduan pelestarian, chunk RAG, pemetaan DC, dan controlled vocabulary.
- Retrieval lokal TF-IDF atau keyword.
- Sitasi sumber, sheet, ID rekod, dan halaman jika tersedia.
- Unggah workbook lain dari sidebar.
- OpenAI LLM opsional. API key tidak disimpan di kode.

## Menjalankan secara lokal

Python 3.10 atau lebih baru direkomendasikan.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat lokal yang ditampilkan Streamlit.

## Mengaktifkan jawaban OpenAI (opsional)

Buat API key di akun OpenAI Anda, lalu set sebagai secret environment. Jangan masukkan key ke GitHub.

Untuk pengujian lokal, buat file `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "isi_api_key_anda"
```

Jangan commit file `secrets.toml`. File tersebut sudah dikecualikan oleh `.gitignore`.

Anda juga dapat mengatur nama model melalui environment variable `OPENAI_MODEL`. Default di aplikasi adalah `gpt-4o-mini`. Pastikan model tersebut tersedia pada akun API Anda.

Tanpa API key, aplikasi tetap dapat mencari dan menampilkan rekod relevan, tetapi jawaban berupa ringkasan hasil pencarian, bukan generasi LLM.

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload semua isi folder proyek ini, termasuk `app.py`, `requirements.txt`, `README.md`, dan folder `data/`.
3. Pastikan file Excel yang akan dipublikasikan memang boleh diakses publik. Repository publik berarti file workbook juga dapat diunduh oleh publik.
4. Masuk ke Streamlit Community Cloud dan pilih **Create app**.
5. Pilih repository, branch, dan file utama `app.py`.
6. Klik **Deploy**.
7. Jika ingin memakai OpenAI, buka pengaturan aplikasi **Secrets** dan tambahkan:

   ```toml
   OPENAI_API_KEY = "isi_api_key_anda"
   ```

8. Simpan lalu reboot aplikasi.

Nama menu dapat berubah mengikuti versi antarmuka Streamlit Community Cloud.

## Struktur proyek

```text
masjid_syuhada_rag_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── Metadata_DCMA_RAG_Masjid_Syuhada.xlsx
```

## Penting untuk publikasi

- Workbook saat ini memuat rekod contoh dan sebagian nomor halaman ditandai "sekitar". Verifikasi sumber sebelum dipublikasikan sebagai data resmi.
- Aplikasi tidak menggantikan pemeriksaan konservasi oleh tenaga ahli.
- Jangan mengunggah informasi pribadi, dokumen internal, atau data berhak cipta tanpa izin.
- TF-IDF merupakan baseline retrieval. Untuk produksi berskala besar, pertimbangkan embedding model, vector database, evaluasi retrieval, dan pengujian jawaban.
- Jika workbook diperbarui, ganti file di `data/` dan redeploy/reboot aplikasi.
