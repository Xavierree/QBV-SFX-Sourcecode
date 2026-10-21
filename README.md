# Q-SFX — Source Code

Q-SFX (Query Sound Effect) adalah sistem pencarian efek suara berbasis imitasi vokal. Pengguna merekam imitasi suara
(dengungan, siulan, atau ucapan) lalu sistem mengembalikan klip yang paling relevan dari galeri 9.708 klip sound-effect.
Dokumen ini hanya mencakup **source code**; model, galeri audio, dan indeks tidak disertakan (lihat §4).

Sistem retrieval yang dilayani adalah **D1**, sistem akhir pada skripsi: ensemble kembar MATPAC++ dan kembar BEATs (masing-masing satu menara imitasi dan satu menara referensi, seed 123; empat jaringan, 351,5 juta parameter),
digabung dengan kepala soft-DTW (HuBERT-base beku lapisan 9 dan dua FrameNet) dan CSLS. Untuk efek suara, hasilnya diteruskan ke gerbang ucapan Audio Flamingo 3 dan, hanya untuk kueri yang berisi ucapan, ke jalur transkrip Whisper
(digabung dengan RRF); Whisper adalah jalan pintas leksikal untuk separuh pustaka yang berisi ucapan, bukan pencocokan deskripsi. D1 **zero-shot** pada efek suara: tidak ada bobotnya yang dilatih atau diadaptasi pada rekaman efek suara
maupun qvim-dev. Angka penelitiannya (MRR, 84 kueri efek suara sebagai set pengembangan, selang kepercayaan sekitar ±0,04): menara 0,3302, kepala + CSLS 0,5007, pipa lengkap dengan gerbang 0,5221; qvim-dev (979 kueri) 0,2884.
Angka itu adalah taksiran pada set pengembangan, bukan hasil uji tersegel. Penjelasan, angka, dan hasil verifikasi: `backend/D1_SFX_PIPELINE.md`.

Seluruh sistem berjalan di **satu komputer** dan saling terhubung melalui `localhost`:

```
Browser ──► Frontend Next.js  (http://localhost:3000)
                │   byte WAV mentah
                ▼
            Backend FastAPI   (http://127.0.0.1:8500)  ← GPU: model retrieval, speech gate, Whisper, captioning
                                                          
Frontend ──(teks saja)──► OmniRoute (http://localhost:20128/v1)  ← hanya untuk tahap Metadata pada halaman Submit
```

## 1. Isi repositori

```
QBV-Sourcecode/
├── README.md                  dokumen ini
├── requirements.txt           dependensi Python backend
├── start-all.sh / .ps1        menjalankan backend lalu frontend sekaligus (Linux/macOS / Windows PowerShell)
├── package.json               workspace Node.js (frontend)
├── backend/                   layanan GPU (Python / FastAPI)
│   ├── server/main.py         API: /health /search /ingest /audio
│   ├── sfxdeploy/             pipeline: D1 (d1.py, d1_towers.py, softdtw_head.py), speech gate AF3, rute transkrip Whisper (RRF), ingest
│   ├── vendor/                softdtw.py dan kode encoder BEATs (unilm/beats, lisensi MIT)
│   ├── scripts/               run_server / stop_server / setup_env (.sh dan .ps1), check_assets.py, build_d1_index.py, build_qdev_d1_index.py, verify_d1_search.py, download_alm.py
│   └── D1_SFX_PIPELINE.md     catatan operator sistem D1
├── frontend/                  antarmuka web (Next.js, App Router)
│   ├── app/                   halaman /search /submit /admin dan rute API /api/*
│   ├── lib/                   gate passcode, penyimpanan submission, tahap metadata OmniRoute
│   └── .env.local.example     contoh konfigurasi
└── deploy/                    unit systemd: sfx-backend.service, sfx-frontend.service
```

## 2. Kebutuhan

| Komponen | Kebutuhan |
|---|---|
| GPU | NVIDIA dengan CUDA dan VRAM ≥ 12 GB (produksi: RTX 3080 Ti 12 GB; terpakai ±10,8 GiB, puncak ±10,4 GiB teralokasi saat Submit) |
| Backend | Python 3, driver NVIDIA dengan CUDA 13, paket pada `requirements.txt` |
| Frontend | Node.js 18+ dan npm |
| OmniRoute | Hanya untuk tahap Metadata pada halaman Submit (lihat manual, Bab 2.4); pencarian tidak memerlukannya |
| Browser | Browser modern dengan izin mikrofon |

Audio Flamingo 3 berlisensi NVIDIA OneWay Noncommercial: hanya untuk riset / non-komersial (termasuk demo skripsi).

## 3. Instalasi dan menjalankan

Cara tercepat (setelah `backend/scripts/setup_env.sh` dan aset pada §4 tersedia): `./start-all.sh` (Linux/macOS) atau
`.\start-all.ps1` (Windows PowerShell). Skrip ini menjalankan backend di latar belakang, memasang paket Node dan
membangun frontend bila belum ada, lalu menjalankan frontend di http://localhost:3000. Skrip `.ps1` belum diuji di Windows.

Jika model/data belum ada, backend **tidak dijalankan**: `scripts/run_server.sh` (dan `python3 server/main.py`) mencetak
daftar file yang hilang dan berhenti (`python3 backend/scripts/check_assets.py` menampilkan daftar yang sama).
Frontend tetap dapat berjalan, tetapi titik status di /admin menampilkan *down* dan pencarian mengembalikan error 502/503.

### 3.1 Backend

```bash
cd backend
scripts/setup_env.sh          # membuat backend/.venv dan memasang ../requirements.txt
# letakkan model, data, dan indeks (lihat §4), lalu:
scripts/run_server.sh         # idempoten; log di backend/logs/server.log
curl http://127.0.0.1:8500/health      # {"status":"ok","models_loaded":true,...}
scripts/stop_server.sh        # mematikan backend dan membebaskan VRAM
```

Backend hanya mendengarkan `127.0.0.1:8500` (tidak dapat diakses dari luar komputer). Port dapat diganti dengan variabel `SFXDEPLOY_PORT` (mis. `SFXDEPLOY_PORT=8501`); `start-all` lalu mengarahkan frontend ke port itu, atau atur `INFERENCE_URL` sendiri. `run_server.sh` menolak berjalan bila port tersebut sudah dijawab oleh backend lain (misalnya salinan lain dari proyek ini), sehingga tidak ada frontend yang tanpa sengaja tersambung ke backend yang salah. Startup memuat semua model ke VRAM
(±14 detik bila cache disk hangat, lebih lama pada start pertama).

Agar backend hidup otomatis saat komputer menyala: sunting `deploy/sfx-backend.service` (User dan path), lalu
`sudo cp deploy/sfx-backend.service /etc/systemd/system/ && sudo systemctl enable --now sfx-backend`.

### 3.2 Frontend

```bash
npm install                                   # dari root repositori
cp frontend/.env.local.example frontend/.env.local   # lalu isi sesuai kebutuhan
cd frontend && npm run build && npm run start        # http://localhost:3000
```

Variabel `frontend/.env.local`:

| Variabel | Fungsi |
|---|---|
| `INFERENCE_URL` | Alamat backend; bawaan `http://127.0.0.1:8500` |
| `QBV_PASSCODE`, `QBV_GATE_SECRET` | Opsional. Jika `QBV_PASSCODE` kosong, aplikasi terbuka tanpa login. Jika diisi, `QBV_GATE_SECRET` wajib (`openssl rand -hex 32`) |
| `OMNIROUTE_BASE_URL`, `OMNIROUTE_API_KEY`, `OMNIROUTE_MODEL` | Tahap Metadata (judul, tag, deskripsi) pada halaman Submit. Bila kosong, klip tetap tersimpan dan dapat dicari, hanya tahap metadata yang dilewati |
| `BACKEND_DIR` | Opsional. Lokasi folder `backend` bila bukan sejajar dengan `frontend` (dipakai tombol Start backend) |

File `.env.local` berisi rahasia dan tidak boleh dimasukkan ke repositori. Untuk dijalankan otomatis saat boot, pakai
`deploy/sfx-frontend.service`.

### 3.3 Halaman

| URL | Halaman |
|---|---|
| `http://localhost:3000/search` | Pencarian (rekam imitasi, pilih galeri SFX Library / qvim-dev) |
| `http://localhost:3000/submit` | Menambahkan klip baru ke galeri SFX |
| `http://localhost:3000/admin` | Status backend, tombol Start backend, daftar submission dan event |

## 4. Aset yang tidak disertakan

Ukuran repositori sengaja kecil (< 500 MB). Letakkan aset berikut di dalam `backend/` agar server dapat berjalan
(jalur ditentukan di `backend/sfxdeploy/config.py`):

| Jalur | Isi |
|---|---|
| `models/d1/` | D1: empat menara (`matpac_mimic.pt`, `matpac_ref.pt`, `beats_mimic.pt`, `beats_ref.pt`: kembar M1 dan S3 seed 123 pada skripsi), kepala soft-DTW (`head_clean_s42.pt`), `beats_cfg.json`, dan `config.json` (checksum bobot yang diperiksa saat memuat). Untuk membangun indeks dan memverifikasi juga diperlukan `reference_tables.npz` dan `reference_speech_gate.npz` (ekspor riset; bukan untuk menjalankan server) |
| `models/asr/whisper-large-v3-turbo/` | Whisper large-v3-turbo |
| `models/alm/af3/` | Audio Flamingo 3 (speech gate dan caption); unduh dengan `scripts/download_alm.py` |
| `models/text/` | Cache TF-IDF transkrip galeri (dibuat otomatis bila belum ada) |
| `hf-cache/` | Cache Hugging Face: `facebook/hubert-base-ls960` |
| `data/sfx/gallery/` | 9.708+ klip galeri SFX |
| `data/qvim-dev/` | Galeri qvim-dev: `DEV Dataset.csv`, `Items/` (121 klip) dan `Queries/` (979 imitasi, dipakai untuk membangun indeks CSLS) |
| `data/sfx/metadata/` | Transkrip Whisper galeri (`gallery_transcripts.json`) |
| `data/indexes/` | Indeks D1: `sfx_gallery_d1.npz` dan `qdev_d1.npz` beserta file `.meta.json`-nya, dibangun dengan `scripts/build_d1_index.py` dan `scripts/build_qdev_d1_index.py` (masing-masing menolak menulis indeks bila hasilnya tidak mereproduksi ekspor riset). Klip yang ditambahkan lewat Submit disimpan pada `sfx_gallery_d1_submitted.npz` |

Setelah mengganti model atau indeks, jalankan pemeriksaan kesetiaan sebelum melayani pengguna:

```bash
backend/.venv/bin/python backend/scripts/verify_d1_search.py <direktori_scratch>   # harus mencetak GATE: PASS
```

## 5. API

Backend (port 8500): `GET /health`, `POST /search?gallery=sfx|qdev&k=10` (body: byte WAV mentah),
`POST /ingest?gallery=sfx` (body: byte WAV mentah, header opsional `X-Filename`), `GET /audio/{gallery}/{path}`.

Frontend (port 3000): `POST /api/search`, `GET /api/health`, `GET /api/audio/[gallery]/[...path]`,
`POST /api/submit`, `GET /api/submit/[id]`, `POST /api/admin/backend/start`, `GET /api/admin/{stats,submissions,events}`,
`POST /api/qbv-login`.

## 6. Catatan

- Data submission dan event disimpan di `frontend/data/db.json` (dibuat otomatis saat pertama kali ada submission).
- Perubahan kode frontend memerlukan `npm run build` ulang sebelum `npm run start`.
- Halaman Search merekam maksimal 10 detik dan berhenti sendiri setelah pengguna bersuara lalu diam 1,5 detik (`frontend/lib/endpointing.ts`; tombol stop manual tetap ada). Ambang batasnya heuristik, bukan hasil pelatihan; halaman Submit tidak memakai berhenti-otomatis. Uji logikanya: `node frontend/scripts/test-endpointing.mjs`.
- Semua inferensi model berjalan lokal; hanya teks (caption, transkrip, name hint) yang dikirim ke OmniRoute pada tahap Metadata. Teks metadata itu tidak ikut pemeringkatan dan kualitasnya tidak dievaluasi.
- Angka yang dicetak skrip verifikasi untuk 84 kueri benchmark lebih tinggi daripada angka penelitian, karena istilah CSLS pada indeks melewatkan target tiap kueri itu sendiri; itu angka pemeriksaan jalur, bukan akurasi. Akurasi yang dapat dikutip adalah angka penelitian di atas (set pengembangan; konfirmasinya memerlukan rekaman baru).
