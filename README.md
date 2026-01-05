# 🖼️ Pengolahan Citra Digital

Aplikasi web untuk pengolahan citra digital yang mendukung berbagai metode pemrosesan gambar seperti segmentasi, morfologi, restorasi, dan deteksi tepi.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Vue.js](https://img.shields.io/badge/Vue.js-3-brightgreen?logo=vue.js)

## 📋 Fitur

### Metode Pemrosesan Tersedia:
| Kategori | Metode |
|----------|--------|
| **Dasar** | Segmentasi (Otsu Threshold) |
| **Morfologi** | Dilasi, Erosi, Opening, Closing, Region Filling |
| **Perbaikan** | Restorasi (Denoising) |
| **Deteksi Tepi** | Sobel, Roberts Cross, Prewitt, Laplacian, Frei-Chen |

## 🛠️ Instalasi

### Prasyarat
- Python 3.8 atau lebih baru
- pip (Python package manager)
- Web browser modern (Chrome, Firefox, Edge)

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/USERNAME/ProjectPCD.git
   cd ProjectPCD
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies Python**
   ```bash
   pip install fastapi uvicorn opencv-python numpy scipy scikit-image python-multipart
   ```

   Atau jika tersedia file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Menjalankan Aplikasi

### 1. Jalankan Backend (FastAPI Server)
```bash
cd Backend
python main.py
```
Server akan berjalan di `http://127.0.0.1:8000`

### 2. Buka Frontend
Buka file `Frontend/index.html` langsung di browser, atau gunakan Live Server extension di VS Code.

## 📖 Cara Penggunaan

1. **Upload Gambar** - Klik area upload untuk memilih gambar (PNG/JPG, maks 10MB)
2. **Pilih Metode** - Pilih metode pemrosesan dari dropdown
3. **Proses** - Klik tombol "Proses Gambar"
4. **Download** - Klik tombol "Download" untuk menyimpan hasil

## 📁 Struktur Proyek

```
ProjectPCD/
├── Backend/
│   └── main.py          # FastAPI server & image processing
├── Frontend/
│   └── index.html       # Vue 3 frontend
└── README.md
```

## 🔧 API Endpoint

### POST /process
Memproses gambar dengan metode yang dipilih.

**Request:**
- `file`: File gambar (multipart/form-data)
- `method`: Metode pemrosesan (string)

**Response:**
```json
{
  "filename": "example.jpg",
  "method": "sobel",
  "image_base64": "data:image/png;base64,..."
}
```

## 📚 Dependencies

- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing
- **scikit-image** - Image processing

## 👥 Tim Pengembang

**Kelompok 2** - Praktikum Pengolahan Citra Digital

## 📄 Lisensi

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan.
