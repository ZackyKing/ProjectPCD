import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from io import BytesIO
from enum import Enum
from scipy import ndimage
from skimage import restoration, morphology

# Inisialisasi Aplikasi FastAPI
app = FastAPI(title="API Pengolahan Citra")

# Konfigurasi CORS agar frontend (browser) bisa mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dalam produksi, ganti "*" dengan URL frontend spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftar Metode Pengolahan yang Tersedia
class ImageMethod(str, Enum):
    SEGMENTATION = "segmentation"
    DILATION = "dilation"
    EROSION = "erosion"
    OPENING = "opening"
    CLOSING = "closing"
    REGION_FILLING = "region_filling"
    RESTORATION = "restoration"
    # Metode Edge Detection
    SOBEL = "sobel"
    ROBERTS = "roberts"
    PREWITT = "prewitt"
    LAPLACIAN = "laplacian"
    FREICHAN = "freichan"

def read_image(file_bytes):
    """Membaca bytes gambar ke format OpenCV"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img):
    """Mengubah gambar OpenCV kembali ke Base64 string"""
    _, buffer = cv2.imencode('.png', img)
    io_buf = BytesIO(buffer)
    b64_str = base64.b64encode(io_buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def process_segmentation(img):
    """Segmentasi menggunakan Thresholding Otsu"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_morphology(img, operation):
    """Menangani Dilasi, Erosi, Opening, Closing"""
    # Morfologi biasanya bekerja lebih baik pada Grayscale/Biner, tapi bisa juga RGB
    # Di sini kita gunakan kernel 5x5
    kernel = np.ones((5, 5), np.uint8)
    
    if operation == ImageMethod.DILATION:
        return cv2.dilate(img, kernel, iterations=1)
    elif operation == ImageMethod.EROSION:
        return cv2.erode(img, kernel, iterations=1)
    elif operation == ImageMethod.OPENING:
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == ImageMethod.CLOSING:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def process_region_filling(img):
    """Mengisi lubang pada objek (Region Filling)"""
    # Region filling butuh citra biner
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Isi lubang menggunakan scipy.ndimage
    filled = ndimage.binary_fill_holes(binary).astype(int)
    # Kembalikan ke format 0-255 uint8
    return (filled * 255).astype(np.uint8)

def process_restoration(img):
    """Restorasi citra (Denoising)"""
    # Menggunakan Fast Non-Local Means Denoising
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# ===================== EDGE DETECTION METHODS =====================

def process_sobel(img):
    """Deteksi tepi menggunakan operator Sobel"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Hitung gradien pada arah X dan Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gabungkan magnitude dari kedua gradien
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalisasi ke 0-255
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude

def process_roberts(img):
    """Deteksi tepi menggunakan operator Roberts Cross"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    
    # Kernel Roberts
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    
    # Konvolusi dengan kernel Roberts
    roberts_x = ndimage.convolve(gray, kernel_x)
    roberts_y = ndimage.convolve(gray, kernel_y)
    
    # Gabungkan magnitude
    magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
    
    # Normalisasi ke 0-255
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude

def process_prewitt(img):
    """Deteksi tepi menggunakan operator Prewitt"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    
    # Kernel Prewitt
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    
    # Konvolusi dengan kernel Prewitt
    prewitt_x = ndimage.convolve(gray, kernel_x)
    prewitt_y = ndimage.convolve(gray, kernel_y)
    
    # Gabungkan magnitude
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    # Normalisasi ke 0-255
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude

def process_laplacian(img):
    """Deteksi tepi menggunakan operator Laplacian"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Laplacian dengan ksize 3
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    
    # Ambil nilai absolut dan normalisasi
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
    return laplacian

def process_freichan(img):
    """Deteksi tepi menggunakan operator Frei-Chen (9 basis masks)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)
    
    sqrt2 = np.sqrt(2)
    
    # 9 Frei-Chen basis masks
    # Edge masks (f1-f4)
    f1 = np.array([[1, sqrt2, 1], [0, 0, 0], [-1, -sqrt2, -1]]) / (2 * sqrt2)
    f2 = np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]]) / (2 * sqrt2)
    f3 = np.array([[0, -1, sqrt2], [1, 0, -1], [-sqrt2, 1, 0]]) / (2 * sqrt2)
    f4 = np.array([[sqrt2, -1, 0], [-1, 0, 1], [0, 1, -sqrt2]]) / (2 * sqrt2)
    
    # Line masks (f5-f8)
    f5 = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]]) / 2
    f6 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / 2
    f7 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 6
    f8 = np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]) / 6
    
    # Average mask (f9)
    f9 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 3
    
    # Konvolusi dengan edge masks saja (f1-f4) untuk deteksi tepi
    edge_masks = [f1, f2, f3, f4]
    
    result = np.zeros_like(gray)
    for mask in edge_masks:
        convolved = ndimage.convolve(gray, mask)
        result += convolved**2
    
    # Ambil akar kuadrat dari jumlah kuadrat
    result = np.sqrt(result)
    
    # Normalisasi ke 0-255
    result = (result / result.max() * 255).astype(np.uint8) if result.max() > 0 else result.astype(np.uint8)
    return result

# ==================================================================

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    method: ImageMethod = Form(...)
):
    try:
        # 1. Baca File
        contents = await file.read()
        img = read_image(contents)
        
        if img is None:
            raise HTTPException(status_code=400, detail="File gambar tidak valid")

        # 2. Proses Berdasarkan Metode
        result_img = None
        
        if method == ImageMethod.SEGMENTATION:
            result_img = process_segmentation(img)
            
        elif method in [ImageMethod.DILATION, ImageMethod.EROSION, ImageMethod.OPENING, ImageMethod.CLOSING]:
            result_img = process_morphology(img, method)
            
        elif method == ImageMethod.REGION_FILLING:
            result_img = process_region_filling(img)
            
        elif method == ImageMethod.RESTORATION:
            result_img = process_restoration(img)
        
        # Edge Detection Methods
        elif method == ImageMethod.SOBEL:
            result_img = process_sobel(img)
            
        elif method == ImageMethod.ROBERTS:
            result_img = process_roberts(img)
            
        elif method == ImageMethod.PREWITT:
            result_img = process_prewitt(img)
            
        elif method == ImageMethod.LAPLACIAN:
            result_img = process_laplacian(img)
            
        elif method == ImageMethod.FREICHAN:
            result_img = process_freichan(img)
            
        else:
            # Fallback jika metode tidak dikenali, kembalikan asli
            result_img = img

        # 3. Kembalikan Hasil
        result_b64 = image_to_base64(result_img)
        return JSONResponse(content={
            "filename": file.filename,
            "method": method,
            "image_base64": result_b64
        })

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Jalankan server development
    uvicorn.run(app, host="127.0.0.1", port=8000)