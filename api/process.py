from http.server import BaseHTTPRequestHandler
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from scipy import ndimage
import cgi

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_morphology(img, operation):
    kernel = np.ones((5, 5), np.uint8)
    if operation == "dilation":
        return cv2.dilate(img, kernel, iterations=1)
    elif operation == "erosion":
        return cv2.erode(img, kernel, iterations=1)
    elif operation == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def process_region_filling(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    filled = ndimage.binary_fill_holes(binary).astype(int)
    return (filled * 255).astype(np.uint8)

def process_restoration(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def process_sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def process_roberts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    roberts_x = ndimage.convolve(gray, kernel_x)
    roberts_y = ndimage.convolve(gray, kernel_y)
    magnitude = np.sqrt(roberts_x**2 + roberts_y**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def process_prewitt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    prewitt_x = ndimage.convolve(gray, kernel_x)
    prewitt_y = ndimage.convolve(gray, kernel_y)
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def process_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))

def process_freichan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    sqrt2 = np.sqrt(2)
    f1 = np.array([[1, sqrt2, 1], [0, 0, 0], [-1, -sqrt2, -1]]) / (2 * sqrt2)
    f2 = np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]]) / (2 * sqrt2)
    f3 = np.array([[0, -1, sqrt2], [1, 0, -1], [-sqrt2, 1, 0]]) / (2 * sqrt2)
    f4 = np.array([[sqrt2, -1, 0], [-1, 0, 1], [0, 1, -sqrt2]]) / (2 * sqrt2)
    
    result = np.zeros_like(gray)
    for mask in [f1, f2, f3, f4]:
        convolved = ndimage.convolve(gray, mask)
        result += convolved**2
    result = np.sqrt(result)
    return (result / result.max() * 255).astype(np.uint8) if result.max() > 0 else result.astype(np.uint8)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        
        # Parse multipart form data
        ctype, pdict = cgi.parse_header(content_type)
        pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
        
        content_length = int(self.headers.get('Content-Length'))
        body = self.rfile.read(content_length)
        
        fields = cgi.parse_multipart(BytesIO(body), pdict)
        
        file_data = fields.get('file')[0]
        method = fields.get('method')[0]
        if isinstance(method, bytes):
            method = method.decode('utf-8')
        
        img = read_image(file_data)
        
        if img is None:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid image"}).encode())
            return
        
        # Process based on method
        if method == "segmentation":
            result_img = process_segmentation(img)
        elif method in ["dilation", "erosion", "opening", "closing"]:
            result_img = process_morphology(img, method)
        elif method == "region_filling":
            result_img = process_region_filling(img)
        elif method == "restoration":
            result_img = process_restoration(img)
        elif method == "sobel":
            result_img = process_sobel(img)
        elif method == "roberts":
            result_img = process_roberts(img)
        elif method == "prewitt":
            result_img = process_prewitt(img)
        elif method == "laplacian":
            result_img = process_laplacian(img)
        elif method == "freichan":
            result_img = process_freichan(img)
        else:
            result_img = img
        
        result_b64 = image_to_base64(result_img)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "method": method,
            "image_base64": result_b64
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
