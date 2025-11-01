"""
File: web/app.py
Tujuan: Flask web server untuk inference model InceptionV3

Fitur:
1. Landing page dengan upload gambar
2. API endpoint /api/predict untuk prediksi gambar
3. API endpoint /health untuk memeriksa status model
4. API endpoint /api/switch_model untuk mengganti model secara runtime
5. Dark mode toggle dan drag-drop upload

Endpoints:
- GET /: Halaman utama (landing page)
- POST /api/predict: Prediksi gambar yang diupload
- GET /health: Status model dan class indices
- POST /api/switch_model: Ganti model ke folder results lain
"""

import os
import io
import json
from typing import Dict

from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

# === KONFIGURASI GLOBAL ===
# Path ke folder results (mutable agar bisa diganti runtime)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR", 
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
)
# Path ke file model .h5
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(RESULTS_DIR, "inceptionv3_model.h5"))
# Path ke file class_indices.json
CLASS_INDICES_PATH = os.environ.get("CLASS_INDICES_PATH", os.path.join(RESULTS_DIR, "class_indices.json"))
# Ukuran input gambar (299x299 untuk InceptionV3)
IMG_SIZE = (299, 299)

# Inisialisasi Flask app
app = Flask(__name__)

# === CACHE GLOBAL ===
# Model dan class indices di-cache agar tidak perlu load berkali-kali
_model = None  # Model Keras yang sudah ditraining
_class_indices: Dict[str, int] | None = None  # Mapping label ke index
_idx_to_class: Dict[int, str] | None = None  # Mapping index ke label (kebalikan)


def load_artifacts():
    """
    Load model dan class indices dari disk (lazy loading)
    
    Fungsi ini hanya load jika belum di-cache, sehingga efisien.
    Model dan class indices disimpan di variabel global untuk digunakan
    di semua request tanpa perlu load ulang.
    
    Raises:
        FileNotFoundError: Jika model atau class_indices.json tidak ditemukan
    """
    global _model, _class_indices, _idx_to_class
    
    # Load model jika belum di-cache
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
        # Load model Keras dari file .h5
        _model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class indices jika belum di-cache
    if _class_indices is None:
        if not os.path.exists(CLASS_INDICES_PATH):
            raise FileNotFoundError(
                f"Class indices not found at {CLASS_INDICES_PATH}. "
                "Ensure training saved class_indices.json."
            )
        # Load mapping label ke index dari file JSON
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            _class_indices = json.load(f)
        # Buat mapping kebalikan: index ke label
        # Contoh: {0: 'buildings', 1: 'forest', ...}
        _idx_to_class = {v: k for k, v in _class_indices.items()}


def set_results_dir(new_results_dir: str):
    """
    Update path global dan reset cache untuk load model baru
    
    Fungsi ini berguna untuk switch model secara runtime tanpa restart server.
    Setelah path diupdate, model dan class indices akan di-load ulang
    saat request berikutnya.
    
    Args:
        new_results_dir: Path ke folder results baru yang berisi model dan class_indices.json
    """
    global RESULTS_DIR, MODEL_PATH, CLASS_INDICES_PATH, _model, _class_indices, _idx_to_class
    
    # Update path global
    RESULTS_DIR = os.path.abspath(new_results_dir)
    MODEL_PATH = os.path.join(RESULTS_DIR, "inceptionv3_model.h5")
    CLASS_INDICES_PATH = os.path.join(RESULTS_DIR, "class_indices.json")
    
    # Reset cache agar model dan class indices di-load ulang
    _model = None
    _class_indices = None
    _idx_to_class = None


def prepare_image(file_bytes: bytes) -> np.ndarray:
    """
    Preprocessing gambar sebelum prediksi
    
    Langkah:
    1. Load gambar dari bytes
    2. Convert ke RGB (3 channels)
    3. Resize ke 299x299 (ukuran input InceptionV3)
    4. Convert ke numpy array
    5. Expand dimensions untuk batch (1, 299, 299, 3)
    6. Preprocess sesuai InceptionV3 (normalisasi ke range [-1, 1])
    
    Args:
        file_bytes: Bytes dari file gambar yang diupload
        
    Returns:
        Numpy array dengan shape (1, 299, 299, 3) siap untuk prediksi
    """
    # Load gambar dari bytes menggunakan PIL
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # Resize ke ukuran input InceptionV3
    img = img.resize(IMG_SIZE)
    
    # Convert ke numpy array dengan tipe float32
    arr = np.array(img).astype("float32")
    
    # Expand dimensions: (299, 299, 3) -> (1, 299, 299, 3)
    # Model expect input batch, jadi perlu dimensi batch
    arr = np.expand_dims(arr, axis=0)
    
    # Preprocess sesuai InceptionV3 (normalisasi)
    # Mengubah pixel values dari [0, 255] ke [-1, 1]
    arr = preprocess_input(arr)
    
    return arr


@app.route("/")
def index():
    """
    Endpoint untuk halaman utama (landing page)
    
    Render template HTML dengan informasi:
    - Daftar kelas yang tersedia
    - Path model yang sedang digunakan
    - Path folder results
    
    Returns:
        HTML rendered dari template index.html
    """
    try:
        # Load model dan class indices
        load_artifacts()
        # Dapatkan daftar kelas yang sudah diurutkan berdasarkan index
        classes = [cls for cls, _ in sorted(_class_indices.items(), key=lambda x: x[1])]
    except Exception as e:
        # Jika gagal load, kirim list kosong
        classes = []
    
    # Render template dengan data
    return render_template(
        "index.html", 
        classes=classes, 
        model_path=MODEL_PATH, 
        results_dir=RESULTS_DIR
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Endpoint untuk prediksi gambar
    
    Method: POST
    Input: Form data dengan field 'file' berisi gambar
    
    Output JSON:
    {
        "pred_class": "buildings",  # Kelas dengan probabilitas tertinggi
        "pred_index": 0,  # Index kelas dalam array
        "confidence": 0.95,  # Probabilitas kelas tertinggi
        "probs": {  # Probabilitas semua kelas
            "buildings": 0.95,
            "forest": 0.02,
            ...
        }
    }
    
    Returns:
        JSON response dengan prediksi atau error
    """
    # Load model dan class indices
    try:
        load_artifacts()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Validasi: pastikan ada file yang diupload
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (use form field 'file')."}), 400

    # Ambil file dari request
    file = request.files["file"]
    
    try:
        # Preprocessing gambar
        arr = prepare_image(file.read())
        
        # Prediksi menggunakan model
        # Output: array probabilitas dengan shape (1, num_classes)
        # verbose=0: Jangan tampilkan progress bar
        probs = _model.predict(arr, verbose=0)[0]
        
        # Ambil index dengan probabilitas tertinggi
        top_idx = int(np.argmax(probs))
        
        # Buat response JSON
        result = {
            "pred_class": _idx_to_class.get(top_idx, str(top_idx)),  # Nama kelas
            "pred_index": top_idx,  # Index kelas
            "confidence": float(probs[top_idx]),  # Confidence score
            "probs": {  # Probabilitas semua kelas
                _idx_to_class.get(i, str(i)): float(p) 
                for i, p in enumerate(probs)
            },
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """
    Endpoint untuk health check - memeriksa status model dan class indices
    
    Method: GET
    
    Output JSON:
    {
        "results_dir": "C:/path/to/results",
        "model_path": "C:/path/to/results/inceptionv3_model.h5",
        "class_indices_path": "C:/path/to/results/class_indices.json",
        "exists_model": true,  # Apakah file model ada?
        "exists_class_indices": true,  # Apakah file class_indices.json ada?
        "classes": ["buildings", "forest", ...]  # Daftar kelas (jika file ada)
    }
    
    Returns:
        JSON response dengan informasi status
    """
    # Cek apakah file model ada
    exists_model = os.path.exists(MODEL_PATH)
    # Cek apakah file class indices ada
    exists_classes = os.path.exists(CLASS_INDICES_PATH)
    
    # Buat response payload
    payload = {
        "results_dir": RESULTS_DIR,
        "model_path": MODEL_PATH,
        "class_indices_path": CLASS_INDICES_PATH,
        "exists_model": exists_model,
        "exists_class_indices": exists_classes,
    }
    
    # Jika class indices ada, load dan tampilkan daftar kelas
    if exists_classes:
        try:
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                cls = json.load(f)
            # Sort berdasarkan index dan ambil nama kelasnya
            payload["classes"] = [k for k, _ in sorted(cls.items(), key=lambda x: x[1])]
        except Exception as e:
            # Jika gagal load, tampilkan error
            payload["classes_error"] = str(e)
    
    return jsonify(payload), 200


@app.post("/api/switch_model")
def api_switch_model():
    """
    Endpoint untuk mengganti model secara runtime
    
    Method: POST
    Input JSON:
    {
        "results_dir": "C:/path/to/new/results"
    }
    
    Fungsi ini memungkinkan user untuk switch model tanpa restart server.
    Berguna jika ada beberapa trained model dan ingin mencoba yang berbeda.
    
    Validasi:
    - Directory harus ada
    - Harus ada file inceptionv3_model.h5
    - Harus ada file class_indices.json
    
    Output JSON:
    {
        "results_dir": "C:/path/to/new/results",
        "model_path": "C:/path/to/new/results/inceptionv3_model.h5",
        "class_indices_path": "C:/path/to/new/results/class_indices.json",
        "classes": ["buildings", "forest", ...]
    }
    
    Returns:
        JSON response dengan informasi model baru atau error
    """
    # Parse JSON request body
    data = request.get_json(silent=True) or {}
    new_dir = data.get("results_dir")
    
    # Validasi: results_dir harus ada
    if not new_dir:
        return jsonify({"error": "Missing 'results_dir'"}), 400

    # Convert ke absolute path
    abs_dir = os.path.abspath(new_dir)
    model_path = os.path.join(abs_dir, "inceptionv3_model.h5")
    classes_path = os.path.join(abs_dir, "class_indices.json")
    
    # Validasi: directory harus ada
    if not os.path.exists(abs_dir):
        return jsonify({"error": f"Directory not found: {abs_dir}"}), 400
    
    # Validasi: file model harus ada
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model not found at {model_path}"}), 400
    
    # Validasi: file class_indices.json harus ada
    if not os.path.exists(classes_path):
        return jsonify({"error": f"class_indices.json not found at {classes_path}"}), 400

    # Update path global dan reset cache
    set_results_dir(abs_dir)
    
    # Load model dan class indices baru
    try:
        load_artifacts()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return informasi model baru
    return jsonify({
        "results_dir": RESULTS_DIR,
        "model_path": MODEL_PATH,
        "class_indices_path": CLASS_INDICES_PATH,
        "classes": [k for k, _ in sorted(_class_indices.items(), key=lambda x: x[1])],
    })


if __name__ == "__main__":
    """
    Entry point untuk menjalankan Flask development server
    
    Konfigurasi:
    - host="0.0.0.0": Listen di semua network interfaces
    - port=5000: Default port (bisa diubah via environment variable PORT)
    - debug=True: Enable debug mode (auto-reload saat code berubah)
    
    Catatan:
    Ini hanya untuk development. Untuk production, gunakan WSGI server
    seperti Gunicorn atau uWSGI.
    """
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
