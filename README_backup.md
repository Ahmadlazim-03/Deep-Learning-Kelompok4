# Deep Learning - Image Classification dengan InceptionV3# Deep Learning Kelompok 4



Proyek ini mengimplementasikan sistem klasifikasi gambar menggunakan arsitektur **InceptionV3** dengan teknik transfer learning. Model dilatih untuk mengklasifikasikan gambar ke dalam 6 kategori: Buildings, Forest, Glacier, Mountain, Sea, dan Street.End-to-end image classification using InceptionV3 with the following setup:



## 📋 Fitur Utama- Split: Train 80%, Test 20%; Validation = 20% of Train (i.e., 16% of total)

- Augmentations (train only): horizontal flip, zoom out (0.2), shear (0.2), rotation (20°)

- **Arsitektur**: InceptionV3 (pretrained ImageNet)- Metrics: Accuracy, Precision, Recall, F1-score, ROC/AUC (OvR)

- **Transfer Learning**: Base model frozen, custom classifier layer- Timing: Training and Testing time (seconds)

- **Data Augmentation**: Horizontal flip, rotation, shear, zoom

- **Split Dataset**: 80% training, 20% testing, 20% validation (dari training)## Dataset structure

- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC

- **Web Interface**: Flask web app dengan UI modern dan dark modePlace images under `data/` with one subfolder per class:

- **Model Switching**: Support multiple models dengan runtime switching

```

## 🎯 Hasil Modeldata/

	class_a/

- **Accuracy**: 86.76%	class_b/

- **ROC AUC (Macro OvR)**: 0.9821	...

- **Training Time**: ~19 menit (1 epoch)```

- **Testing Time**: ~5 menit

## Quick start (Windows PowerShell)

## 📁 Struktur Proyek

```powershell

```python -m venv .venv

Deep_Learning_Kelompok4/.\.venv\Scripts\Activate.ps1

├── data/                          # Dataset (tidak di-push ke git)pip install -r requirement.txt

│   ├── buildings/

│   ├── forest/# Run (defaults: img 299x299, batch 32, epochs 5)

│   ├── glacier/python main.py --data-dir data --epochs 5 --batch-size 32 --img-size 299 299 --results-dir results

│   ├── mountain/

│   ├── sea/# Optional: fine-tune Inception base

│   └── street/python main.py --fine-tune --epochs 10

├── src/                           # Source code untuk training```

│   ├── data_preprocessing.py      # Data loading dan augmentasi

│   ├── model_inception.py         # Arsitektur InceptionV3Results (model, metrics, plots) are saved under `results/`.

│   ├── train_model.py             # Training pipeline

│   ├── evaluate_model.py          # Evaluasi dan metrics
│   └── utils.py                   # Utility functions
├── web/                           # Flask web application
│   ├── app.py                     # Flask server
│   ├── templates/
│   │   └── index.html             # Landing page UI
│   └── static/
│       ├── style.css              # Custom styling
│       └── app.js                 # Frontend JavaScript
├── results/                       # Hasil training (tidak di-push)
│   ├── model.h5                   # Trained model
│   ├── class_indices.json         # Mapping kelas
│   ├── confusion_matrix.png       # Confusion matrix
│   ├── roc_curves.png             # ROC curves
│   └── metrics.json               # Evaluation metrics
├── main.py                        # CLI untuk training
├── requirement.txt                # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Dokumentasi ini
```

## 🚀 Cara Menjalankan dari Awal

### 1. Prerequisites

Pastikan sistem Anda sudah terinstal:
- **Python 3.8+** (tested on Python 3.12.6)
- **pip** (Python package manager)
- **Git** (untuk clone repository)

### 2. Clone Repository

```bash
git clone https://github.com/Ahmadlazim-03/Deep-Learning-Kelompok4.git
cd Deep-Learning-Kelompok4
```

### 3. Buat Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirement.txt
```

Dependencies yang akan diinstall:
- TensorFlow 2.17.1
- Flask 3.1.2
- NumPy, Pillow, Matplotlib
- scikit-learn 1.7.2

### 5. Persiapkan Dataset

Struktur folder dataset harus seperti ini:

```
data/
├── buildings/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── forest/
│   ├── image1.jpg
│   └── ...
├── glacier/
├── mountain/
├── sea/
└── street/
```

**Catatan**: 
- Setiap subfolder adalah nama kelas
- Gambar bisa format: `.jpg`, `.jpeg`, `.png`
- Dataset akan otomatis di-split: 80% train, 20% test, 20% validation

### 6. Training Model

#### Mode Normal (Full Training)

```bash
python main.py --data-dir data --epochs 10 --batch-size 32 --img-size 299 299 --results-dir results
```

**Parameter:**
- `--data-dir`: Path ke folder dataset
- `--epochs`: Jumlah epoch (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--img-size`: Ukuran input (299x299 untuk InceptionV3)
- `--results-dir`: Folder output hasil

#### Mode Fast (Testing/Development)

Untuk testing cepat dengan data terbatas:

```bash
python main.py --data-dir data --epochs 1 --batch-size 32 --img-size 299 299 --results-dir results_fast --fast
```

Flag `--fast` akan membatasi:
- Steps per epoch: 50
- Validation steps: 20
- Test steps: 30

#### Opsi Tambahan

```bash
# Fine-tuning (unfreeze base model)
python main.py --data-dir data --epochs 5 --fine-tune

# Custom learning rate
python main.py --data-dir data --lr 0.0001

# Custom seed untuk reproducibility
python main.py --data-dir data --seed 42
```

Lihat semua opsi:
```bash
python main.py --help
```

### 7. Hasil Training

Setelah training selesai, folder `results/` akan berisi:

1. **model.h5** - Model terlatih (untuk inference)
2. **class_indices.json** - Mapping index ke nama kelas
3. **split_info.json** - Informasi dataset split
4. **confusion_matrix.png** - Confusion matrix visualization
5. **roc_curves.png** - ROC curves untuk setiap kelas
6. **classification_report.txt** - Precision, Recall, F1-Score per kelas
7. **metrics.json** - Summary metrics (Accuracy, ROC AUC, etc.)
8. **training_summary.txt** - Waktu training dan testing

### 8. Menjalankan Web Application

#### Start Flask Server

```bash
cd web
python app.py
```

Server akan berjalan di: **http://127.0.0.1:5000**

#### Atau jalankan dari root directory:

```bash
python -c "import sys; sys.path.insert(0, 'web'); import app; app.app.run(debug=True)"
```

#### Fitur Web UI:

1. **Landing Page Modern**
   - Hero section dengan ilustrasi
   - Dark mode toggle (persistent)
   - Responsive design

2. **Upload & Predict**
   - Drag & drop file upload
   - Preview gambar
   - Real-time prediction
   - Probability bars untuk setiap kelas

3. **Model Switching**
   - Switch antara multiple models
   - Model info (classes, input shape)

4. **API Endpoints**
   - `POST /api/predict` - Prediksi gambar
   - `GET /health` - Health check & diagnostics
   - `POST /api/switch_model` - Ganti model

### 9. Testing

#### Test Import Modules

```bash
python -c "from src.data_preprocessing import create_generators; from src.model_inception import build_inception_model; print('✓ All modules imported successfully')"
```

#### Test Model Build

```bash
python -c "from src.model_inception import build_inception_model; model = build_inception_model(num_classes=6); print(f'✓ Model built: {len(model.layers)} layers')"
```

#### Test Utils

```bash
python -c "from src.utils import format_seconds; print(format_seconds(3665))"
```

## 📊 Data Augmentation

Model menggunakan augmentasi berikut:

```python
ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,      # Mirror horizontal
    rotation_range=20,         # Rotasi ±20 derajat
    shear_range=0.2,           # Shear transformation
    zoom_range=(0.8, 1.0),     # Zoom out 0.8-1.0x
    validation_split=0.2       # 20% untuk validasi
)
```

## 🏗️ Arsitektur Model

```
Input (299x299x3)
    ↓
InceptionV3 Base (frozen, pretrained ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.5)
    ↓
Dense (6 units, softmax)
    ↓
Output (6 classes)
```

**Parameter:**
- Total params: ~22M
- Trainable params: ~12K (hanya classifier layer)
- Non-trainable params: ~22M (frozen base)

## 📈 Evaluation Metrics

Model dievaluasi dengan metrics berikut:

1. **Accuracy**: Persentase prediksi benar
2. **Precision**: TP / (TP + FP) per kelas
3. **Recall**: TP / (TP + FN) per kelas
4. **F1-Score**: Harmonic mean precision & recall
5. **ROC AUC**: Area Under ROC Curve (One-vs-Rest)
6. **Confusion Matrix**: Visualisasi prediksi vs aktual

## 🔧 Troubleshooting

### Error: TensorFlow not found

```bash
pip install tensorflow==2.17.1
```

### Error: Module 'src' not found

Pastikan menjalankan command dari root directory proyek.

### Error: No images found in dataset

Periksa struktur folder `data/` dan pastikan ada gambar di subfolder.

### Web UI tidak bisa diakses

1. Pastikan Flask server running
2. Buka browser: http://127.0.0.1:5000
3. Check firewall settings

### Model loading error

Pastikan file `results/model.h5` dan `results/class_indices.json` ada.

## 📝 Requirements

Lihat `requirement.txt` untuk daftar lengkap dependencies:

```
tensorflow==2.17.1
numpy==2.2.1
pillow==11.0.0
matplotlib==3.9.3
scikit-learn==1.7.2
flask==3.1.2
```

## 👥 Tim Pengembang

**Kelompok 4 - Deep Learning Project**

## 📄 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- InceptionV3 architecture by Google
- TensorFlow and Keras teams
- Bootstrap for UI components

## 📞 Contact

Untuk pertanyaan atau issues, silakan buka issue di GitHub repository.

---

**Happy Coding! 🚀**

