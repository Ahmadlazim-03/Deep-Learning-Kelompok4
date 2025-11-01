"""
File: data_preprocessing.py
Tujuan: Melakukan preprocessing data gambar dan membuat data generators untuk training

Fungsi utama:
1. Membaca struktur folder dataset
2. Membagi data menjadi train/validation/test dengan stratified split
3. Menerapkan augmentasi data pada training set
4. Membuat ImageDataGenerator untuk batch processing
"""

import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input


def _build_dataframe(data_dir: str) -> pd.DataFrame:
	"""
	Membangun DataFrame yang berisi path file gambar dan labelnya
	
	Fungsi ini men-scan semua subfolder di dalam data_dir dan mengumpulkan
	informasi file gambar beserta label kelasnya.
	
	Struktur folder yang diharapkan:
		data_dir/
			buildings/
				img1.jpg
				img2.png
			forest/
				img3.jpg
			...
	
	Args:
		data_dir: Path ke folder dataset utama
		
	Returns:
		DataFrame dengan kolom 'file_path' dan 'label'
		
	Raises:
		RuntimeError: Jika tidak ada gambar ditemukan dalam folder
	"""
	records = []
	
	# Dapatkan daftar semua folder kelas yang sudah diurutkan
	classes = sorted([
		d for d in os.listdir(data_dir)
		if os.path.isdir(os.path.join(data_dir, d))
	])
	
	# Loop untuk setiap kelas
	for cls in classes:
		class_dir = os.path.join(data_dir, cls)
		# Walk melalui semua subfolder (jika ada)
		for root, _, files in os.walk(class_dir):
			# Loop untuk setiap file
			for f in files:
				# Cek apakah file adalah gambar berdasarkan ekstensinya
				if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
					records.append({
						"file_path": os.path.join(root, f),  # Path lengkap file
						"label": cls,  # Nama kelas
					})

	# Validasi: pastikan minimal ada satu gambar
	if not records:
		raise RuntimeError(f"No images found under {data_dir}")

	# Konversi list of dict menjadi DataFrame
	return pd.DataFrame.from_records(records)


def create_generators(
	data_dir: str,
	img_size: Tuple[int, int] = (299, 299),
	batch_size: int = 32,
	seed: int = 42,
) -> Tuple:
	"""
	Membuat train/validation/test generators dengan split dan augmentasi yang diperlukan
	
	Pembagian Data (Stratified Split):
	  - Train: 80% dari total data
	  - Test: 20% dari total data
	  - Validation: 20% dari Train (setara dengan 16% dari total)
	  
	Augmentasi Data (hanya untuk training):
	  - Horizontal flip: Membalik gambar secara horizontal (kiri-kanan)
	  - Rotation (20°): Memutar gambar hingga 20 derajat
	  - Shear (0.2): Transformasi shear/memiringkan gambar
	  - Zoom out (0.2): Hanya zoom out 20%, range (0.8-1.0)
	
	Args:
		data_dir: Path ke folder dataset
		img_size: Ukuran target gambar (tinggi, lebar) dalam pixel
		batch_size: Jumlah gambar per batch untuk training
		seed: Random seed untuk reproducibility
		
	Returns:
		Tuple berisi:
		- train_gen: Generator untuk data training (dengan augmentasi)
		- val_gen: Generator untuk data validation (tanpa augmentasi)
		- test_gen: Generator untuk data testing (tanpa augmentasi)
		- class_indices: Dictionary mapping label ke index (contoh: {'buildings': 0, 'forest': 1, ...})
	"""
	# Buat DataFrame dari struktur folder dataset
	df = _build_dataframe(data_dir)

	# === LANGKAH 1: Split Train+Val vs Test (80% vs 20%) ===
	# Menggunakan stratified split agar proporsi kelas seimbang
	train_val_df, test_df = train_test_split(
		df,
		test_size=0.2,  # 20% untuk test
		stratify=df["label"],  # Pastikan setiap kelas proporsional
		random_state=seed,  # Untuk reproducibility
	)

	# === LANGKAH 2: Split Train vs Validation dari Train+Val ===
	# Validation = 20% dari Train+Val (jadi 16% dari total dataset)
	train_df, val_df = train_test_split(
		train_val_df,
		test_size=0.2,  # 20% dari Train+Val untuk validation
		stratify=train_val_df["label"],  # Proporsi kelas tetap seimbang
		random_state=seed,
	)

	# === LANGKAH 3: Definisikan ImageDataGenerator ===
	
	# Generator untuk TRAINING dengan AUGMENTASI
	train_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,  # Normalisasi sesuai InceptionV3
		horizontal_flip=True,  # Flip horizontal secara acak
		rotation_range=20,  # Rotasi acak ±20 derajat
		shear_range=0.2,  # Shear transformation dengan intensitas 0.2
		zoom_range=(0.8, 1.0),  # Zoom out 20%: 0.8 = 80% size, 1.0 = 100% size
		fill_mode="nearest",  # Cara mengisi pixel kosong setelah transformasi
	)

	# Generator untuk VALIDATION dan TEST tanpa augmentasi
	# Hanya melakukan preprocessing (normalisasi)
	test_val_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input
	)

	# === LANGKAH 4: Buat Class Indices yang Konsisten ===
	# Urutkan nama kelas agar mapping index konsisten di semua generator
	classes_sorted = sorted(df["label"].unique().tolist())
	class_mode = "categorical"  # Output berupa one-hot encoding

	# === LANGKAH 5: Buat Training Generator ===
	train_gen = train_datagen.flow_from_dataframe(
		train_df,
		x_col="file_path",  # Kolom yang berisi path file gambar
		y_col="label",  # Kolom yang berisi label kelas
		target_size=img_size,  # Resize semua gambar ke ukuran ini
		color_mode="rgb",  # Gambar berwarna (3 channel)
		class_mode=class_mode,  # Categorical untuk klasifikasi multi-class
		classes=classes_sorted,  # Urutan kelas yang konsisten
		batch_size=batch_size,  # Jumlah gambar per batch
		shuffle=True,  # Acak urutan data setiap epoch
		seed=seed,
	)

	# === LANGKAH 6: Buat Validation Generator ===
	val_gen = test_val_datagen.flow_from_dataframe(
		val_df,
		x_col="file_path",
		y_col="label",
		target_size=img_size,
		color_mode="rgb",
		class_mode=class_mode,
		classes=classes_sorted,  # Urutan kelas sama dengan training
		batch_size=batch_size,
		shuffle=True,  # Boleh diacak untuk validation
		seed=seed,
	)

	# === LANGKAH 7: Buat Test Generator ===
	test_gen = test_val_datagen.flow_from_dataframe(
		test_df,
		x_col="file_path",
		y_col="label",
		target_size=img_size,
		color_mode="rgb",
		class_mode=class_mode,
		classes=classes_sorted,  # Urutan kelas sama
		batch_size=batch_size,
		shuffle=False,  # PENTING: Jangan diacak agar evaluasi akurat
	)

	# === LANGKAH 8: Buat Dictionary Class Indices ===
	# Mapping dari nama kelas ke index numerik
	# Contoh: {'buildings': 0, 'forest': 1, 'glacier': 2, ...}
	class_indices: Dict[str, int] = {cls: i for i, cls in enumerate(classes_sorted)}

	# Return semua generator dan class indices
	return train_gen, val_gen, test_gen, class_indices

