"""
File: train_model.py
Tujuan: Melakukan proses training model InceptionV3 dan menyimpan hasil training

Proses:
1. Membuat data generators (train/validation/test)
2. Membangun model InceptionV3
3. Melakukan training dengan mencatat waktu
4. Menyimpan model, class indices, dan informasi split dataset
"""

import os
import time
from typing import Tuple, Dict

from .data_preprocessing import create_generators
from .model_inception import build_inception_model


def train(
	data_dir: str = "data",
	img_size: Tuple[int, int] = (299, 299),
	batch_size: int = 32,
	epochs: int = 5,
	learning_rate: float = 1e-4,
	base_trainable: bool = False,
	results_dir: str = "results",
	seed: int = 42,
	steps_per_epoch: int | None = None,
	val_steps: int | None = None,
) -> Dict:
	"""
	Melakukan training model InceptionV3 dan mengembalikan hasil training
	
	Args:
		data_dir: Path ke folder dataset (berisi subfolder untuk setiap kelas)
		img_size: Ukuran target gambar (tinggi, lebar) dalam pixel. Default (299, 299) untuk InceptionV3
		batch_size: Jumlah gambar per batch. Default 32
		epochs: Jumlah epoch training (1 epoch = 1 kali melihat seluruh training data). Default 5
		learning_rate: Learning rate untuk optimizer. Default 1e-4 (0.0001)
		base_trainable: Apakah base InceptionV3 akan ditraining (fine-tuning)?
		                False = frozen (lebih cepat), True = fine-tuning (lebih lambat)
		results_dir: Folder untuk menyimpan hasil training (model, history, dll)
		seed: Random seed untuk reproducibility
		steps_per_epoch: Jumlah batch per epoch (untuk fast mode). None = gunakan semua data
		val_steps: Jumlah batch untuk validation (untuk fast mode). None = gunakan semua data
		
	Returns:
		Dictionary berisi:
		- model: Model Keras yang sudah ditraining
		- history: History training (loss, accuracy per epoch)
		- train_time_sec: Waktu training dalam detik
		- generators: Dictionary berisi train/val/test generators
		- class_indices: Mapping dari label ke index
		- model_path: Path file model yang disimpan
	"""
	
	# === LANGKAH 1: Buat Folder Results ===
	# Buat folder jika belum ada untuk menyimpan hasil training
	os.makedirs(results_dir, exist_ok=True)

	# === LANGKAH 2: Buat Data Generators ===
	# Membuat generators untuk train/validation/test dengan augmentasi
	train_gen, val_gen, test_gen, class_indices = create_generators(
		data_dir=data_dir,
		img_size=img_size,
		batch_size=batch_size,
		seed=seed,
	)

	# === LANGKAH 3: Bangun Model ===
	# Hitung jumlah kelas dari class_indices
	num_classes = len(class_indices)
	
	# Tentukan ukuran input (tinggi, lebar, channels)
	input_shape = (img_size[0], img_size[1], 3)
	
	# Bangun model InceptionV3 dengan parameter yang ditentukan
	model = build_inception_model(
		num_classes=num_classes,  # Jumlah kelas output
		input_shape=input_shape,  # Ukuran input gambar
		learning_rate=learning_rate,  # Learning rate optimizer
		base_trainable=base_trainable,  # Apakah base model ditraining?
	)

	# === LANGKAH 4: Simpan Informasi Dataset Split ===
	# Catat waktu mulai training
	start = time.time()
	
	# Simpan informasi jumlah sampel di setiap split untuk debugging
	split_info = {
		"train_samples": int(train_gen.samples),  # Jumlah gambar training
		"val_samples": int(val_gen.samples),  # Jumlah gambar validation
		"test_samples": int(test_gen.samples),  # Jumlah gambar testing
		"batch_size": int(batch_size),  # Ukuran batch
	}
	
	# Simpan informasi split ke file JSON
	with open(os.path.join(results_dir, "dataset_split.json"), "w", encoding="utf-8") as f:
		import json
		json.dump(split_info, f, indent=2)

	# === LANGKAH 5: Training Model ===
	# Fit model menggunakan training dan validation generators
	# verbose=1: Tampilkan progress bar saat training
	# steps_per_epoch: Jumlah batch per epoch (None = gunakan semua data)
	# validation_steps: Jumlah batch untuk validation (None = gunakan semua data)
	history = model.fit(
		train_gen,  # Training data generator
		validation_data=val_gen,  # Validation data generator
		epochs=epochs,  # Jumlah epoch training
		verbose=1,  # Tampilkan progress
		steps_per_epoch=steps_per_epoch,  # Batasi jumlah step per epoch (untuk fast mode)
		validation_steps=val_steps,  # Batasi jumlah step validation (untuk fast mode)
	)
	
	# Hitung waktu training yang telah berlalu
	train_time_sec = time.time() - start

	# === LANGKAH 6: Simpan Model ===
	# Simpan model yang sudah ditraining ke file .h5
	model_path = os.path.join(results_dir, "inceptionv3_model.h5")
	model.save(model_path)

	# === LANGKAH 7: Simpan Class Indices ===
	# Simpan mapping dari label ke index untuk inference nanti
	# Contoh: {'buildings': 0, 'forest': 1, 'glacier': 2, ...}
	import json as _json
	with open(os.path.join(results_dir, "class_indices.json"), "w", encoding="utf-8") as f:
		_json.dump(class_indices, f, indent=2)

	# === LANGKAH 8: Return Semua Hasil ===
	# Kembalikan dictionary berisi semua informasi training
	return {
		"model": model,  # Model yang sudah ditraining
		"history": history.history,  # History training (loss, accuracy)
		"train_time_sec": train_time_sec,  # Waktu training dalam detik
		"generators": {
			"train": train_gen,  # Training generator
			"val": val_gen,  # Validation generator
			"test": test_gen,  # Test generator
		},
		"class_indices": class_indices,  # Mapping label ke index
		"model_path": model_path,  # Path file model yang disimpan
	}

