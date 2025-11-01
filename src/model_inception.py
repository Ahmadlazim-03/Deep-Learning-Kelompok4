"""
File: model_inception.py
Tujuan: Membangun arsitektur model InceptionV3 untuk klasifikasi gambar

Arsitektur:
1. Base Model: InceptionV3 pretrained (ImageNet) - default FROZEN
2. Global Average Pooling: Merangkum feature maps menjadi vector
3. Dropout Layer: Regularisasi untuk mencegah overfitting
4. Dense Layer: Output layer dengan aktivasi softmax untuk klasifikasi multi-class
"""

from typing import Tuple

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam


def build_inception_model(
	num_classes: int,
	input_shape: Tuple[int, int, int] = (299, 299, 3),
	learning_rate: float = 1e-4,
	base_trainable: bool = False,
	dropout_rate: float = 0.3,
) -> Model:
	"""
	Membangun model InceptionV3 dengan custom classification head
	
	Arsitektur Model:
	  Input (299x299x3)
	    ↓
	  InceptionV3 Base (default: Frozen - tidak ditraining)
	    ↓
	  GlobalAveragePooling2D (mengubah feature maps menjadi vector)
	    ↓
	  Dropout (regularisasi - mematikan neuron secara acak)
	    ↓
	  Dense(num_classes, activation='softmax') (output layer)
	
	Args:
		num_classes: Jumlah kelas untuk klasifikasi (contoh: 6 untuk buildings, forest, glacier, mountain, sea, street)
		input_shape: Ukuran input gambar (tinggi, lebar, channels). Default (299, 299, 3) untuk InceptionV3
		learning_rate: Learning rate untuk optimizer Adam. Default 1e-4 = 0.0001
		base_trainable: Apakah base model InceptionV3 akan ditraining ulang (fine-tuning)?
		                False = frozen (lebih cepat), True = fine-tuning (lebih lambat tapi mungkin lebih akurat)
		dropout_rate: Proporsi neuron yang akan dimatikan secara acak (0.3 = 30%)
		
	Returns:
		Model Keras yang sudah di-compile dan siap untuk training
		
	Catatan:
		- Model sudah di-compile dengan optimizer Adam, loss categorical_crossentropy, dan metric accuracy
		- InceptionV3 base default di-freeze (trainable=False) untuk transfer learning
		- Hanya classification head yang ditraining (lebih cepat dan mencegah overfitting)
	"""
	
	# === LANGKAH 1: Load InceptionV3 Pretrained ===
	# include_top=False: Tidak menggunakan fully connected layer bawaan InceptionV3
	# weights='imagenet': Menggunakan bobot yang sudah ditraining pada dataset ImageNet (1000 kelas, 1.2 juta gambar)
	# input_shape: Ukuran input gambar (tinggi, lebar, channels RGB)
	base = InceptionV3(
		include_top=False,  # Buang FC layer asli, kita akan tambah custom head
		weights="imagenet",  # Gunakan pretrained weights untuk transfer learning
		input_shape=input_shape,  # (299, 299, 3) - ukuran standar InceptionV3
	)
	
	# === LANGKAH 2: Freeze/Unfreeze Base Model ===
	# Tentukan apakah layer-layer di base model bisa ditraining atau tidak
	# False (default): Base model frozen - hanya head yang ditraining (transfer learning)
	# True: Base model juga ditraining (fine-tuning - butuh waktu lebih lama)
	base.trainable = base_trainable

	# === LANGKAH 3: Bangun Classification Head ===
	
	# Input layer - menerima gambar dengan ukuran input_shape
	inputs = layers.Input(shape=input_shape)
	
	# Base model InceptionV3 - feature extractor
	# training=False: Menggunakan mode inference (bahkan saat training)
	# Ini penting untuk BatchNormalization layer agar tidak mengupdate statistik
	x = base(inputs, training=False)
	
	# Global Average Pooling 2D
	# Contoh: Jika output base model adalah (8, 8, 2048)
	# GAP akan menghitung rata-rata dari setiap channel: (8*8 values -> 1 value)
	# Output: vector dengan ukuran (2048,)
	# Keuntungan: Lebih sedikit parameter dibanding Flatten, mengurangi overfitting
	x = layers.GlobalAveragePooling2D()(x)
	
	# Dropout layer untuk regularisasi
	# Jika dropout_rate > 0, tambahkan dropout layer
	# Dropout secara acak mematikan neuron saat training untuk mencegah overfitting
	# Contoh: dropout_rate=0.3 berarti 30% neuron dimatikan setiap iterasi
	if dropout_rate and dropout_rate > 0:
		x = layers.Dropout(dropout_rate)(x)
	
	# Dense layer output dengan softmax activation
	# num_classes: jumlah kelas (6 untuk dataset kita)
	# softmax: mengubah output menjadi probabilitas (jumlah = 1.0)
	# Contoh output: [0.05, 0.15, 0.60, 0.10, 0.05, 0.05] -> kelas ke-3 memiliki probabilitas tertinggi
	outputs = layers.Dense(num_classes, activation="softmax")(x)

	# === LANGKAH 4: Buat Model Final ===
	# Gabungkan input dan output menjadi satu model lengkap
	model = Model(inputs, outputs, name="inceptionv3_classifier")
	
	# === LANGKAH 5: Compile Model ===
	# Compile model dengan optimizer, loss function, dan metrics
	model.compile(
		optimizer=Adam(learning_rate=learning_rate),  # Adam optimizer dengan learning rate yang ditentukan
		loss="categorical_crossentropy",  # Loss untuk multi-class classification dengan one-hot encoding
		metrics=["accuracy"],  # Metric yang akan ditampilkan saat training
	)
	
	return model

