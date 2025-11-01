"""
File: main.py
Tujuan: File utama untuk menjalankan training dan evaluasi model InceptionV3

Proses:
1. Parse command line arguments
2. Training model dengan data augmentation
3. Evaluasi model pada test set
4. Menyimpan hasil training dan evaluasi
5. Menampilkan summary hasil

Usage:
	python main.py --data-dir data --epochs 1 --batch-size 32
	python main.py --fast  # Mode cepat untuk testing
"""

import argparse
import json
import os

from src.train_model import train
from src.evaluate_model import evaluate
from src.utils import format_seconds


def parse_args():
	"""
	Parse command line arguments untuk konfigurasi training dan evaluasi
	
	Arguments:
		--data-dir: Path ke folder dataset (default: 'data')
		--img-size: Ukuran gambar input dalam pixel (tinggi lebar), default: 299 299
		--batch-size: Jumlah gambar per batch (default: 32)
		--epochs: Jumlah epoch training (default: 5)
		--lr: Learning rate untuk optimizer (default: 1e-4 = 0.0001)
		--fine-tune: Flag untuk melakukan fine-tuning (unfreeze base model)
		--results-dir: Folder untuk menyimpan hasil (default: 'results')
		--seed: Random seed untuk reproducibility (default: 42)
		--fast: Mode cepat dengan jumlah step terbatas (untuk testing)
		--steps-per-epoch: Batasi jumlah step per epoch
		--val-steps: Batasi jumlah step untuk validation
		--test-steps: Batasi jumlah step untuk testing
		
	Returns:
		Namespace object berisi semua arguments
	"""
	p = argparse.ArgumentParser(description="Train and evaluate InceptionV3 on image dataset")
	
	# Argumen untuk dataset dan preprocessing
	p.add_argument("--data-dir", default="data", 
				   help="Path ke folder dataset dengan subfolder per kelas")
	p.add_argument("--img-size", type=int, nargs=2, default=(299, 299), 
				   help="Ukuran input gambar (tinggi lebar) dalam pixel")
	
	# Argumen untuk hyperparameters training
	p.add_argument("--batch-size", type=int, default=32, 
				   help="Jumlah gambar per batch")
	p.add_argument("--epochs", type=int, default=5, 
				   help="Jumlah epoch training")
	p.add_argument("--lr", type=float, default=1e-4, 
				   help="Learning rate untuk optimizer Adam")
	p.add_argument("--fine-tune", action="store_true", 
				   help="Unfreeze base InceptionV3 untuk fine-tuning (lebih lambat)")
	
	# Argumen untuk output dan reproducibility
	p.add_argument("--results-dir", default="results", 
				   help="Folder untuk menyimpan model dan hasil evaluasi")
	p.add_argument("--seed", type=int, default=42, 
				   help="Random seed untuk reproducibility")
	
	# Argumen untuk fast mode (testing/debugging)
	p.add_argument("--fast", action="store_true", 
				   help="Mode cepat dengan jumlah step terbatas (untuk testing)")
	p.add_argument("--steps-per-epoch", type=int, default=None, 
				   help="Batasi jumlah step per epoch (None = gunakan semua data)")
	p.add_argument("--val-steps", type=int, default=None, 
				   help="Batasi jumlah step validation per epoch")
	p.add_argument("--test-steps", type=int, default=None, 
				   help="Batasi jumlah step saat evaluasi")
	
	return p.parse_args()


def main():
	"""
	Fungsi utama untuk menjalankan pipeline training dan evaluasi
	
	Alur:
	1. Parse command line arguments
	2. Setup fast mode jika diperlukan
	3. Training model dengan data augmentation
	4. Evaluasi model pada test set
	5. Simpan summary hasil
	6. Tampilkan hasil di console
	"""
	# === LANGKAH 1: Parse Arguments ===
	args = parse_args()
	
	# Buat folder results jika belum ada
	os.makedirs(args.results_dir, exist_ok=True)

	# === LANGKAH 2: Setup Fast Mode ===
	# Fast mode berguna untuk testing cepat tanpa menunggu training penuh
	steps_per_epoch = args.steps_per_epoch
	val_steps = args.val_steps
	test_steps = args.test_steps
	
	if args.fast:
		# Jika fast mode aktif, gunakan default kecil untuk iterasi cepat
		# Nilai ini cukup untuk smoke test tapi tidak representative untuk evaluasi final
		steps_per_epoch = steps_per_epoch or 20  # 20 batch per epoch
		val_steps = val_steps or 10  # 10 batch untuk validation
		test_steps = test_steps or 10  # 10 batch untuk testing

	# === LANGKAH 3: Training Model ===
	print("=== TRAINING MODEL ===")
	print(f"Dataset: {args.data_dir}")
	print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}")
	print(f"Image Size: {args.img_size}")
	print(f"Learning Rate: {args.lr}")
	print(f"Fine-tuning: {'Yes' if args.fine_tune else 'No (frozen base)'}")
	if args.fast:
		print(f"Fast Mode: {steps_per_epoch} steps/epoch, {val_steps} val steps")
	print()
	
	# Panggil fungsi train dari src/train_model.py
	# Mengembalikan model, history, waktu training, generators, dll
	artifacts = train(
		data_dir=args.data_dir,
		img_size=tuple(args.img_size),
		batch_size=args.batch_size,
		epochs=args.epochs,
		learning_rate=args.lr,
		base_trainable=args.fine_tune,
		results_dir=args.results_dir,
		seed=args.seed,
		steps_per_epoch=steps_per_epoch,
		val_steps=val_steps,
	)

	# Tampilkan waktu training dalam format yang mudah dibaca
	print(f"\n✓ Training selesai dalam: {format_seconds(artifacts['train_time_sec'])}")

	# === LANGKAH 4: Evaluasi Model ===
	print("\n=== EVALUATING MODEL ===")
	
	# Panggil fungsi evaluate dari src/evaluate_model.py
	# Menghitung accuracy, precision, recall, f1, AUC, confusion matrix, dll
	metrics = evaluate(
		model=artifacts["model"],
		test_gen=artifacts["generators"]["test"],
		class_indices=artifacts["class_indices"],
		results_dir=args.results_dir,
		test_steps=test_steps,
	)

	# === LANGKAH 5: Simpan Summary ===
	# Kumpulkan metrik penting dalam satu dictionary
	summary = {
		"train_time_sec": artifacts["train_time_sec"],  # Waktu training
		"test_time_sec": metrics.get("test_time_sec"),  # Waktu testing
		"accuracy": metrics.get("accuracy"),  # Akurasi
		"precision_macro": metrics.get("precision_macro"),  # Precision rata-rata
		"recall_macro": metrics.get("recall_macro"),  # Recall rata-rata
		"f1_macro": metrics.get("f1_macro"),  # F1-Score rata-rata
		"auc_macro_ovr": metrics.get("auc_macro_ovr"),  # ROC AUC (One-vs-Rest)
		"model_path": artifacts.get("model_path"),  # Path file model
	}
	
	# Simpan summary ke file JSON
	with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	# === LANGKAH 6: Tampilkan Summary ===
	print("\n" + "="*50)
	print("SUMMARY HASIL TRAINING DAN EVALUASI")
	print("="*50)
	for k, v in summary.items():
		# Format output berdasarkan tipe data
		if isinstance(v, (int, float)):
			# Jika angka, format dengan 4 desimal atau 2 desimal untuk waktu
			if "time" not in k:
				print(f"- {k}: {v:.4f}")
			else:
				print(f"- {k}: {format_seconds(v)}")
		else:
			# Jika string (seperti model_path), tampilkan apa adanya
			print(f"- {k}: {v}")
	print("="*50)
	print(f"\n✓ Semua hasil disimpan di folder: {args.results_dir}")


if __name__ == "__main__":
	# Entry point: Jalankan fungsi main() saat file dieksekusi
	main()

