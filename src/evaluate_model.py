"""
File: evaluate_model.py
Tujuan: Melakukan evaluasi model pada test set dan menghasilkan berbagai metrik

Metrik yang dihitung:
1. Accuracy (Akurasi)
2. Precision (Macro & Weighted)
3. Recall (Macro & Weighted)
4. F1-Score (Macro & Weighted)
5. ROC AUC Score (One-vs-Rest)
6. Confusion Matrix
7. Classification Report per kelas
8. ROC Curves per kelas

Output:
- confusion_matrix.png: Visualisasi confusion matrix
- roc/*.png: ROC curve untuk setiap kelas
- metrics.json: Semua metrik evaluasi
- classification_report.json: Laporan klasifikasi detail
"""

import json
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
	roc_auc_score,
	roc_curve,
)
from sklearn.preprocessing import label_binarize


def _ensure_dir(path: str):
	"""
	Membuat direktori jika belum ada
	
	Args:
		path: Path direktori yang akan dibuat
	"""
	os.makedirs(path, exist_ok=True)


def evaluate(
	model,
	test_gen,
	class_indices: Dict[str, int],
	results_dir: str = "results",
	test_steps: int | None = None,
) -> Dict:
	"""
	Melakukan evaluasi model pada test set dengan berbagai metrik
	
	Proses:
	1. Prediksi pada test set dan hitung waktu inferensi
	2. Hitung metrik: Accuracy, Precision, Recall, F1-Score
	3. Generate Confusion Matrix
	4. Hitung ROC AUC Score (One-vs-Rest)
	5. Generate ROC Curves per kelas
	6. Simpan semua metrik dan visualisasi
	
	Args:
		model: Model Keras yang sudah ditraining
		test_gen: Test data generator
		class_indices: Dictionary mapping dari label ke index
		results_dir: Folder untuk menyimpan hasil evaluasi
		test_steps: Jumlah batch untuk testing (None = gunakan semua data)
		
	Returns:
		Dictionary berisi semua metrik evaluasi:
		- accuracy: Akurasi keseluruhan
		- precision_macro/weighted: Precision rata-rata
		- recall_macro/weighted: Recall rata-rata
		- f1_macro/weighted: F1-Score rata-rata
		- auc_macro_ovr: ROC AUC Score (One-vs-Rest)
		- test_time_sec: Waktu testing dalam detik
		- per_class_auc: AUC untuk setiap kelas
	"""
	# Pastikan direktori results ada
	_ensure_dir(results_dir)

	# Dapatkan nama-nama kelas yang sudah diurutkan
	class_names: List[str] = sorted(class_indices.keys())
	n_classes = len(class_names)

	# === LANGKAH 1: Prediksi dengan Timing ===
	# Catat waktu mulai testing
	t0 = time.time()
	
	# Prediksi probabilitas untuk setiap kelas
	# Output: array dengan shape (n_samples, n_classes)
	# Setiap baris berisi probabilitas untuk setiap kelas
	# verbose=1: Tampilkan progress bar
	# steps: Jumlah batch yang akan diprediksi (None = semua data)
	y_prob = model.predict(test_gen, verbose=1, steps=test_steps)
	
	# Hitung waktu testing
	test_time_sec = time.time() - t0

	# === LANGKAH 2: Siapkan Label True dan Predicted ===
	# y_true_idx: Label sebenarnya (ground truth) dalam bentuk index
	y_true_idx = test_gen.classes
	
	# y_pred_idx: Label prediksi (index kelas dengan probabilitas tertinggi)
	# np.argmax mengambil index dengan nilai tertinggi di setiap baris
	y_pred_idx = np.argmax(y_prob, axis=1)

	# Jika menggunakan limited test steps, sesuaikan panjang y_true
	# agar sama dengan jumlah prediksi yang dihasilkan
	if y_prob.shape[0] != len(y_true_idx):
		y_true_idx = y_true_idx[: y_prob.shape[0]]

	# === LANGKAH 3: Hitung Metrik Klasifikasi ===
	
	# Accuracy: Proporsi prediksi yang benar
	# Formula: (jumlah prediksi benar) / (total prediksi)
	acc = accuracy_score(y_true_idx, y_pred_idx)
	
	# Precision, Recall, F1-Score (Macro Average)
	# Macro: Rata-rata dari metrik setiap kelas (semua kelas diperlakukan sama)
	# zero_division=0: Jika ada kelas tanpa prediksi, set metrik = 0
	precision, recall, f1, support = precision_recall_fscore_support(
		y_true_idx, y_pred_idx, average="macro", zero_division=0
	)
	
	# Precision, Recall, F1-Score (Weighted Average)
	# Weighted: Rata-rata tertimbang berdasarkan jumlah sampel di setiap kelas
	precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
		y_true_idx, y_pred_idx, average="weighted", zero_division=0
	)

	# === LANGKAH 4: Generate dan Simpan Confusion Matrix ===
	# Confusion Matrix menunjukkan jumlah prediksi untuk setiap kombinasi (true, predicted)
	# Baris: label sebenarnya, Kolom: label prediksi
	cm = confusion_matrix(y_true_idx, y_pred_idx)
	
	# Visualisasi Confusion Matrix menggunakan heatmap
	plt.figure(figsize=(8, 6))
	sns.heatmap(
		cm, 
		annot=True,  # Tampilkan angka di setiap cell
		fmt="d",  # Format angka sebagai integer
		cmap="Blues",  # Skema warna biru
		xticklabels=class_names,  # Label sumbu X
		yticklabels=class_names  # Label sumbu Y
	)
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.title("Confusion Matrix")
	cm_path = os.path.join(results_dir, "confusion_matrix.png")
	plt.tight_layout()
	plt.savefig(cm_path)
	plt.close()

	# === LANGKAH 5: Hitung ROC AUC Score (One-vs-Rest) ===
	# Binarize label untuk multi-class ROC AUC
	# One-vs-Rest: Setiap kelas vs semua kelas lainnya
	y_true_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))
	
	try:
		# Hitung ROC AUC Score dengan strategi One-vs-Rest
		# macro: Rata-rata AUC dari semua kelas
		auc_macro_ovr = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
	except Exception:
		# Jika gagal (misalnya ada kelas yang tidak muncul di test set)
		auc_macro_ovr = float("nan")

	# === LANGKAH 6: Generate ROC Curves per Kelas ===
	roc_dir = os.path.join(results_dir, "roc")
	_ensure_dir(roc_dir)
	# === LANGKAH 6: Generate ROC Curves per Kelas ===
	roc_dir = os.path.join(results_dir, "roc")
	_ensure_dir(roc_dir)
	
	# Dictionary untuk menyimpan False Positive Rate, True Positive Rate, dan AUC per kelas
	fpr_dict, tpr_dict, auc_dict = {}, {}, {}
	
	# Loop untuk setiap kelas
	for i, cls in enumerate(class_names):
		try:
			# Hitung ROC curve untuk kelas ini
			# fpr: False Positive Rate (sumbu X)
			# tpr: True Positive Rate (sumbu Y)
			# _: Threshold values (tidak digunakan)
			fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
			
			# Hitung AUC (Area Under Curve) untuk kelas ini
			# AUC = 1.0 berarti klasifikasi sempurna
			# AUC = 0.5 berarti random guess
			auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
			
			# Simpan nilai untuk kelas ini
			fpr_dict[cls], tpr_dict[cls], auc_dict[cls] = fpr, tpr, float(auc_i)

			# Plot ROC curve untuk kelas ini
			plt.figure(figsize=(6, 4))
			plt.plot(fpr, tpr, label=f"AUC = {auc_i:.3f}")  # Kurva ROC
			plt.plot([0, 1], [0, 1], "k--")  # Garis diagonal (random classifier)
			plt.xlabel("False Positive Rate")
			plt.ylabel("True Positive Rate")
			plt.title(f"ROC Curve - {cls}")
			plt.legend(loc="lower right")
			plt.tight_layout()
			plt.savefig(os.path.join(roc_dir, f"roc_{cls}.png"))
			plt.close()
		except Exception:
			# Jika gagal (misalnya kelas tidak muncul di test set)
			auc_dict[cls] = float("nan")

	# === LANGKAH 7: Generate Classification Report ===
	# Classification report berisi precision, recall, f1-score untuk setiap kelas
	report = classification_report(
		y_true_idx, 
		y_pred_idx, 
		target_names=class_names,  # Nama kelas
		zero_division=0,  # Handling untuk kelas tanpa prediksi
		output_dict=True  # Output sebagai dictionary
	)

	# === LANGKAH 8: Kumpulkan Semua Metrik ===
	metrics = {
		"accuracy": float(acc),  # Akurasi keseluruhan
		"precision_macro": float(precision),  # Precision rata-rata (macro)
		"recall_macro": float(recall),  # Recall rata-rata (macro)
		"f1_macro": float(f1),  # F1-Score rata-rata (macro)
		"precision_weighted": float(precision_w),  # Precision tertimbang
		"recall_weighted": float(recall_w),  # Recall tertimbang
		"f1_weighted": float(f1_w),  # F1-Score tertimbang
		"auc_macro_ovr": float(auc_macro_ovr) if not np.isnan(auc_macro_ovr) else None,  # ROC AUC (OvR)
		"test_time_sec": float(test_time_sec),  # Waktu testing dalam detik
		"per_class_auc": auc_dict,  # AUC per kelas
	}

	# === LANGKAH 9: Simpan Metrik dan Report ===
	# Simpan metrics.json: Semua metrik evaluasi
	with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	
	# Simpan classification_report.json: Laporan detail per kelas
	with open(os.path.join(results_dir, "classification_report.json"), "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	# Return dictionary metrik untuk digunakan lebih lanjut
	return metrics

