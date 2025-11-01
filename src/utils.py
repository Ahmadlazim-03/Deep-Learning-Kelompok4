"""
File: utils.py
Tujuan: Fungsi-fungsi utility yang digunakan di berbagai bagian project

Fungsi:
- format_seconds: Mengkonversi detik menjadi format yang mudah dibaca (jam/menit/detik)
"""


def format_seconds(seconds: float) -> str:
	"""
	Mengkonversi durasi dalam detik menjadi string yang mudah dibaca
	
	Konversi:
	- Jika < 60 detik: tampilkan dalam detik (contoh: "45.32s")
	- Jika < 60 menit: tampilkan dalam menit dan detik (contoh: "5m 23.4s")
	- Jika >= 60 menit: tampilkan dalam jam, menit, dan detik (contoh: "2h 15m 30s")
	
	Args:
		seconds: Durasi dalam detik (float)
		
	Returns:
		String durasi yang terformat (contoh: "1h 23m 45s")
		
	Contoh:
		>>> format_seconds(45.5)
		'45.50s'
		>>> format_seconds(125.3)
		'2m 5.3s'
		>>> format_seconds(3665)
		'1h 1m 5s'
	"""
	# Jika kurang dari 1 menit, tampilkan dalam detik
	if seconds < 60:
		return f"{seconds:.2f}s"
	
	# Konversi ke menit dan detik
	# divmod(a, b) mengembalikan (quotient, remainder)
	# Contoh: divmod(125, 60) = (2, 5) -> 2 menit 5 detik
	m, s = divmod(seconds, 60)
	
	# Jika kurang dari 1 jam, tampilkan dalam menit dan detik
	if m < 60:
		return f"{int(m)}m {s:.1f}s"
	
	# Konversi menit ke jam dan sisa menit
	h, m = divmod(m, 60)
	
	# Tampilkan dalam jam, menit, dan detik
	return f"{int(h)}h {int(m)}m {s:.0f}s"

