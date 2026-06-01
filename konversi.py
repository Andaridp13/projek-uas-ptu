import os
import subprocess
from pathlib import Path

FOLDER_SUARA_PRIBADI = r'D:\semester 6\projek_ptu\dataset_kita'
FOLDER_TUJUAN = r'D:\semester 6\projek_ptu\projek uas ptu wav'
FFMPEG = r'D:\ffmpeg\bin\ffmpeg.exe'

for root, dirs, files in os.walk(FOLDER_SUARA_PRIBADI):
    audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.mp4', '.ogg'))]
    if not audio_files:
        continue

    label = os.path.basename(root)
    target_dir = os.path.join(FOLDER_TUJUAN, label)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Cari nomor terakhir yang sudah ada
    existing = [f for f in os.listdir(target_dir) if f.startswith(label + '_') and f.endswith('.wav')]
    if existing:
        numbers = [int(f.replace(label + '_', '').replace('.wav', '')) for f in existing if f.replace(label + '_', '').replace('.wav', '').isdigit()]
        start = max(numbers) + 1 if numbers else 1
    else:
        start = 1

    print(f"[{label}] {len(audio_files)} file ditemukan, mulai dari nomor {start}")

    for i, file in enumerate(sorted(audio_files), start=start):
        source_path = os.path.join(root, file)
        target_path = os.path.join(target_dir, f"{label}_{i}.wav")

        cmd = [FFMPEG, '-y', '-i', source_path, '-ar', '22050', '-ac', '1', target_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Berhasil: {target_path}")
        else:
            print(f"Gagal: {source_path}")

print("Selesai!")