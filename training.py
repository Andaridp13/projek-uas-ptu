import numpy as np
import os
import librosa
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ========== KONFIGURASI ==========
FOLDER_WAV = r'D:\semester 6\projek_ptu\projek uas ptu wav'   # folder berisi subfolder warna
FOLDER_SIMPAN = r'D:\semester 6\projek_ptu'
SAMPLE_RATE = 22050
DURASI_TRIM = 1.0          # ambil 1 detik setelah onset
ONSET_PRE = 0.1            # ambil 0.1 detik sebelum onset (untuk sedikit konteks)
# =================================

def ekstrak_fitur(audio, sr):
    """
    Ekstrak fitur: MFCC (40) + std MFCC + delta MFCC + chroma + ZCR
    Pemotongan menggunakan onset detection (VAD sederhana)
    """
    audio = librosa.util.normalize(audio)
    
    # Deteksi onset (awal suara)
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, 
                                               hop_length=512, 
                                               backtrack=True,
                                               units='samples')
    if len(onset_frames) > 0:
        onset_sample = onset_frames[0]
        # Ambil sedikit sebelum onset (0.1 detik) untuk konteks
        start = max(0, onset_sample - int(ONSET_PRE * sr))
        end = min(len(audio), start + int(DURASI_TRIM * sr))
        audio_trim = audio[start:end]
        if len(audio_trim) < int(0.3 * sr):
            # Jika terlalu pendek, pakai audio asli
            audio_trim = audio
    else:
        # Jika onset tidak terdeteksi, pakai seluruh audio
        audio_trim = audio
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_trim, sr=sr, n_mfcc=40)
    mfcc_rata = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Delta MFCC
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_rata = np.mean(delta_mfcc, axis=1)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_trim, sr=sr)
    chroma_rata = np.mean(chroma, axis=1)
    
    # Zero Crossing Rate (membantu deteksi konsonan letup seperti /k/)
    zcr = librosa.feature.zero_crossing_rate(y=audio_trim)
    zcr_rata = np.mean(zcr)
    
    # Gabungkan semua fitur
    fitur = np.concatenate([mfcc_rata, mfcc_std, delta_rata, chroma_rata, [zcr_rata]])
    return fitur

def augmentasi_audio(audio, sr):
    """Hasilkan beberapa versi audio untuk augmentasi"""
    versi = [audio]  # asli
    
    # 1. Noise kecil
    noise = np.random.randn(len(audio)) * 0.005
    versi.append(audio + noise)
    
    # 2. Noise sedang
    noise2 = np.random.randn(len(audio)) * 0.015
    versi.append(audio + noise2)
    
    # 3. Pitch shift +2
    versi.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
    
    # 4. Pitch shift -2
    versi.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    
    # 5. Time stretch 0.85
    versi.append(librosa.effects.time_stretch(audio, rate=0.85))
    
    # 6. Time stretch 1.15
    versi.append(librosa.effects.time_stretch(audio, rate=1.15))
    
    # 7. Time shift (geser maju 0.2 detik)
    shift = int(0.2 * sr)
    audio_shift = np.roll(audio, shift)
    audio_shift[:shift] = 0
    versi.append(audio_shift)
    
    # 8. Time shift (geser mundur 0.1 detik)
    shift2 = -int(0.1 * sr)
    audio_shift2 = np.roll(audio, shift2)
    if shift2 < 0:
        audio_shift2[shift2:] = 0
    versi.append(audio_shift2)
    
    return versi

print("Mulai ekstraksi fitur dari dataset...")
data = []
label = []

for nama_warna in os.listdir(FOLDER_WAV):
    folder_path = os.path.join(FOLDER_WAV, nama_warna)
    if not os.path.isdir(folder_path):
        continue
    
    file_count = 0
    for nama_file in os.listdir(folder_path):
        if not nama_file.lower().endswith('.wav'):
            continue
        path_file = os.path.join(folder_path, nama_file)
        try:
            audio, sr = librosa.load(path_file, sr=SAMPLE_RATE)
            versi_audio = augmentasi_audio(audio, sr)
            for audio_versi in versi_audio:
                fitur = ekstrak_fitur(audio_versi, sr)
                data.append(fitur)
                label.append(nama_warna)
            file_count += 1
        except Exception as e:
            print(f"Gagal memproses {path_file}: {e}")
    
    print(f"Warna {nama_warna}: {file_count} file asli -> {len(versi_audio)*file_count} sampel")

data = np.array(data)
label = np.array(label)
print(f"Total sampel data: {data.shape[0]}, dimensi fitur: {data.shape[1]}")

# Encode label
le = LabelEncoder()
label_encoded = le.fit_transform(label)

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data_scaled, label_encoded, test_size=0.2, random_state=42, stratify=label_encoded
)

# Grid Search untuk hyperparameter terbaik (opsional, bisa skip jika ingin cepat)
print("Melakukan tuning hyperparameter SVM...")
param_grid = {'C': [1, 10, 50], 'gamma': ['scale', 'auto', 0.1]}
grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

# Evaluasi
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi: {akurasi*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Simpan model dan preprocessor
with open(os.path.join(FOLDER_SIMPAN, 'model_svm.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(FOLDER_SIMPAN, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
with open(os.path.join(FOLDER_SIMPAN, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("Model, label encoder, dan scaler berhasil disimpan.")